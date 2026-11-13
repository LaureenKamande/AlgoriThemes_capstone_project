import copy
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from hdbscan import HDBSCAN
def assign_outliers(model, docs, topics, embeddings, threshold=0.0):
    """Move every outlier (-1) to its closest topic, then refresh the keywords."""
    topics = np.array(topics).tolist()
    n_before = int(np.sum(np.array(topics) == -1))

    model = copy.deepcopy(model)          # keeps your original model untouched

    new_topics = model.reduce_outliers(
        docs, topics,
        strategy="embeddings",
        embeddings=embeddings,
        threshold=threshold
    )

    # refresh keywords using the same settings as the original model
    model.update_topics(
        docs,
        topics=new_topics,
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1),
        top_n_words=10
    )

    new_topics = np.array(new_topics)
    n_after = int(np.sum(new_topics == -1))
    return model, new_topics, n_before, n_after

def min_cluster_size_for(n, k=0.4, floor=5):
    """Smallest allowed topic grows with the square root of the number of answers."""
    return max(floor, round(k * np.sqrt(n)))

def representative_answers(docs, labels, embeddings, n=3):
    """The n answers closest to the centre of each topic."""
    out = {}
    for t in sorted(set(labels.tolist())):
        idx = np.where(labels == t)[0]
        centre = embeddings[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(embeddings[idx], centre).ravel()
        out[t] = [docs[i] for i in idx[np.argsort(sims)[::-1][:n]]]
    return out

def run_bertopic(docs, embedding_model, embeddings=None, seed=42, top_n_words=10):
    n = len(docs)
    if n < 30:
        raise ValueError(f"Only {n} answers. At least 30 are needed to find themes.")

    # 1. meaning of each answer (skipped if you already have the embeddings)
    if embeddings is None:
        embeddings = embedding_model.encode(docs, batch_size=64, show_progress_bar=False)

    # 2. minimum topic size from the data size
    mcs = min_cluster_size_for(n)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)

    # 3. fit with automatic merging of similar topics; if nothing is found, relax the size and retry
    while True:
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                          metric="cosine", random_state=seed)
        hdbscan_model = HDBSCAN(min_cluster_size=mcs, min_samples=max(2, mcs // 3),
                                metric="euclidean", cluster_selection_method="eom",
                                prediction_data=True)
        model = BERTopic(embedding_model=embedding_model, umap_model=umap_model,
                         hdbscan_model=hdbscan_model, vectorizer_model=vectorizer,
                         top_n_words=top_n_words, verbose=False)
        topics, _ = model.fit_transform(docs, embeddings)
        topics = np.array(topics)

        if (topics != -1).any() or mcs <= 5:
            break
        mcs = max(5, mcs // 2)

    if not (topics != -1).any():
        raise ValueError("No themes found. The answers may be too short or too varied.")

    # 4. give every outlier to its closest topic (your existing function)
    model, topics, n_out_before, n_out_after = assign_outliers(model, docs, topics, embeddings)

    # 5. topic table for the app
    reps = representative_answers(docs, topics, embeddings)
    topic_ids = sorted(set(topics.tolist()))
    topic_table = pd.DataFrame({
        "Topic": topic_ids,
        "Answers": [int((topics == t).sum()) for t in topic_ids],
        "Share %": [round(100 * (topics == t).mean(), 1) for t in topic_ids],
        "Keywords": [", ".join(w for w, _ in model.get_topic(t)[:6]) for t in topic_ids],
        "Example answers": [" | ".join(reps[t]) for t in topic_ids],
    }).sort_values("Answers", ascending=False).reset_index(drop=True)

    summary = {
        "answers": n,
        "min_cluster_size": mcs,
        "topics_found": len(set(topics.tolist()) - {-1}),
        "outliers_assigned": n_out_before - n_out_after,
        "outliers_left": n_out_after,
    }
    return topics, topic_table, model, summary, embeddings

def sample_for_gemini(
    model_df,
    embeddings,
    response_column,
    sample_fraction=0.25,
    min_per_topic=5,
    representative_fraction=0.50,
    seed=42
):
    rng = np.random.default_rng(seed)
    labels = model_df["topic"].to_numpy()

    topic_ids = sorted(set(labels) - {-1})
    topic_sizes = {
        t: int((labels == t).sum())
        for t in topic_ids
    }

    n_assigned = sum(topic_sizes.values())

    if n_assigned == 0:
        raise ValueError("No assigned topics available.")

    # Target 25% of assigned responses
    target = min(
        n_assigned,
        max(len(topic_ids), round(n_assigned * sample_fraction))
    )

    # Allocate minimum samples where possible
    allocations = {
        t: min(min_per_topic, topic_sizes[t])
        for t in topic_ids
    }

    # If minimum allocations exceed the target, reduce them
    while sum(allocations.values()) > target:
        eligible = [t for t in topic_ids if allocations[t] > 1]
        if not eligible:
            break

        t = max(eligible, key=lambda x: allocations[x])
        allocations[t] -= 1

    # Distribute remaining places approximately proportionally
    remaining_budget = target - sum(allocations.values())

    while remaining_budget > 0:
        eligible = [
            t for t in topic_ids
            if allocations[t] < topic_sizes[t]
        ]

        if not eligible:
            break

        # Prioritize topics furthest below proportional allocation
        t = max(
            eligible,
            key=lambda x: (
                sample_fraction * topic_sizes[x] - allocations[x],
                topic_sizes[x]
            )
        )

        allocations[t] += 1
        remaining_budget -= 1

    sampled_records = []

    for topic_id in topic_ids:
        indices = np.where(labels == topic_id)[0]
        sample_size = allocations[topic_id]

        # Responses closest to the topic centre
        topic_embeddings = embeddings[indices]
        centre = topic_embeddings.mean(axis=0, keepdims=True)

        similarities = cosine_similarity(
            topic_embeddings, centre
        ).ravel()

        ranked_indices = indices[
            np.argsort(similarities)[::-1]
        ]

        n_representative = max(
            1,
            round(sample_size * representative_fraction)
        )

        representative = ranked_indices[:n_representative]

        # Randomly sample the remaining responses
        remaining = np.setdiff1d(indices, representative)

        n_random = sample_size - len(representative)

        random_indices = rng.choice(
            remaining,
            size=n_random,
            replace=False
        )

        chosen = list(representative) + list(random_indices)

        for idx in chosen:
            sampled_records.append({
                "topic_id": int(topic_id),
                "response": str(
                    model_df.iloc[idx][response_column]
                )
            })

    return pd.DataFrame(sampled_records)


def analyse_topics(model_df, response_column, survey_question, embedding_model,
                   checkpoint=None, save_checkpoint=None):
    """Streamlit adapter with resumable in-session checkpoints.

    The original notebook modelling and sampling functions above are unchanged.
    """
    from gemini_analysis import generate_topic_names, generate_themes

    checkpoint = dict(checkpoint or {})
    if "bertopic" in checkpoint:
        saved = checkpoint["bertopic"]
        model_df = saved["model_df"].copy()
        topic_table = saved["topic_table"]
        final_model = saved["model"]
        summary = saved["summary"]
        gemini_samples = saved["gemini_samples"]
        gemini_topic_data = saved["gemini_topic_data"]
    else:
        model_df = model_df.copy().reset_index(drop=True)
        docs = model_df["bert_text"].astype(str).tolist()
        topics, topic_table, final_model, summary, embeddings = run_bertopic(
            docs, embedding_model
        )
        model_df["topic"] = topics
        gemini_samples = sample_for_gemini(
            model_df=model_df, embeddings=embeddings,
            response_column=response_column,
            sample_fraction=0.25, min_per_topic=5,
            representative_fraction=0.50, seed=42
        )
        gemini_topic_data = []
        for _, row in topic_table.iterrows():
            topic_id = int(row["Topic"])
            if topic_id == -1:
                continue
            samples = gemini_samples.loc[
                gemini_samples["topic_id"] == topic_id, "response"
            ].tolist()
            gemini_topic_data.append({
                "topic_id": topic_id,
                "total_responses": int(row["Answers"]),
                "keywords": row["Keywords"],
                "sample_size": len(samples),
                "sample_responses": samples
            })
        checkpoint["bertopic"] = {
            "model_df": model_df.copy(),
            "topic_table": topic_table,
            "model": final_model,
            "summary": summary,
            "gemini_samples": gemini_samples,
            "gemini_topic_data": gemini_topic_data,
        }
        if save_checkpoint is not None:
            save_checkpoint(checkpoint)

    if "topic_names" in checkpoint:
        topic_names = checkpoint["topic_names"]
    else:
        topic_names = generate_topic_names(gemini_topic_data, survey_question)
        checkpoint["topic_names"] = topic_names
        if save_checkpoint is not None:
            save_checkpoint(checkpoint)
    theme_input = []
    for topic in gemini_topic_data:
        topic_id = topic["topic_id"]
        name_info = topic_names.loc[
            topic_names["topic_id"] == topic_id
        ].iloc[0]
        theme_input.append({
            "topic_id": topic_id,
            "topic_name": name_info["topic_name"],
            "topic_description": name_info["description"],
            "total_responses": topic["total_responses"],
            "keywords": topic["keywords"],
            "sample_responses": topic["sample_responses"]
        })
    # Theme grouping is only useful when there are more than 8 distinct
    # BERTopic topics. Count real topics, never the outlier (-1).
    topic_count = len({int(t["topic_id"]) for t in gemini_topic_data
                       if int(t["topic_id"]) != -1})
    themes_generated = topic_count > 8
    if themes_generated:
        if "theme_groups" in checkpoint:
            theme_groups = checkpoint["theme_groups"]
        else:
            theme_groups = generate_themes(theme_input, survey_question)
            checkpoint["theme_groups"] = theme_groups
            if save_checkpoint is not None:
                save_checkpoint(checkpoint)
        theme_mapping = topic_names.merge(
            theme_groups, on="topic_id", validate="one_to_one"
        )
        theme_mapping["theme_id"] = pd.factorize(theme_mapping["theme_name"])[0] + 1
    else:
        # Keep topic metadata, but do NOT fabricate themes from topic names.
        theme_mapping = topic_names.copy()
        theme_mapping["theme_name"] = pd.NA
        theme_mapping["theme_id"] = pd.NA
        checkpoint.pop("theme_groups", None)
    response_explorer = model_df.merge(
        theme_mapping, left_on="topic", right_on="topic_id",
        how="left", validate="many_to_one"
    )
    response_explorer["topic_name"] = response_explorer["topic_name"].fillna("Unassigned")
    if themes_generated:
        response_explorer["theme_name"] = response_explorer["theme_name"].fillna("Unassigned")
        response_explorer["theme_id"] = response_explorer["theme_id"].fillna(0).astype(int)
    else:
        response_explorer["theme_name"] = pd.NA
        response_explorer["theme_id"] = pd.Series(pd.NA, index=response_explorer.index, dtype="Int64")
    return {
        "response_explorer": response_explorer,
        "topic_table": topic_table,
        "topic_names": topic_names,
        "theme_mapping": theme_mapping,
        "themes_generated": themes_generated,
        "topic_count": topic_count,
        "theme_threshold": 8,
        "theme_input": theme_input,
        "gemini_samples": gemini_samples,
        "model_summary": summary,
        "model": final_model,
        "survey_question": survey_question,
        "response_column": response_column,
    }
