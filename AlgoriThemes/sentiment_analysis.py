"""Survey Insight Studio — BERTopic first, then RoBERTa sentiment per response.

Reuses the existing topic-only BERTopic + Gemini workflow for shared topics,
then adds RoBERTa sentiment, confidence and review flags to each response.

RoBERTa class scores are softmax probabilities, NOT calibrated accuracy estimates.
Responses with low maximum class probability are flagged for review, but keep
one of the original Positive / Neutral / Negative labels.

Dependencies: torch, transformers>=4.40,<5, pandas, numpy, bertopic,
scikit-learn, umap-learn, hdbscan, and the existing gemini_analysis.py module.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
SENTIMENT_CONFIG = {
    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "confidence_threshold": 0.70,
    "batch_size": 8,
    "max_tokens": 512,
    "chunk_stride": 64,
    "random_state": 42,
}


@lru_cache(maxsize=1)
def _load_classifier():
    """Load the transformer once per running Python process."""
    name = SENTIMENT_CONFIG["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    raw_labels = [
        str(model.config.id2label[i]).strip().lower()
        for i in range(model.config.num_labels)
    ]
    label_lookup = {
        "negative": "Negative", "neutral": "Neutral", "positive": "Positive"
    }
    if all(label in label_lookup for label in raw_labels):
        labels = [label_lookup[label] for label in raw_labels]
    elif name == "cardiffnlp/twitter-roberta-base-sentiment-latest" and raw_labels == [
        "label_0", "label_1", "label_2"
    ]:
        labels = ["Negative", "Neutral", "Positive"]
    else:
        raise ValueError(f"Unexpected model labels: {raw_labels}")
    if set(labels) != set(SENTIMENT_ORDER) or len(labels) != 3:
        raise ValueError(f"Expected 3 sentiment classes, got: {labels}")
    return tokenizer, model, device, labels


def predict_sentiment(texts):
    """Classify raw responses using RoBERTa; return one row per input response.

    Output contains only sentiment, confidence_score (0–100), and review_flag.
    Low-confidence responses retain their predicted sentiment and remain in BERTopic.
    """
    texts = [str(text) if pd.notna(text) else "" for text in texts]
    columns = ["sentiment", "confidence_score", "review_flag"]
    if not texts:
        return pd.DataFrame(columns=columns)

    tokenizer, model, device, labels = _load_classifier()
    max_tokens = int(SENTIMENT_CONFIG["max_tokens"])
    stride = int(SENTIMENT_CONFIG["chunk_stride"])
    batch_size = int(SENTIMENT_CONFIG["batch_size"])
    threshold = float(SENTIMENT_CONFIG["confidence_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0 <= stride < max_tokens - 2:
        raise ValueError("chunk_stride must be smaller than max_tokens - 2")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    # Split long responses into overlapping chunks instead of silently dropping
    # everything after the first 512 tokens. Each response receives one score
    # vector, calculated as the mean of its chunk probability vectors.
    all_chunks = []
    chunk_owners = []
    for response_index, response in enumerate(texts):
        encoded = tokenizer(
            response,
            truncation=True,
            max_length=max_tokens,
            stride=stride,
            return_overflowing_tokens=True,
            add_special_tokens=True,
        )
        chunks = encoded["input_ids"]
        # A fast tokenizer returns list[list[int]] with overflow enabled.
        if chunks and isinstance(chunks[0], int):
            chunks = [chunks]
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_owners.append(response_index)

    probability_sums = np.zeros((len(texts), 3), dtype=np.float64)
    chunk_counts = np.zeros(len(texts), dtype=np.int32)

    for start in range(0, len(all_chunks), batch_size):
        ids = all_chunks[start:start + batch_size]
        batch = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            logits = model(**batch).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        for offset, vector in enumerate(probabilities):
            owner = chunk_owners[start + offset]
            probability_sums[owner] += vector
            chunk_counts[owner] += 1

    if np.any(chunk_counts == 0):
        raise RuntimeError("One or more responses were not processed by RoBERTa")

    probabilities = probability_sums / chunk_counts[:, None]
    indices = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    return pd.DataFrame({
        "sentiment": [labels[i] for i in indices],
        "confidence_score": np.round(confidence * 100, 1),
        "review_flag": np.where(confidence < threshold, "Review", "Classified"),
    })



def analyse_sentiment(clean_df, response_column, survey_question, embedding_model,
                      model_df=None, checkpoint=None, save_checkpoint=None):
    """Topic-first combined route: BERTopic once on all eligible responses,
    then RoBERTa on those same responses. Topic IDs/names are shared by all sentiments.

    Short responses are excluded from both analyses and all downstream outputs.
    Low-confidence responses keep their predicted label and remain in summaries.
    """
    from topic_analysis import analyse_topics

    # The BERTopic input is the canonical retained sample for BOTH models.
    # Never reintroduce short responses from the broader preparation frame.
    if model_df is None:
        if "eligible_for_topic_model" not in clean_df.columns:
            raise ValueError("Provide model_df or eligible_for_topic_model")
        model_df = clean_df.loc[clean_df["eligible_for_topic_model"].eq(True)].copy()
    full = model_df.copy().reset_index(drop=True)
    if "_studio_response_id" not in full.columns:
        raise ValueError("Prepared data is missing _studio_response_id")
    if "bert_text" not in full.columns:
        raise ValueError("Prepared data is missing bert_text")
    if full["_studio_response_id"].duplicated().any():
        raise ValueError("Prepared response identifiers must be unique")
    if ("eligible_for_topic_model" in full.columns
            and not full["eligible_for_topic_model"].eq(True).all()):
        raise ValueError("Short responses cannot be included in sentiment analysis")
    model_df = full.copy()
    if len(model_df) < 30:
        raise ValueError("BERTopic requires at least 30 topic-eligible responses")

    # Use the SAME BERTopic + Gemini naming pipeline as the topic-only route.
    # All eligible answers are modelled together; do not split by sentiment.
    topic_results = analyse_topics(
        model_df=model_df, response_column=response_column,
        survey_question=survey_question, embedding_model=embedding_model,
        checkpoint=checkpoint, save_checkpoint=save_checkpoint,
    )
    topic_explorer = topic_results["response_explorer"]
    topic_columns = ["_studio_response_id", "topic", "topic_name", "theme_name", "theme_id"]
    if not set(topic_columns).issubset(topic_explorer.columns):
        raise ValueError("Topic modelling did not return all required assignments")
    if topic_explorer["_studio_response_id"].duplicated().any():
        raise ValueError("BERTopic returned duplicate response identifiers")

    # Classify exactly the BERTopic-eligible responses, using bert_text.
    predictions = predict_sentiment(full["bert_text"].astype(str).tolist())
    full = pd.concat([full, predictions], axis=1)
    # Prevent a merge from creating suffixes if input has old result columns.
    full = full.drop(columns=["topic", "topic_name", "theme_name", "theme_id"], errors="ignore")
    explorer = full.merge(
        topic_explorer[topic_columns], on="_studio_response_id", how="left",
        validate="one_to_one", sort=False,
    )
    explorer["topic"] = explorer["topic"].fillna(-1).astype(int)
    explorer["topic_name"] = explorer["topic_name"].fillna("Unassigned")
    if topic_results.get("themes_generated", True):
        explorer["theme_name"] = explorer["theme_name"].fillna("Unassigned")
        explorer["theme_id"] = explorer["theme_id"].fillna(0).astype(int)
    else:
        explorer["theme_name"] = pd.NA
        explorer["theme_id"] = pd.Series(pd.NA, index=explorer.index, dtype="Int64")
    if len(explorer) != len(full):
        raise AssertionError("Sentiment and topic merge changed the number of responses")

    assigned = explorer.loc[explorer["topic_name"].ne("Unassigned")].copy()
    sentiment_by_topic = (
        assigned.groupby(["topic", "theme_name", "topic_name", "sentiment"], dropna=False)
        .size().reset_index(name="Count")
    )
    if not sentiment_by_topic.empty:
        sentiment_by_topic["Topic Total"] = (
            sentiment_by_topic.groupby("topic")["Count"].transform("sum")
        )
        sentiment_by_topic["Within Topic (%)"] = (
            100 * sentiment_by_topic["Count"] / sentiment_by_topic["Topic Total"]
        ).round(2)

    return {
        **topic_results,
        "response_explorer": explorer,
        "sentiment_df": full,
        "sentiment_by_topic": sentiment_by_topic,
        "analysis_sequence": "topic_then_sentiment",
        "sentiment_model": SENTIMENT_CONFIG["model_name"],
        "sentiment_confidence_threshold": SENTIMENT_CONFIG["confidence_threshold"],
    }
