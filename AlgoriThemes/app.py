
"""AlgoriThemes — Streamlit application.

Presentation and interface changes only.
The original BERTopic, Gemini and export modules remain separate; RoBERTa provides sentiment labels.
"""

from __future__ import annotations

import io
import hashlib
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from topic_analysis import analyse_topics
from gpu_client import use_modal
from sentiment_analysis import analyse_sentiment
from report_builder import compile_report, report_to_docx
from html_export import build_interactive_html
from recommendations import (
    prepare_recommendation_evidence,
    generate_recommendations,
    recommendations_table,
)
from outputs import (
    create_results_zip,
    dataframe_to_csv,
    dataframe_to_excel,
    get_sentiment_distribution,
    get_sentiment_theme_distribution,
    get_theme_distribution,
)


st.set_page_config(
    page_title="AlgoriThemes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: #321b45;
    }
    [data-testid="stSidebar"] * {
        color: #f9f4fc !important;
    }
    .block-container {
        padding-top: 1.5rem;
        max-width: 1450px;
    }
    h1, h2, h3 {
        letter-spacing: -0.025em;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 12px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Home",
    "Upload Data",
    "Configure",
    "Analyse",
    "Results",
    "Export",
    "About",
]

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]

THEME_COLORS = [
    "#7655B8",
    "#3B8FAD",
    "#D08A43",
    "#4D9B78",
    "#C16B96",
    "#6878C7",
    "#9C785A",
    "#728F45",
]


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():
    defaults = {
        "raw_df": None,
        "dataset_name": None,
        "sheet_name": None,
        "working_df": None,
        "model_df": None,
        "short_df": None,
        "prep_summary": None,
        "response_column": None,
        "id_column": None,
        "survey_question": "",
        "duplicate_policy": "Keep all responses",
        "analysis_type": None,
        "analysis_results": None,
        "report": None,
        "report_evidence": None,
        "recommendations": None,
        "recommendation_evidence": None,
        "recommendation_error": None,
        "quote_bank": None,
        "analysis_error": None,
        "topic_checkpoint": None,
        "topic_checkpoint_id": None,
        "page": "Home",
        "analysis_route_selector": "topic",
        "demo_loaded": False,
        "demo_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_analysis():
    keys = [
        "analysis_type",
        "analysis_results",
        "report",
        "report_evidence",
        "quote_bank",
        "analysis_error",
        "recommendations",
        "recommendation_evidence",
        "recommendation_error",
    ]

    for key in keys:
        st.session_state[key] = None


def clear_preparation():
    keys = [
        "topic_checkpoint",
        "topic_checkpoint_id",
        "working_df",
        "model_df",
        "short_df",
        "prep_summary",
        "response_column",
        "id_column",
    ]

    for key in keys:
        st.session_state[key] = None

    clear_analysis()


def navigate(page):
    st.session_state.page = page
    st.rerun()


@st.cache_resource(
    show_spinner="Loading the sentence embedding model..."
)
def get_embedding_model():
    # Only load the local embedding model for the CPU route.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes, filename, sheet=0):
    extension = filename.lower().rsplit(".", 1)[-1]
    source = io.BytesIO(file_bytes)

    if extension == "csv":
        return pd.read_csv(source)

    if extension in ("xlsx", "xls"):
        return pd.read_excel(source, sheet_name=sheet)

    raise ValueError("Upload a CSV, XLSX or XLS file.")


# ============================================================
# DATA PREPARATION
# ============================================================

def build_prepared_data(
    raw,
    response_column,
    id_column,
    duplicate_policy,
):
    from data_processing import clean_text, clean_for_bertopic

    working = raw.copy()
    original_count = len(working)

    if response_column not in working.columns:
        raise ValueError(
            "The selected response column is no longer present."
        )

    working = working.loc[
        working[response_column].notna()
    ].copy()

    working = working.loc[
        working[response_column]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    blank_removed = original_count - len(working)
    duplicates_removed = 0
    duplicate_respondents_removed = 0
    duplicate_text_removed = 0

    if duplicate_policy == "Remove duplicate response text":
        before = len(working)

        working = working.drop_duplicates(
            subset=[response_column],
            keep="first",
        )

        duplicates_removed = before - len(working)
        duplicate_text_removed = duplicates_removed

    elif (
        duplicate_policy == "Keep first response per respondent ID"
        and id_column
    ):
        before = len(working)

        has_id = (
            working[id_column].notna()
            & working[id_column].astype(str).str.strip().ne("")
        )

        with_id = working.loc[has_id].drop_duplicates(
            subset=[id_column],
            keep="first",
        )

        without_id = working.loc[~has_id]

        working = pd.concat(
            [with_id, without_id]
        ).sort_index()

        duplicates_removed = before - len(working)
        duplicate_respondents_removed = duplicates_removed

    working = working.reset_index(drop=True)

    working["_studio_response_id"] = range(
        1, len(working) + 1
    )

    if working.empty:
        raise ValueError(
            "No nonblank responses remain after preparation."
        )

    original_text = working[response_column].astype(str)

    working["bert_text"] = original_text.map(
        clean_for_bertopic
    )

    working["clean_text"] = original_text.map(clean_text)

    working["word_count"] = (
        working["clean_text"].str.split().str.len()
    )

    working["eligible_for_topic_model"] = (
        working["word_count"] >= 3
    )

    model_df = working.loc[
        working["eligible_for_topic_model"]
    ].copy().reset_index(drop=True)

    short_df = working.loc[
        ~working["eligible_for_topic_model"]
    ].copy().reset_index(drop=True)

    # Short responses are excluded from BOTH routes, not just BERTopic.
    # Retain their count for preparation diagnostics only.
    working = model_df.copy()

    summary = {
        "uploaded_rows": int(original_count),
        "uploaded_columns": int(len(raw.columns)),
        "blank_removed": int(blank_removed),
        "null_removed": int(blank_removed),
        "duplicates_removed": int(duplicates_removed),
        "duplicate_respondents_removed": int(duplicate_respondents_removed),
        "duplicate_text_removed": int(duplicate_text_removed),
        "retained_responses": int(len(working)),
        "topic_eligible_responses": int(len(model_df)),
        "short_responses": int(len(short_df)),
    }

    return working, model_df, short_df, summary


def show_metrics(metrics):
    columns = st.columns(len(metrics))

    for col, (label, value) in zip(columns, metrics):
        with col:
            st.metric(label, value)


# ============================================================
# GUIDED COLUMN SELECTION AND DEMO DATA
# ============================================================

def likely_open_ended_columns(df):
    """Suggest free-text response fields; always allow manual override.

    This is a transparent heuristic, not a trained classifier. It prioritises
    substantive, varied text rather than IDs, categories, or single-word labels.
    """
    import re

    hints = ("comment", "feedback", "response", "answer", "review",
             "suggest", "improv", "experience", "opinion", "explain",
             "reason", "description", "remark", "open", "text", "why",
             "challenge", "concern", "recommend", "what", "how")
    metadata = ("id", "email", "phone", "name", "gender", "age", "date",
                "time", "status", "rating", "score", "country", "county",
                "department", "category", "code", "latitude", "longitude")
    ranked = []
    for column in df.columns:
        series = df[column]
        if not (pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)):
            continue
        nonempty = series.dropna().astype(str).str.strip()
        nonempty = nonempty.loc[nonempty.ne("")]
        if nonempty.empty:
            continue
        sample = nonempty.head(500)
        lengths = sample.str.split().str.len()
        median_words = float(lengths.median())
        substantial_share = float((lengths >= 5).mean())
        unique_share = float(sample.nunique() / len(sample))
        col_name = re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip()
        is_metadata = any(re.search(r"\b" + re.escape(x) + r"\b", col_name)
                          for x in metadata)
        name_hint = any(x in col_name for x in hints)
        if is_metadata and not name_hint:
            continue
        if unique_share < .25:
            continue
        if not ((median_words >= 5 and substantial_share >= .35)
                or (name_hint and median_words >= 3 and substantial_share >= .20)):
            continue
        score = (int(name_hint) * 10 + min(median_words, 30)
                 + substantial_share * 5 + unique_share * 3)
        ranked.append((score, column))
    return [column for _, column in sorted(ranked, key=lambda x: -x[0])]


def select_demo_route(route):
    """Load a real demo dataset using the same preparation as user uploads."""
    from demo_data import get_demo_dataset, get_demo_settings

    # Validate and prepare before touching the existing session, so a
    # missing/misconfigured demo file does not discard the user's work.
    try:
        settings = get_demo_settings(route)
        raw, question = get_demo_dataset(route)
        response_column = settings["response_column"]
        id_column = settings.get("id_column")
        duplicate_policy = settings.get("duplicate_policy", "Keep all responses")
        working, model_df, short_df, summary = build_prepared_data(
            raw, response_column, id_column, duplicate_policy
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        st.session_state.demo_error = f"Unable to load demo: {exc}"
        st.session_state.page = "Home"
        return

    clear_preparation()
    st.session_state.update({
        "raw_df": raw,
        "dataset_name": settings["dataset_name"],
        "sheet_name": settings.get("sheet_name"),
        "survey_question": question,
        "duplicate_policy": duplicate_policy,
        "response_column": response_column,
        "id_column": id_column,
        "working_df": working,
        "model_df": model_df,
        "short_df": short_df,
        "prep_summary": summary,
        "analysis_route_selector": route,
        "page": "Analyse",
        "demo_loaded": True,
        "demo_error": None,
    })


def select_own_data_route(route):
    st.session_state.analysis_route_selector = route
    st.session_state.demo_loaded = False
    st.session_state.demo_error = None
    st.session_state.page = "Upload Data"


# ============================================================
# HOME
# ============================================================


def show_home():
    st.title("AlgoriThemes")
    st.write(
        "Turn open-ended survey responses into clear themes and topics. "
        "Discover recurring topics, understand people's experiences, and turn feedback into insights."
    )

    left, right = st.columns(2, gap="large")

    # ==========================================
    # TOPIC ANALYSIS
    # ==========================================
    with left:
        st.subheader("Explore topics")
        st.write(
            "Find the main themes and topics in your responses. "
            "When there are many topics, group related ones into broader themes."
        )

        st.button(
            "Upload my data",
            key="own_topics",
            type="primary",
            on_click=select_own_data_route,
            args=("topic",),
            use_container_width=True,
        )

        st.button(
            "Try topic analysis demo",
            key="demo_topics",
            type="secondary",
            on_click=select_demo_route,
            args=("topic",),
            use_container_width=True,
        )

    # ==========================================
    # TOPIC + SENTIMENT ANALYSIS
    # ==========================================
    with right:
        st.subheader("Explore topics and sentiment")
        st.write(
            "Discover what people are talking about and compare positive, "
            "neutral, and negative feedback within each topic."
        )

        st.button(
            "Upload my data",
            key="own_sentiment",
            type="primary",
            on_click=select_own_data_route,
            args=("sentiment",),
            use_container_width=True,
        )

        st.button(
            "Try sentiment analysis demo",
            key="demo_sentiment",
            type="secondary",
            on_click=select_demo_route,
            args=("sentiment",),
            use_container_width=True,
        )

    # ==========================================
    # DEMO INFORMATION
    # ==========================================
    st.caption(
        "Demos run the analysis on real research datasets. "
        "No results are precomputed."
    )

    if st.session_state.get("demo_error"):
        st.error(st.session_state.demo_error)

    st.divider()

    # ==========================================
    # LOADED DATASET SUMMARY
    # ==========================================
    if st.session_state.raw_df is not None:
        st.success(
            f"Dataset loaded: {st.session_state.dataset_name}"
        )

        if st.session_state.prep_summary:
            summary = st.session_state.prep_summary

            show_metrics([
                (
                    "Uploaded",
                    f"{summary.get('uploaded_rows', 0):,}"
                ),
                (
                    "Retained",
                    f"{summary.get('retained_responses', 0):,}"
                ),
                (
                    "Short responses dropped",
                    f"{summary.get('short_responses', 0):,}"
                ),
                (
                    "Null values dropped",
                    f"{summary.get('null_removed', summary.get('blank_removed', 0)):,}"
                ),
                (
                    "Duplicate respondents dropped",
                    f"{summary.get('duplicate_respondents_removed', 0):,}"
                ),
            ])

    # ==========================================
    # PRIVACY NOTE
    # ==========================================
    st.info(
        "Privacy note: Remove identifying information before analysing "
        "sensitive survey responses with external AI services."
    )



# ============================================================
# ABOUT — TECHNOLOGY AND METHODOLOGY
# ============================================================

def show_about():
    st.title("About")
    st.write(
        "AlgoriThemes combines topic discovery, sentiment analysis, and "
        "AI-assisted interpretation to help you analyse open-ended survey "
        "responses. This page explains the technology behind the results."
    )

    st.subheader("How the analysis works")
    st.markdown(
        "**Prepare responses** → **Discover topics** → "
        "**Name and describe topics** → **Analyse sentiment (optional)** "
        "→ **Explore and export findings**"
    )
    st.caption(
        "In the combined analysis, topics are identified across all eligible "
        "responses first. Sentiment is then added to each response, allowing "
        "positive, neutral, and negative feedback to be compared within "
        "the same topic."
    )

    st.divider()
    st.subheader("The models behind AlgoriThemes")

    with st.container(border=True):
        st.markdown("### 1. BERTopic — Discovering recurring topics")
        st.write(
            "BERTopic is an unsupervised topic-modelling method. It looks "
            "for responses with similar meanings and groups them into topics "
            "without requiring you to provide categories in advance."
        )
        st.markdown(
            "**Its role in AlgoriThemes**\n"
            "- Converts responses into numerical representations of meaning "
            "using a sentence-embedding model (all-MiniLM-L6-v2).\n"
            "- Uses UMAP and HDBSCAN to find groups of semantically related "
            "responses.\n"
            "- Uses distinctive words and phrases to help describe each "
            "topic.\n"
            "- Assigns responses initially marked as outliers to their "
            "closest topic where possible."
        )
        st.info(
            "Topics are discovered from the data, so their number and size "
            "can vary between datasets. A topic is a useful starting point "
            "for interpretation, not a definitive category."
        )
        st.markdown("[Read about BERTopic](https://maartengr.github.io/BERTopic/)")

    with st.container(border=True):
        st.markdown("### 2. RoBERTa — Understanding sentiment")
        st.write(
            "For the combined analysis option, AlgoriThemes uses a "
            "pretrained RoBERTa sentiment classifier from CardiffNLP "
            "to estimate the overall tone of each response."
        )
        st.markdown(
            "**It assigns one of three labels:** **Positive**, **Neutral**, "
            "or **Negative**. The application also displays the model's "
            "highest class probability as a confidence score."
        )
        st.markdown(
            "**How to read the result**\n"
            "- Sentiment is assigned to the *whole response*, not separately "
            "to each issue mentioned within it.\n"
            "- Responses with a score below **70%** are flagged for review, "
            "but retain their predicted label and remain in the analysis.\n"
            "- A high confidence score is not a guarantee that the "
            "classification is correct."
        )
        st.caption(
            "The classifier was developed for social-media sentiment; "
            "language, domain-specific expressions, sarcasm, and mixed "
            "opinions can affect its predictions on survey responses."
        )
        st.markdown(
            "[View the sentiment model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)"
        )

    with st.container(border=True):
        st.markdown("### 3. Gemini — Making topics easier to interpret")
        st.write(
            "Gemini is used to help translate model-discovered topics into "
            "clear, readable findings. It does not decide the initial "
            "topic clusters or replace the sentiment classifier."
        )
        st.markdown(
            "**Its role in AlgoriThemes**\n"
            "- Suggests a name and short description for each discovered "
            "topic, using topic keywords and sampled responses.\n"
            "- When **9 or more topics** are discovered, helps organise "
            "related topics into broader themes. For **1–8 topics**, "
            "the application presents topics directly.\n"
            "- Supports evidence-based recommendations using the "
            "completed topic findings and response examples."
        )
        st.warning(
            "AI-generated topic names, descriptions, broader themes, and "
            "recommendations should be checked against the original responses."
        )
        st.markdown("[Learn about Gemini](https://ai.google.dev/gemini-api/docs)")

    st.divider()
    st.subheader("What happens to your data?")
    st.write(
        "Before analysis, AlgoriThemes excludes missing or blank answers "
        "and responses with fewer than three words after cleaning. "
        "Duplicate responses or respondent IDs are removed only if you "
        "choose the corresponding option during configuration."
    )
    st.write(
        "Topic discovery and sentiment classification run through the "
        "application's modelling pipeline. To generate topic labels and "
        "recommendations, selected response examples and related topic "
        "information are sent to an external Gemini service. Remove names, "
        "contact details, and other identifying information before analysing "
        "confidential or sensitive data."
    )

    with st.expander("Important limitations and interpretation guidance"):
        st.markdown(
            "- **Topic discovery is exploratory.** Closely related issues "
            "may be grouped together, and a single response may discuss "
            "several issues even when assigned to one main topic.\n"
            "- **Sentiment is response-level.** A mixed response may receive "
            "one overall label, which does not represent sentiment toward "
            "every topic it mentions.\n"
            "- **Confidence is not accuracy.** Review flagged responses "
            "and important quotations in context.\n"
            "- **AI interpretation requires verification.** Check topic "
            "names, descriptions, and recommendations against the evidence.\n"
            "- **Findings reflect the submitted data.** Topic frequencies "
            "alone do not establish population representativeness, "
            "causation, or statistical significance."
        )

    st.caption(
        "AlgoriThemes is designed to support qualitative research and "
        "survey analysis—not to replace researcher judgement."
    )


# ============================================================
# UPLOAD
# ============================================================

def show_upload():
    st.title("Upload Data")

    upload = st.file_uploader(
        "Choose a CSV or Excel dataset",
        type=["csv", "xlsx", "xls"],
        key="dataset_upload",
    )

    if upload is None:
        if st.session_state.raw_df is not None:
            st.info(
                f"Current dataset: "
                f"{st.session_state.dataset_name}"
            )

            st.dataframe(
                st.session_state.raw_df.head(10),
                use_container_width=True,
            )

        return

    contents = upload.getvalue()
    filename = upload.name
    sheet = 0

    if filename.lower().endswith((".xlsx", ".xls")):
        try:
            sheets = pd.ExcelFile(
                io.BytesIO(contents)
            ).sheet_names

            sheet = st.selectbox(
                "Excel worksheet",
                sheets,
                key="selected_sheet",
            )

        except Exception as exc:
            st.error(
                f"Could not read Excel worksheets: {exc}"
            )
            return

    try:
        preview = read_uploaded_file(
            contents,
            filename,
            sheet,
        )

    except Exception as exc:
        st.error(f"Could not read dataset: {exc}")
        return

    show_metrics([
        ("Rows", f"{len(preview):,}"),
        ("Columns", len(preview.columns)),
    ])

    st.dataframe(
        preview.head(15),
        use_container_width=True,
    )

    if st.button("Use this dataset", type="primary"):
        changed = (
            st.session_state.dataset_name != filename
            or st.session_state.sheet_name != sheet
            or st.session_state.raw_df is None
            or not st.session_state.raw_df.equals(preview)
        )

        if changed:
            clear_preparation()

        st.session_state.raw_df = preview.copy()
        st.session_state.dataset_name = filename
        st.session_state.sheet_name = sheet
        st.session_state.demo_loaded = False

        navigate("Configure")


# ============================================================
# CONFIGURE
# ============================================================

def show_configure():
    st.title("Configure Analysis")

    raw = st.session_state.raw_df

    if raw is None:
        st.warning("Upload a dataset first.")

        st.button(
            "Go to Upload Data",
            on_click=navigate,
            args=("Upload Data",),
        )
        return

    columns = list(raw.columns)
    suggested = likely_open_ended_columns(raw)
    other_label = "Other column (choose manually)..."
    current_response = st.session_state.response_column
    response_options = suggested + [other_label]
    initial_choice = (
        current_response if current_response in suggested
        else other_label if current_response in columns
        else suggested[0] if suggested else other_label
    )

    st.caption(
        "Showing likely open-ended response columns based on their content. "
        "If yours is missing, choose 'Other column'."
    )
    response_choice = st.selectbox(
        "Open-ended response column",
        response_options,
        index=response_options.index(initial_choice),
    )
    if response_choice == other_label:
        other_columns = [c for c in columns if c not in suggested]
        if not other_columns:
            other_columns = columns
        response_column = st.selectbox(
            "Select another column",
            other_columns,
            index=(other_columns.index(current_response)
                   if current_response in other_columns else 0),
        )
    else:
        response_column = response_choice

    with st.form("prepare_form"):
        id_options = [
            "(None)"
        ] + [
            c for c in columns
            if c != response_column
        ]

        selected_id = st.session_state.id_column

        id_index = (
            id_options.index(selected_id)
            if selected_id in id_options
            else 0
        )

        id_selection = st.selectbox(
            "Respondent identifier (optional)",
            id_options,
            index=id_index,
        )

        id_column = (
            None
            if id_selection == "(None)"
            else id_selection
        )

        policies = [
            "Keep all responses",
            "Remove duplicate response text",
        ]

        if id_column:
            policies.append(
                "Keep first response per respondent ID"
            )

        default_policy = st.session_state.duplicate_policy

        duplicate_policy = st.selectbox(
            "Duplicate handling",
            policies,
            index=(
                policies.index(default_policy)
                if default_policy in policies
                else 0
            ),
        )

        survey_question = st.text_area(
            "Original open-ended survey question",
            value=st.session_state.survey_question,
            placeholder=(
                "Paste the question respondents "
                "were answering..."
            ),
            height=110,
        )

        submitted = st.form_submit_button(
            "Prepare responses",
            type="primary",
        )

    if submitted:
        if not survey_question.strip():
            st.error(
                "Enter the original survey question."
            )
            return

        try:
            with st.spinner("Preparing responses..."):
                (
                    working,
                    model_df,
                    short_df,
                    summary,
                ) = build_prepared_data(
                    raw,
                    response_column,
                    id_column,
                    duplicate_policy,
                )

        except Exception as exc:
            st.error(f"Preparation failed: {exc}")
            return

        clear_analysis()

        st.session_state.update({
            "working_df": working,
            "model_df": model_df,
            "short_df": short_df,
            "prep_summary": summary,
            "response_column": response_column,
            "id_column": id_column,
            "duplicate_policy": duplicate_policy,
            "survey_question": survey_question.strip(),
        })

        st.success("Responses prepared successfully.")

    summary = st.session_state.prep_summary

    if summary:
        metric_items = [
            ("Uploaded", f"{summary.get('uploaded_rows', len(st.session_state.raw_df)):,}"),
            ("Retained", f"{summary.get('retained_responses', len(st.session_state.working_df) if st.session_state.working_df is not None else 0):,}"),
            ("Short responses dropped", f"{summary.get('short_responses', 0):,}"),
            ("Null values dropped", f"{summary.get('null_removed', summary.get('blank_removed', 0)):,}"),
            ("Duplicate respondents dropped",
             f"{summary.get('duplicate_respondents_removed', 0):,}"),
        ]
        if summary.get("duplicate_text_removed", 0):
            metric_items.append(
                ("Duplicate response texts dropped",
                 f"{summary['duplicate_text_removed']:,}")
            )
        show_metrics(metric_items)

        st.caption(
            "Null values includes empty responses. Responses with fewer "
            "than 3 cleaned words are excluded from both analysis options. "
            "Duplicate respondents are removed only if that option is selected."
        )

        with st.expander("Preview prepared data"):
            st.dataframe(
                st.session_state.working_df.head(20),
                use_container_width=True,
            )

        st.button(
            "Continue to Analyse",
            type="primary",
            on_click=navigate,
            args=("Analyse",),
        )


# ============================================================
# RUN ANALYSIS — ORIGINAL ANALYTICAL FUNCTIONS
# ============================================================

def show_analyse():
    st.title("Run Analysis")

    if st.session_state.working_df is None:
        st.warning(
            "Prepare a dataset before running an analysis."
        )

        st.button(
            "Go to Configure",
            on_click=navigate,
            args=("Configure",),
        )
        return

    analysis_type = st.radio(
        "Choose analysis route",
        ["topic", "sentiment"],
        key="analysis_route_selector",
        format_func=lambda x: (
            "Explore topics" if x == "topic"
            else "Explore topics and sentiment"
        ),
    )

    summary = st.session_state.prep_summary
    if st.session_state.get("demo_loaded"):
        st.info(
            f"Real-data demo: {st.session_state.dataset_name}. "
            "Select Run analysis to generate results; "
            "no results have been precomputed."
        )

    if analysis_type == "topic":
        st.write(
            f"Topic discovery will analyse "
            f"**{summary['topic_eligible_responses']:,}** "
            f"eligible responses."
        )

        if len(st.session_state.model_df) < 30:
            st.warning(
                "Topic discovery requires "
                "at least 30 eligible responses."
            )

    else:
        st.write(
            f"Topics will be discovered from "
            f"**{summary['retained_responses']:,}** retained responses, "
            f"then sentiment will be assessed for those same responses. "
            f"**{summary['short_responses']:,}** short responses were dropped "
            f"during preparation."
        )
        if len(st.session_state.model_df) < 30:
            st.warning("Topic discovery needs at least 30 eligible responses.")
        st.caption(
            "Use sentiment analysis for questions about opinions, attitudes or "
            "experiences. For factual answers or improvement suggestions, "
            "topic modelling alone may be more meaningful."
        )

    st.caption(
        "Selected example responses may be sent to an external AI service "
        "to describe topics and group them into broader themes."
    )

    if st.session_state.analysis_results is not None:
        st.info(
            "An earlier analysis is stored in this session. "
            "Running again will replace it."
        )

    disabled = len(st.session_state.model_df) < 30

    # A checkpoint is valid only for the exact prepared inputs and question.
    checkpoint_id = None
    if analysis_type in ("topic", "sentiment"):
        prepared = st.session_state.model_df
        fingerprint = hashlib.sha256()
        fingerprint.update(pd.util.hash_pandas_object(
            prepared, index=True
        ).values.tobytes())
        fingerprint.update(str(tuple(prepared.columns)).encode("utf-8"))
        fingerprint.update(st.session_state.response_column.encode("utf-8"))
        fingerprint.update(st.session_state.survey_question.encode("utf-8"))
        checkpoint_id = fingerprint.hexdigest()
        if st.session_state.topic_checkpoint_id != checkpoint_id:
            st.session_state.topic_checkpoint = None
            st.session_state.topic_checkpoint_id = checkpoint_id

    saved_steps = st.session_state.topic_checkpoint or {}
    if saved_steps:
        completed = [
            label for key, label in (
                ("bertopic", "BERTopic"),
                ("topic_names", "topic naming"),
                ("theme_groups", "theme grouping"),
            ) if key in saved_steps
        ]
        st.info("Saved in this session: " + ", ".join(completed)
                + ". Running again will resume from the next step.")

    if st.button(
        "Run analysis",
        type="primary",
        disabled=disabled,
    ):
        clear_analysis()

        try:
            with st.spinner(
                "Running the analytical pipeline..."
            ):
                # Do not download/load a local transformer when Modal is enabled.
                embedding_model = None if use_modal() else get_embedding_model()

                if analysis_type == "topic":
                    results = analyse_topics(
                        model_df=st.session_state.model_df,
                        response_column=(
                            st.session_state.response_column
                        ),
                        survey_question=(
                            st.session_state.survey_question
                        ),
                        embedding_model=embedding_model,
                        checkpoint=st.session_state.topic_checkpoint,
                        save_checkpoint=lambda value: st.session_state.update(
                            topic_checkpoint=value
                        ),
                    )

                else:
                    results = analyse_sentiment(
                        clean_df=st.session_state.working_df,
                        model_df=st.session_state.model_df,
                        response_column=st.session_state.response_column,
                        survey_question=st.session_state.survey_question,
                        embedding_model=embedding_model,
                        checkpoint=st.session_state.topic_checkpoint,
                        save_checkpoint=lambda value: st.session_state.update(
                            topic_checkpoint=value
                        ),
                    )

            st.session_state.analysis_type = analysis_type
            st.session_state.analysis_results = results

            # Recommendations are a final stage of the same analysis, not a
            # separate user-triggered workflow. A Gemini failure here must
            # never discard the completed BERTopic / sentiment results.
            try:
                with st.spinner("Preparing evidence-based recommendations..."):
                    recommendation_evidence = prepare_recommendation_evidence(
                        results, analysis_type
                    )
                    recommendations = generate_recommendations(recommendation_evidence)
                st.session_state.recommendation_evidence = recommendation_evidence
                st.session_state.recommendations = recommendations
            except Exception as rec_exc:
                st.session_state.recommendation_error = (
                    f"{type(rec_exc).__name__}: {rec_exc}"
                )
                st.warning(
                    "The analysis completed, but recommendations could not be "
                    "generated: " + st.session_state.recommendation_error
                )

            st.success("Analysis completed successfully.")

        except Exception as exc:
            st.session_state.analysis_error = (
                f"{type(exc).__name__}: {exc}"
            )

            st.error(
                f"Analysis failed: "
                f"{st.session_state.analysis_error}"
            )

    if st.session_state.analysis_results is not None:
        st.button(
            "View Results",
            type="primary",
            on_click=navigate,
            args=("Results",),
        )

    elif st.session_state.analysis_error:
        st.warning(st.session_state.analysis_error)


# ============================================================
# CHART HELPERS
# ============================================================

def theme_color_map(theme_names):
    names = sorted(set(str(n) for n in theme_names))

    return {
        name: THEME_COLORS[i % len(THEME_COLORS)]
        for i, name in enumerate(names)
    }


def percentage_bar_chart(distribution, title):
    data = distribution.copy()

    if data.empty:
        return None

    data["Percentage"] = pd.to_numeric(
        data["Percentage"],
        errors="coerce",
    ).fillna(0)

    data["Count"] = pd.to_numeric(
        data["Count"],
        errors="coerce",
    ).fillna(0).astype(int)

    data = data.sort_values(
        "Percentage",
        ascending=True,
    )

    colors = theme_color_map(data["theme_name"])

    fig = px.bar(
        data,
        x="Percentage",
        y="theme_name",
        orientation="h",
        color="theme_name",
        color_discrete_map=colors,
        text="Percentage",
        custom_data=["Count"],
        title=title,
        labels={
            "Percentage": "Responses (%)",
            "theme_name": "Theme",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "%{x:.1f}% of responses<br>"
            "n=%{customdata[0]}"
            "<extra></extra>"
        ),
        cliponaxis=False,
    )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            range=[
                0,
                max(
                    10,
                    float(data["Percentage"].max()) * 1.20,
                ),
            ],
            ticksuffix="%",
        ),
        yaxis_title=None,
        margin=dict(l=10, r=55, t=55, b=30),
        height=max(350, 48 * len(data) + 130),
    )

    return fig


def topic_distribution_chart(
    explorer,
    denominator,
    title="Topic distribution (%)",
):
    """Topic bars coloured by their parent themes."""

    assigned = explorer.loc[
        explorer["topic_name"].notna()
        & explorer["topic_name"].astype(str).ne("Unassigned")
    ].copy()
    has_themes = assigned["theme_name"].notna().any()
    if has_themes:
        assigned = assigned.loc[assigned["theme_name"].astype(str).ne("Unassigned")]

    if assigned.empty:
        return None

    grouping = (["theme_name"] if has_themes else []) + ["topic_name"]

    if "sentiment_topic_id" in assigned.columns:
        grouping.append("sentiment_topic_id")

    elif "topic" in assigned.columns:
        grouping.append("topic")

    topic_dist = (
        assigned.groupby(
            grouping,
            dropna=False,
        )
        .size()
        .reset_index(name="Count")
    )

    topic_dist["Percentage"] = (
        100 * topic_dist["Count"] / max(denominator, 1)
    )

    topic_dist = topic_dist.sort_values(
        "Percentage",
        ascending=True,
    )

    colors = theme_color_map(
        assigned["theme_name"].dropna().unique() if has_themes
        else assigned["topic_name"].unique()
    )

    fig = px.bar(
        topic_dist,
        x="Percentage",
        y="topic_name",
        orientation="h",
        color="theme_name" if has_themes else "topic_name",
        color_discrete_map=colors,
        text="Percentage",
        custom_data=["Count"],
        title=title,
        labels={
            "Percentage": "Responses (%)",
            "topic_name": "Topic",
            "theme_name": "Theme",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "%{x:.1f}% of responses<br>"
            "n=%{customdata[0]}"
            "<extra></extra>"
        ),
        cliponaxis=False,
    )

    fig.update_layout(
        xaxis=dict(
            range=[
                0,
                max(
                    5,
                    float(topic_dist["Percentage"].max()) * 1.25,
                ),
            ],
            ticksuffix="%",
        ),
        yaxis_title=None,
        legend_title_text="Parent theme" if has_themes else "Topic",
        margin=dict(l=10, r=55, t=55, b=30),
        height=max(420, len(topic_dist) * 43 + 150),
    )

    return fig


def percentage_sentiment_chart(distribution):
    data = distribution.copy()

    fig = px.pie(
        data,
        names="sentiment",
        values="Count",
        hole=0.35,
        title="Sentiment distribution (%)",
        custom_data=["Count"],
        color="sentiment",
        color_discrete_map={
            "Positive": "#4D9B78",
            "Neutral": "#8794A3",
            "Negative": "#C16B6B",
        },
    )

    fig.update_traces(
        textinfo="label+percent",
        textposition="outside",
        hovertemplate=(
            "%{label}<br>"
            "%{percent}<br>"
            "n=%{customdata[0]}"
            "<extra></extra>"
        ),
    )

    return fig


# ============================================================
# RESULTS — THEME DETAIL HELPERS
# ============================================================

def get_assigned_responses(explorer):
    if "topic_name" not in explorer.columns:
        return explorer.iloc[0:0].copy()
    return explorer.loc[
        explorer["topic_name"].notna()
        & explorer["topic_name"].astype(str).ne("Unassigned")
    ].copy()


def get_topic_mapping(results, sentiment=None):
    mapping = results.get(
        "theme_mapping",
        pd.DataFrame(),
    )

    if mapping is None or mapping.empty:
        return pd.DataFrame()

    mapping = mapping.copy()

    if sentiment is not None and "sentiment" in mapping:
        mapping = mapping.loc[
            mapping["sentiment"] == sentiment
        ]

    return mapping


def render_theme_details(
    explorer,
    mapping,
    denominator,
    section_key,
):
    """Display each theme sequentially with its constituent topics."""

    assigned = get_assigned_responses(explorer)

    if assigned.empty:
        st.info("No assigned themes are available.")
        return

    theme_counts = (
        assigned.groupby("theme_name")
        .size()
        .sort_values(ascending=False)
    )

    st.subheader("Explore themes")

    st.caption(
        "Each theme is shown below with its topics, "
        "descriptions and response distributions."
    )

    for theme_index, (theme_name, theme_count) in enumerate(
        theme_counts.items(),
        start=1,
    ):
        theme_rows = assigned.loc[
            assigned["theme_name"] == theme_name
        ].copy()

        theme_percentage = (
            100 * theme_count / max(denominator, 1)
        )

        st.divider()

        st.caption(f"THEME {theme_index}")

        st.subheader(str(theme_name))

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Responses",
                f"{theme_count:,}",
            )

        with b:
            st.metric(
                "% of all responses",
                f"{theme_percentage:.1f}%",
            )

        with c:
            st.metric(
                "Topics",
                int(theme_rows["topic_name"].loc[
                    theme_rows["topic_name"] != "Unassigned"
                ].nunique()),
            )

        topic_id_col = (
            "sentiment_topic_id"
            if "sentiment_topic_id" in theme_rows.columns
            else "topic"
        )

        topic_counts = (
            theme_rows.loc[
                theme_rows["topic_name"] != "Unassigned"
            ]
            .groupby(
                [topic_id_col, "topic_name"],
                dropna=False,
            )
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )

        if topic_counts.empty:
            st.info(
                "No named topics are available "
                "within this theme."
            )
            continue

        topic_counts["Within Theme (%)"] = (
            100 * topic_counts["Count"] / theme_count
        )

        topic_counts["All Responses (%)"] = (
            100 * topic_counts["Count"] / max(denominator, 1)
        )

        st.markdown("**Topic distribution within this theme**")

        fig = px.bar(
            topic_counts.sort_values(
                "Within Theme (%)",
                ascending=True,
            ),
            x="Within Theme (%)",
            y="topic_name",
            orientation="h",
            text="Within Theme (%)",
            custom_data=[
                "Count",
                "All Responses (%)",
            ],
            labels={
                "topic_name": "Topic",
                "Within Theme (%)": "Within theme (%)",
            },
        )

        fig.update_traces(
            marker_color=THEME_COLORS[
                (theme_index - 1) % len(THEME_COLORS)
            ],
            texttemplate="%{text:.1f}%",
            textposition="outside",
            hovertemplate=(
                "%{y}<br>"
                "%{x:.1f}% of this theme<br>"
                "n=%{customdata[0]}<br>"
                "%{customdata[1]:.1f}% of all responses"
                "<extra></extra>"
            ),
            cliponaxis=False,
        )

        fig.update_layout(
            xaxis=dict(
                range=[0, 105],
                ticksuffix="%",
            ),
            yaxis_title=None,
            margin=dict(l=10, r=55, t=40, b=30),
            height=max(
                250,
                len(topic_counts) * 55 + 110,
            ),
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"{section_key}_theme_chart_{theme_index}",
        )

        st.markdown("**Topics and descriptions**")

        for _, topic_row in topic_counts.iterrows():
            topic_id = topic_row[topic_id_col]
            topic_name = topic_row["topic_name"]

            st.markdown(f"#### {topic_name}")

            st.caption(
                f"Topic {topic_id} · "
                f"{int(topic_row['Count']):,} responses · "
                f"{topic_row['Within Theme (%)']:.1f}% "
                f"of this theme"
            )

            description = None

            if not mapping.empty:
                matching = mapping.loc[
                    (mapping["theme_name"] == theme_name)
                    & (mapping["topic_id"] == topic_id)
                ]

                if not matching.empty:
                    for description_column in [
                        "description",
                        "topic_description",
                    ]:
                        if description_column in matching.columns:
                            value = matching.iloc[0][
                                description_column
                            ]

                            if pd.notna(value) and str(value).strip():
                                description = str(value)
                                break

            if description:
                st.write(description)

            else:
                st.caption(
                    "No additional topic description available."
                )

            with st.expander(
                f"View responses — {topic_name}"
            ):
                response_column = (
                    st.session_state.response_column
                )

                examples = theme_rows.loc[
                    theme_rows[topic_id_col] == topic_id
                ]

                if response_column in examples.columns:
                    display_columns = [
                        response_column,
                    ]

                    if (
                        st.session_state.id_column
                        and st.session_state.id_column
                        in examples.columns
                    ):
                        display_columns.insert(
                            0,
                            st.session_state.id_column,
                        )

                    st.dataframe(
                        examples[display_columns],
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.warning(
                        "The original response column "
                        "could not be located."
                    )


def render_topic_details(explorer, mapping, denominator, section_key):
    """Show named topics directly when theme grouping was intentionally skipped."""
    assigned = get_assigned_responses(explorer)
    st.subheader("Explore topics")
    if assigned.empty:
        st.info("No named topics are available.")
        return
    counts = (assigned.groupby(["topic", "topic_name"], dropna=False)
              .size().reset_index(name="Count")
              .sort_values("Count", ascending=False))
    response_col = st.session_state.response_column
    for index, row in enumerate(counts.itertuples(index=False), 1):
        st.divider()
        st.caption(f"TOPIC {index} · BERTopic ID {row.topic}")
        st.subheader(str(row.topic_name))
        st.caption(f"{row.Count:,} responses · {100 * row.Count / max(denominator, 1):.1f}% of all analysed responses")
        matched = mapping.loc[mapping["topic_id"].eq(row.topic)] if not mapping.empty else pd.DataFrame()
        if not matched.empty:
            for field in ("description", "topic_description"):
                if field in matched and pd.notna(matched.iloc[0][field]):
                    st.write(str(matched.iloc[0][field]))
                    break
        subset = assigned.loc[assigned["topic"].eq(row.topic)]
        with st.expander(f"View responses — {row.topic_name} ({len(subset):,})"):
            columns = ([st.session_state.id_column] if st.session_state.id_column
                       and st.session_state.id_column in subset.columns else [])
            columns += [response_col]
            if "sentiment" in subset:
                columns += ["sentiment", "confidence_score", "review_flag"]
            st.dataframe(subset[columns], use_container_width=True, hide_index=True)


# ============================================================
# RESULTS — THEMES PAGE
# ============================================================

def display_theme_section(results, analysis_type):
    explorer = results["response_explorer"]

    if analysis_type == "topic":
        mapping = get_topic_mapping(results)
        if not results.get("themes_generated", True):
            assigned = get_assigned_responses(explorer)
            show_metrics([
                ("Responses analysed", f"{len(explorer):,}"),
                ("Topics identified", len(mapping)),
                ("Themes generated", 0),
            ])
            st.info("Theme grouping skipped: BERTopic identified 8 or fewer topics.")
            st.subheader("Topic distribution")
            fig = topic_distribution_chart(explorer, len(explorer))
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key="topic_only_distribution")
            st.caption("Percentages use all modelled responses as the denominator.")
            render_topic_details(explorer, mapping, len(explorer), "topic")
            if results.get("model_summary"):
                with st.expander("BERTopic model information"):
                    st.json(results["model_summary"])
            return

        theme_count = (
            mapping["theme_name"].nunique()
            if not mapping.empty
            else 0
        )

        topic_count = (
            mapping["topic_id"].nunique()
            if not mapping.empty
            else 0
        )

        show_metrics([
            ("Responses analysed", f"{len(explorer):,}"),
            ("Topics identified", topic_count),
            ("Themes generated", theme_count),
        ])

        st.divider()

        dist = get_theme_distribution(results)

        if dist.empty:
            st.info("No named themes are available.")
            return

        st.subheader("Distribution of findings")

        view = st.radio(
            "Explore distribution by",
            ["Themes", "Topics"],
            horizontal=True,
            key="topic_distribution_view",
        )

        if view == "Themes":
            fig = percentage_bar_chart(
                dist,
                "Theme distribution (%)",
            )

        else:
            fig = topic_distribution_chart(
                explorer,
                len(explorer),
                "Topic distribution (%)",
            )

        if fig is not None:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"main_topic_chart_{view.lower()}",
            )

        st.caption(
            "Percentages use all modelled responses "
            "as the denominator."
        )

        # Descriptive statistics, not AI-generated interpretation.
        leading = dist.sort_values(
            "Count",
            ascending=False,
        ).iloc[0]

        st.markdown("**At a glance**")

        st.write(
            f"The most frequently identified theme is "
            f"**{leading['theme_name']}**, representing "
            f"**{leading['Percentage']:.1f}%** "
            f"of modelled responses "
            f"({int(leading['Count']):,} responses). "
            f"The analysis identified "
            f"**{theme_count} themes** containing "
            f"**{topic_count} named topics**."
        )

        render_theme_details(
            explorer=explorer,
            mapping=mapping,
            denominator=len(explorer),
            section_key="topic",
        )

        summary = results.get("model_summary", {})

        if summary:
            st.divider()

            with st.expander("BERTopic model information"):
                st.json(summary)

    else:
        sentiment_dist = get_sentiment_distribution(results)
        mapping = get_topic_mapping(results)
        review_count = int(explorer["review_flag"].eq("Review").sum())
        show_metrics([
            ("Responses classified", f"{len(explorer):,}"),
            ("Flagged for review", f"{review_count:,}"),
            ("Topics identified", len(mapping) if not mapping.empty else 0),
            ("Themes identified", mapping["theme_name"].nunique() if results.get("themes_generated", True) and not mapping.empty else 0),
        ])
        st.caption(
            "BERTopic identified topics using all eligible responses before "
            "RoBERTa classified sentiment. Confidence is a model probability, "
            "not a validated accuracy estimate. Low-confidence responses "
            "retain their predicted label and are flagged for review."
        )
        st.divider()
        st.subheader("Overall sentiment distribution")
        st.plotly_chart(percentage_sentiment_chart(sentiment_dist),
                        use_container_width=True, key="sentiment_overall_chart")

        assigned = get_assigned_responses(explorer)
        if assigned.empty:
            st.info("No named topics were identified.")
            return
        st.divider()
        st.subheader("Sentiment within each topic")
        st.caption("Percentages use responses assigned to each topic as the denominator. "
                   "Only retained responses appear in the overall sentiment chart. "
                   "Responses unassigned by BERTopic are excluded from topic-level comparisons.")
        dist = results.get("sentiment_by_topic", pd.DataFrame())
        if not dist.empty:
            fig = px.bar(
                dist, x="topic_name", y="Within Topic (%)", color="sentiment",
                color_discrete_map={"Positive":"#4D9B78", "Neutral":"#8794A3",
                                    "Negative":"#C16B6B"},
                category_orders={"sentiment": SENTIMENT_ORDER},
                barmode="stack", hover_data=["Count", "Topic Total"],
                title="Sentiment composition of identified topics (%)",
            )
            fig.update_layout(yaxis_range=[0, 100], xaxis_title=None,
                              yaxis_title="Within topic (%)", height=480,
                              xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True,
                            key="sentiment_by_topic_chart")
            columns = (["theme_name"] if results.get("themes_generated", True) else []) + [
                "topic_name", "sentiment", "Count", "Topic Total", "Within Topic (%)"]
            st.dataframe(
                dist[columns].sort_values(["topic_name", "sentiment"]),
                use_container_width=True, hide_index=True,
            )

        st.subheader("Explore opinions within a topic")
        choices = (assigned[["topic", "topic_name", "theme_name"]]
                   .drop_duplicates().sort_values(["topic_name"]))
        ids = choices["topic"].astype(int).tolist()
        names = {int(r.topic): (f"{r.theme_name} / {r.topic_name}"
                               if results.get("themes_generated", True)
                               else str(r.topic_name))
                 for r in choices.itertuples(index=False)}
        selected_id = st.selectbox("Select a topic", ids,
                                   format_func=lambda x: names[x],
                                   key="sentiment_topic_selected")
        topic_rows = assigned.loc[assigned["topic"] == selected_id]
        counts = topic_rows["sentiment"].value_counts()
        st.write(f"**{len(topic_rows):,} responses** in this topic")
        for sentiment in SENTIMENT_ORDER:
            subset = topic_rows.loc[topic_rows["sentiment"] == sentiment]
            pct = 100 * len(subset) / max(len(topic_rows), 1)
            with st.expander(f"{sentiment}: {len(subset):,} responses ({pct:.1f}%)"):
                if subset.empty:
                    st.caption("No responses with this sentiment in this topic.")
                else:
                    columns = ([st.session_state.id_column]
                               if st.session_state.id_column
                               and st.session_state.id_column in subset.columns else [])
                    columns += [st.session_state.response_column,
                                "confidence_score", "review_flag"]
                    st.dataframe(subset[columns], use_container_width=True,
                                 hide_index=True)
        st.divider()
        if results.get("themes_generated", True):
            render_theme_details(explorer, mapping, len(explorer), "topic_sentiment")
        else:
            st.info("Theme grouping skipped: BERTopic identified 8 or fewer topics.")
            render_topic_details(explorer, mapping, len(explorer), "topic_sentiment")
        if results.get("model_summary"):
            with st.expander("BERTopic model information"):
                st.json(results["model_summary"])


# ============================================================
# RESULTS — RESPONSE EXPLORER
# ============================================================

def build_coded_outputs(results, analysis_type):
    """Create reusable response-level codes and a one-row-per-theme dictionary."""
    explorer = results["response_explorer"].copy()
    # The modelled sample is the canonical export sample: short responses
    # must not reappear in coded CSV, Excel or standalone HTML.
    original = st.session_state.model_df
    if original is None:
        raise ValueError("Prepared model data is not available in this session.")

    # Use the original uploaded column names, values and order. Prepared rows
    # retain a unique studio ID, so joins do not rely on duplicate response text.
    source_columns = list(st.session_state.raw_df.columns)
    base = original[["_studio_response_id"] + source_columns].copy()
    id_col = "_studio_response_id"
    if id_col not in explorer.columns:
        raise ValueError("Analysis results are missing the response identifier.")
    if explorer[id_col].duplicated().any():
        raise ValueError("Analysis results contain duplicate response identifiers.")

    # BERTopic topic IDs are shared across all sentiments.
    topic_col = (
        "sentiment_topic_id" if analysis_type == "sentiment"
        and "sentiment_topic_id" in explorer.columns else "topic"
    )
    if topic_col not in explorer.columns:
        raise ValueError("Analysis results are missing BERTopic topic IDs.")
    themes_generated = results.get("themes_generated", True)
    required = ["topic_name", topic_col] + (["theme_name"] if themes_generated else [])
    missing = [c for c in required if c not in explorer.columns]
    if missing:
        raise ValueError(f"Analysis results are missing columns: {missing}")

    fields = [id_col, "topic_name", topic_col] + (["theme_name"] if themes_generated else [])
    if analysis_type == "sentiment" and "sentiment" in explorer.columns:
        fields.extend(["sentiment", "confidence_score", "review_flag"])
        if not all(field in explorer.columns for field in fields):
            raise ValueError("Sentiment analysis results are missing confidence or review fields.")
    codes = explorer[fields].copy()
    if themes_generated:
        codes["theme_name"] = codes["theme_name"].fillna("Unassigned")
    codes["topic_name"] = codes["topic_name"].fillna("Unassigned")
    codes["topic_code"] = pd.to_numeric(codes[topic_col], errors="coerce").astype("Int64")
    codes.loc[codes["topic_name"].eq("Unassigned"), "topic_code"] = -1

    if themes_generated:
        named = codes.loc[codes["theme_name"].ne("Unassigned")].copy()
        unique_themes = (named[["theme_name"]].drop_duplicates()
                         .sort_values("theme_name").reset_index(drop=True))
        unique_themes["theme_code"] = range(len(unique_themes))
        codes = codes.merge(unique_themes, on="theme_name", how="left", validate="many_to_one")
        codes["theme_code"] = codes["theme_code"].astype("Int64")
    codes = codes.rename(columns={
        "theme_name": "Theme Name",
        "theme_code": "Theme Code",
        "topic_name": "Topic Name",
        "topic_code": "Topic Code",
        "sentiment": "Sentiment",
        "confidence_score": "Confidence Score (%)",
        "review_flag": "Review Flag",
    })
    output_fields = [id_col] + (["Theme Name", "Theme Code"] if themes_generated else []) + ["Topic Name", "Topic Code"]
    if "Sentiment" in codes.columns:
        output_fields.extend(["Sentiment", "Confidence Score (%)", "Review Flag"])
    coded = base.merge(codes[output_fields], on=id_col, how="left", validate="one_to_one")
    if themes_generated:
        coded["Theme Name"] = coded["Theme Name"].fillna("Unassigned")
    coded["Topic Name"] = coded["Topic Name"].fillna("Unassigned")
    coded["Topic Code"] = coded["Topic Code"].fillna(-1).astype("Int64")
    # The studio identifier is internal; never add it to the original data.
    coded = coded.drop(columns=[id_col])
    added = (["Theme Code", "Theme Name"] if themes_generated else []) + ["Topic Code", "Topic Name"]
    if "Sentiment" in coded.columns:
        added.extend(["Sentiment", "Confidence Score (%)", "Review Flag"])
    # Place the new coding columns directly after the analysed survey column.
    response_column = st.session_state.response_column
    if response_column in source_columns:
        position = source_columns.index(response_column) + 1
        ordered = source_columns[:position] + added + source_columns[position:]
    else:
        ordered = source_columns + added
    coded = coded[ordered]

    if themes_generated:
        dictionary_rows = []
        assigned = codes.loc[codes["Theme Code"].notna()].copy()
        for theme_code, group in assigned.groupby("Theme Code", sort=True):
            distinct = (group[["Topic Code", "Topic Name"]].drop_duplicates()
                        .sort_values(["Topic Code", "Topic Name"]))
            dictionary_rows.append({
                "Theme Code": int(theme_code),
                "Theme Name": str(group["Theme Name"].iloc[0]),
                "Topic Codes Under Theme": ", ".join(str(int(v)) for v in distinct["Topic Code"]),
                "Topic Names Under Theme": ", ".join(distinct["Topic Name"].astype(str)),
            })
        dictionary = pd.DataFrame(dictionary_rows, columns=[
            "Theme Code", "Theme Name", "Topic Codes Under Theme", "Topic Names Under Theme"])
    else:
        dictionary = (codes.loc[codes["Topic Name"].ne("Unassigned"),
                                ["Topic Code", "Topic Name"]]
                      .drop_duplicates().sort_values("Topic Code").reset_index(drop=True))
        # Topic code/name are the complete coding dictionary when no themes exist.
    return coded, dictionary


def coded_workbook(coded, dictionary):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        coded.to_excel(writer, sheet_name="Coded Responses", index=False)
        dictionary.to_excel(writer, sheet_name="Coding Dictionary", index=False)
    return buffer.getvalue()


def display_explorer(results, analysis_type):
    st.subheader("Response Explorer")
    st.write("Explore and export your original survey data with assigned topic codes" +
             (" and theme codes." if results.get("themes_generated", True) else "."))
    coded, dictionary = build_coded_outputs(results, analysis_type)
    visible = coded.copy()

    if "Sentiment" in visible.columns:
        choices = [v for v in SENTIMENT_ORDER if v in visible["Sentiment"].dropna().unique()]
        chosen = st.multiselect("Sentiment", choices, default=choices)
        visible = visible.loc[visible["Sentiment"].isin(chosen)]
        review_options = ["All responses", "Classified", "Review"]
        review_choice = st.selectbox("Review flag", review_options)
        if review_choice != "All responses":
            visible = visible.loc[visible["Review Flag"].eq(review_choice)]
        threshold = 100 * results.get("sentiment_confidence_threshold", 0.70)
        st.caption(
            "Confidence Score (%) is the model's highest class probability, "
            "not a validated accuracy estimate. 'Review' flags predictions "
            f"below the configured {threshold:g}% threshold; they remain in the analysis."
        )

    if results.get("themes_generated", True):
        themes = sorted(visible["Theme Name"].dropna().astype(str).unique())
        selected_theme = st.selectbox("Filter by theme", ["All themes"] + themes)
        if selected_theme != "All themes":
            visible = visible.loc[visible["Theme Name"] == selected_theme]

    topics = sorted(visible["Topic Name"].dropna().astype(str).unique())
    selected_topic = st.selectbox("Filter by topic", ["All topics"] + topics)
    if selected_topic != "All topics":
        visible = visible.loc[visible["Topic Name"] == selected_topic]

    response_column = st.session_state.response_column
    search = st.text_input("Search original responses", placeholder="Type a word or phrase...")
    if search and response_column in visible.columns:
        visible = visible.loc[visible[response_column].astype(str).str.contains(
            search, case=False, na=False, regex=False
        )]

    st.caption(f"Showing {len(visible):,} matching responses")
    st.dataframe(visible, use_container_width=True, hide_index=True)
    st.download_button("Download complete coded responses (CSV)",
                       dataframe_to_csv(coded), "coded_responses.csv", "text/csv")

    st.subheader("Coding Dictionary")
    st.caption("One row per theme with its constituent topics." if results.get("themes_generated", True)
               else "One row per identified BERTopic topic; theme grouping was skipped.")
    st.dataframe(dictionary, use_container_width=True, hide_index=True)
    st.download_button("Download coding dictionary (CSV)",
                       dataframe_to_csv(dictionary), "coding_dictionary.csv", "text/csv")
    st.download_button("Download both tables (Excel)",
                       coded_workbook(coded, dictionary), "coded_survey_data.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ============================================================
# RESULTS — REPORT
# ============================================================

def display_report(results, analysis_type):
    st.subheader("Survey Analysis Report")
    st.caption("Compiled from named topics, original responses and recommendations" +
               (" and broader themes." if results.get("themes_generated", True) else ".") +
               " No additional Gemini requests.")
    try:
        report = compile_report(
            results, analysis_type,
            survey_question=st.session_state.survey_question,
            recommendations=st.session_state.recommendations,
            recommendation_evidence=st.session_state.recommendation_evidence,
        )
    except Exception as exc:
        st.error(f"Unable to compile report: {type(exc).__name__}: {exc}")
        return

    st.title("Survey Analysis Report")
    if report["question"]:
        st.markdown(f"**Survey question:** {report['question']}")
    st.caption("Sentiment and thematic analysis" if analysis_type == "sentiment"
               else "Thematic analysis")

    st.markdown("## 1. Executive Summary")
    level = report.get("group_label", "theme")
    st.write(f"The analysis included {report['total']:,} responses. "
             f"{report['assigned']:,} were assigned to named topics, "
             f"while {report['unassigned']:,} were unassigned. "
             f"The findings comprise {len(report['themes'])} {level} groups "
             f"and {len(report['recommendations'])} recommendations.")
    if report["themes"]:
        st.write(f"Most frequently represented {level}s: " + ", ".join(
            theme["name"] for theme in report["themes"][:3]) + ".")

    st.markdown("## 2. Overview of Findings")
    if report["themes"]:
        overview = pd.DataFrame([{
            ("Theme" if report.get("themes_generated", True) else "Topic"): t["name"],
            "Sentiment": t["sentiment"] or "—",
            "Responses": t["count"], "% of all responses": round(t["percentage"], 1)
        } for t in report["themes"]])
        if analysis_type != "sentiment":
            overview = overview.drop(columns=["Sentiment"])
        st.dataframe(overview, use_container_width=True, hide_index=True)
    else:
        st.info("No named topics were identified.")

    st.markdown("## 3. Key Findings")
    for index, theme in enumerate(report["themes"], 1):
        name = theme["name"] + (f" ({theme['sentiment']})" if theme["sentiment"] else "")
        st.markdown(f"### 3.{index} {name}")
        st.caption(f"{theme['count']:,} responses · {theme['percentage']:.1f}% of all responses")
        for topic in theme["topics"]:
            if report.get("themes_generated", True):
                st.markdown(f"**{topic['name']}** — {topic['count']:,} responses "
                            f"({topic['percentage']:.1f}% of all responses)")
            if topic["description"]:
                st.write(topic["description"])
            if analysis_type == "sentiment":
                breakdown = topic.get("sentiment_counts", {})
                if breakdown:
                    st.caption("Sentiment within topic: " + "; ".join(
                        f"{label}: {breakdown.get(label, 0)} "
                        f"({100 * breakdown.get(label, 0) / max(topic['count'], 1):.1f}%)"
                        for label in SENTIMENT_ORDER
                    ))
                for label, examples in topic.get("examples_by_sentiment", {}).items():
                    if examples:
                        with st.expander(f"{label} example responses — {topic['name']}"):
                            for example in examples:
                                st.write("• " + example)
        if theme["quotes"]:
            st.markdown("**Illustrative responses**")
            for quote in theme["quotes"]:
                st.markdown("> " + quote.replace("\n", " "))

    st.markdown("## 4. Recommendations")
    if report["recommendations"]:
        for index, rec in enumerate(report["recommendations"], 1):
            st.markdown(f"### 4.{index} {rec['title']}")
            st.markdown(f"**Priority:** {rec['priority']}")
            st.write(rec["action"])
            st.write("**Rationale:** " + rec["rationale"])
            st.write("**Priority rationale:** " + rec["priority_reason"])
            if rec["sources"]:
                st.caption("Supporting topics: " + "; ".join(
                    (f"{t['topic_name']} ({t['theme_name']})" if t.get("theme_name")
                     else t['topic_name']) for t in rec["sources"]))
    else:
        st.info("Recommendations are unavailable for this analysis.")

    st.caption("Interpretation: Percentages use all analysed responses as the "
               "denominator. Topic assignments are model-derived and may include "
               "unassigned responses. Findings are descriptive, not evidence of "
               "statistical significance or population representativeness.")
    try:
        st.download_button(
            "Download Word report", report_to_docx(report),
            file_name="algorithemes_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary", key="download_compiled_report",
        )
    except Exception as exc:
        st.warning(f"Word export unavailable: {exc}")


# ============================================================
# RESULTS — RECOMMENDATIONS
# ============================================================

def display_recommendations(results, analysis_type):
    st.subheader("Recommendations")
    st.caption(
        "Actions based on the topics, response frequencies and any broader themes "
        "already identified in this analysis."
    )
    recommendations = st.session_state.recommendations
    evidence = st.session_state.recommendation_evidence
    if not recommendations or not evidence:
        if st.session_state.recommendation_error:
            st.warning(
                "Recommendations were unavailable when this analysis ran. "
                "Your completed findings are safe. "
                + st.session_state.recommendation_error
            )
        else:
            st.info("No recommendations are available for this analysis.")
        return

    high = sum(r["priority"] == "High" for r in recommendations)
    covered = {
        topic["theme_name"] if results.get("themes_generated", True) else topic["topic_name"]
        for r in recommendations
        for key in r["supporting_evidence_ids"]
        for topic in evidence["topics"] if topic["evidence_id"] == key
    }
    c1, c2, c3 = st.columns(3)
    c1.metric("Recommendations", len(recommendations))
    c2.metric("High priority", high)
    c3.metric("Themes addressed" if results.get("themes_generated", True) else "Topics addressed", len(covered))
    lookup = {t["evidence_id"]: t for t in evidence["topics"]}
    for index, rec in enumerate(recommendations, start=1):
        st.divider()
        st.markdown(f"### {index}. {rec['title']}")
        st.markdown(f"**Priority:** {rec['priority']}")
        st.write(rec["action"])
        st.markdown("**Evidence and rationale**")
        st.write(rec["rationale"])
        st.caption("Priority rationale: " + rec["priority_reason"])
        linked = [lookup[key] for key in rec["supporting_evidence_ids"]]
        st.markdown("**Supporting topics**")
        for t in linked:
            sentiment = f"{t['sentiment']} · " if t['sentiment'] else ""
            st.write(
                f"{sentiment}{(str(t['theme_name']) + ' / ') if t.get('theme_name') else ''}{t['topic_name']} "
                f"(Topic {t['topic_code']}; {t['response_count']:,} responses; "
                f"{t['percentage_of_all_responses']:.1f}% of all responses)"
            )
        with st.expander("View supporting original responses"):
            for t in linked:
                st.markdown(f"**{t['topic_name']}**")
                for response in t["sample_responses"]:
                    st.write("• " + response)
    st.divider()
    st.download_button(
        "Download recommendations — CSV",
        dataframe_to_csv(recommendations_table(recommendations, evidence)),
        file_name="recommendations.csv",
        mime="text/csv",
        key="download_recommendations_results",
    )


# ============================================================
# RESULTS — MAIN PAGE
# ============================================================

def show_results():
    st.title("Results")

    results = st.session_state.analysis_results

    if results is None:
        st.warning("Run an analysis first.")

        st.button(
            "Go to Analyse",
            on_click=navigate,
            args=("Analyse",),
        )
        return

    analysis_type = st.session_state.analysis_type

    st.caption(
        "Topic modelling"
        if analysis_type == "topic"
        else "Topic-first sentiment analysis"
    )

    # Download the completed interactive results directly from the Results page.
    try:
        coded_html, dictionary_html = build_coded_outputs(results, analysis_type)
        standalone_html = build_interactive_html(
            results=results,
            analysis_type=analysis_type,
            coded=coded_html,
            dictionary=dictionary_html,
            recommendations=st.session_state.recommendations,
            recommendation_evidence=st.session_state.recommendation_evidence,
            survey_question=st.session_state.survey_question,
            response_column=st.session_state.response_column,
            id_column=st.session_state.id_column,
        )
        st.download_button(
            "Download Interactive HTML — Complete Results",
            data=standalone_html,
            file_name="algorithemes_results.html",
            mime="text/html",
            key="download_html_from_results",
        )
    except Exception as exc:
        st.warning(f"Interactive HTML export unavailable: {exc}")

    # The first tab reflects whether broader theme grouping was performed.
    tabs = st.tabs([
        "Themes" if results.get("themes_generated", True) else "Topics",
        "Response Explorer",
        "Recommendations",
        "Report",
    ])

    with tabs[0]:
        display_theme_section(
            results,
            analysis_type,
        )

    with tabs[1]:
        display_explorer(
            results,
            analysis_type,
        )

    with tabs[2]:
        display_recommendations(results, analysis_type)

    with tabs[3]:
        display_report(
            results,
            analysis_type,
        )


# ============================================================
# EXPORT — EXISTING FUNCTIONALITY
# ============================================================

def show_export():
    st.title("Export")
    results = st.session_state.analysis_results
    if results is None:
        st.warning("Complete an analysis before exporting.")
        return

    analysis_type = st.session_state.analysis_type
    coded, dictionary = build_coded_outputs(results, analysis_type)
    st.write("Download the coded survey dataset and its coding dictionary.")

    st.download_button("Coded responses — CSV", dataframe_to_csv(coded),
                       "coded_responses.csv", "text/csv", use_container_width=True)
    st.download_button("Coding dictionary — CSV", dataframe_to_csv(dictionary),
                       "coding_dictionary.csv", "text/csv", use_container_width=True)
    st.download_button("Coded responses + coding dictionary — Excel",
                       coded_workbook(coded, dictionary), "coded_survey_data.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

    st.divider()
    st.subheader("Interactive HTML — complete results")
    st.caption("Open the downloaded HTML in your browser to explore charts, topics, coded responses and recommendations without Streamlit or an internet connection.")
    try:
        html_bytes = build_interactive_html(
            results=results,
            analysis_type=analysis_type,
            coded=coded,
            dictionary=dictionary,
            recommendations=st.session_state.recommendations,
            recommendation_evidence=st.session_state.recommendation_evidence,
            survey_question=st.session_state.survey_question,
            response_column=st.session_state.response_column,
            id_column=st.session_state.id_column,
        )
        st.download_button(
            "Download Interactive HTML — complete analysis",
            data=html_bytes,
            file_name="algorithemes_results.html",
            mime="text/html",
            use_container_width=True,
            key="download_interactive_html",
        )
    except Exception as exc:
        st.error(f"Interactive HTML export unavailable: {exc}")

    try:
        compiled = compile_report(
            results, analysis_type,
            survey_question=st.session_state.survey_question,
            recommendations=st.session_state.recommendations,
            recommendation_evidence=st.session_state.recommendation_evidence,
        )
        st.download_button(
            "Analysis report — Word", report_to_docx(compiled),
            "algorithemes_report.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"Report export unavailable: {exc}")


# ============================================================
# APPLICATION NAVIGATION
# ============================================================

def main():
    initialize_state()

    with st.sidebar:
        st.title("AlgoriThemes")
        st.caption("Open-ended survey analysis")

        current = (
            st.session_state.page
            if st.session_state.page in PAGES
            else "Home"
        )

        selected = st.radio(
            "Navigation",
            PAGES,
            index=PAGES.index(current),
        )

        st.session_state.page = selected

        st.divider()

        if st.session_state.dataset_name:
            st.caption(
                f"Dataset: "
                f"{st.session_state.dataset_name}"
            )

        if st.session_state.analysis_type:
            st.caption(
                f"Latest analysis: "
                f"{st.session_state.analysis_type.title()}"
            )

    pages = {
        "Home": show_home,
        "Upload Data": show_upload,
        "Configure": show_configure,
        "Analyse": show_analyse,
        "Results": show_results,
        "Export": show_export,
        "About": show_about,
    }

    pages[st.session_state.page]()


if __name__ == "__main__":
    main()
