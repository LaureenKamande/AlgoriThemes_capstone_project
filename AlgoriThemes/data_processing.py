
import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ============================================================
# TEXT PREPROCESSING
# ============================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Preprocessing from the research notebook."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"https?://\S+", " ", text)

    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [w for w in words if len(w) > 2]

    return " ".join(words)


def clean_for_bertopic(text):
    """Minimal preprocessing used for BERTopic."""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# FILE LOADING
# ============================================================

def get_sheet_names(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return ["CSV Data"]

    uploaded_file.seek(0)

    with pd.ExcelFile(uploaded_file) as excel:
        return excel.sheet_names


def load_data(uploaded_file, sheet_name=0):
    uploaded_file.seek(0)
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx"):
        return pd.read_excel(
            uploaded_file,
            sheet_name=sheet_name
        )

    raise ValueError("Please upload a CSV or XLSX file.")


# ============================================================
# SHORT RESPONSE HANDLING
# ============================================================

def examine_short_responses(
    df,
    text_col="clean_text",
    min_words=3
):
    df = df.copy()

    df["word_count"] = (
        df[text_col].fillna("").str.split().str.len()
    )

    return df[df["word_count"] < min_words].copy()


# ============================================================
# APPLICATION DATA PREPARATION
# ============================================================

def prepare_data(
    df,
    response_column,
    id_column=None,
    duplicate_handling="drop",
    min_words=3
):
    if response_column not in df.columns:
        raise ValueError("Response column not found.")

    if id_column and id_column not in df.columns:
        raise ValueError("Identifier column not found.")

    # Keep original columns for future demographic filters.
    working_df = df.copy()

    original_rows = len(working_df)

    working_df = working_df.dropna(
        subset=[response_column]
    )

    working_df = working_df[
        working_df[response_column]
        .astype(str).str.strip().ne("")
    ].copy()

    empty_removed = original_rows - len(working_df)

    duplicates_removed = 0

    if id_column and duplicate_handling == "drop":
        before = len(working_df)

        working_df = working_df.drop_duplicates(
            subset=[id_column],
            keep="first"
        )

        duplicates_removed = before - len(working_df)

    working_df = working_df.reset_index(drop=True)

    # Internal identifier: does not depend on respondent ID.
    working_df.insert(
        0, "response_id", range(len(working_df))
    )

    working_df["clean_text"] = (
        working_df[response_column].apply(clean_text)
    )

    working_df["bert_text"] = (
        working_df[response_column].apply(
            clean_for_bertopic
        )
    )

    working_df["word_count"] = (
        working_df["clean_text"].str.split().str.len()
    )

    short_df = examine_short_responses(
        working_df,
        text_col="clean_text",
        min_words=min_words
    )

    model_df = working_df[
        working_df["word_count"] >= min_words
    ].copy().reset_index(drop=True)

    summary = {
        "original_rows": original_rows,
        "empty_removed": empty_removed,
        "duplicates_removed": duplicates_removed,
        "valid_responses": len(working_df),
        "model_responses": len(model_df),
        "short_responses": len(short_df)
    }

    return working_df, model_df, short_df, summary
