"""
STUDENT FEEDBACK SENTIMENT ANALYSIS
===================================

Dataset:
    TopicModellingStudentFeedbackData (1).xlsx

Workflow:
    1. Import libraries
    2. Load and inspect the Excel dataset
    3. Explore missing values and duplicates
    4. Combine the three student-feedback columns
    5. Clean and preprocess text
    6. Run VADER sentiment analysis
    7. Summarise and visualise sentiment
    8. Export a file for human sentiment labelling
    9. Optionally train/evaluate TF-IDF + Logistic Regression
   10. Predict sentiment for new feedback
   11. Save results and trained model

IMPORTANT:
    The source dataset does not contain verified Positive/Negative/Neutral
    sentiment labels. VADER labels are automatic estimates, not ground truth.
    For supervised machine learning, manually label the exported annotation
    spreadsheet and set TRAIN_SUPERVISED_MODEL = True below.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# 0. SETTINGS
# ---------------------------------------------------------------------

# Put this Python file and the Excel dataset in the same folder,
# or replace DATA_FILE with the full path to your Excel file.
DATA_FILE = Path("TopicModellingStudentFeedbackData (1).xlsx")

OUTPUT_DIR = Path("sentiment_analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FEEDBACK_COLUMNS = [
    "FacultySupportResponse",
    "TASupportResponse",
    "PeerSupportResponse",
]

# Set True only after manually labelling the exported spreadsheet.
TRAIN_SUPERVISED_MODEL = False

ANNOTATION_FILE = OUTPUT_DIR / "Student_Feedback_For_Labelling.xlsx"
LABELLED_FILE = OUTPUT_DIR / "Student_Feedback_For_Labelling.xlsx"

VALID_LABELS = ["Positive", "Negative", "Neutral"]
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# 1. DOWNLOAD REQUIRED NLTK RESOURCE
# ---------------------------------------------------------------------

print("\n[1/11] Preparing NLTK resources...")
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


# ---------------------------------------------------------------------
# 2. LOAD DATASET
# ---------------------------------------------------------------------

print("\n[2/11] Loading Excel dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find: {DATA_FILE.resolve()}\n"
        "Place the Excel file in the same folder as this script, "
        "or update DATA_FILE with its full path."
    )

df = pd.read_excel(DATA_FILE)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all").to_string())


# ---------------------------------------------------------------------
# 3. EXPLORE MISSING VALUES AND DUPLICATES
# ---------------------------------------------------------------------

print("\n[3/11] Exploring data quality...")

print("\nMissing values by column:")
print(df.isnull().sum())

print("\nExact duplicate rows:", df.duplicated().sum())

# Save a missing-value heatmap.
plt.figure(figsize=(12, 5))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values in Student Feedback Dataset")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "missing_values.png", dpi=150)
plt.close()

# Remove only exact duplicate rows. Repeated comments in different
# feedback categories are retained because they may be valid responses.
df = df.drop_duplicates().copy()


# ---------------------------------------------------------------------
# 4. CHECK AND COMBINE FEEDBACK COLUMNS
# ---------------------------------------------------------------------

print("\n[4/11] Preparing student feedback...")

missing_columns = [c for c in FEEDBACK_COLUMNS if c not in df.columns]
if missing_columns:
    raise KeyError(
        "The following expected feedback columns were not found: "
        f"{missing_columns}\nAvailable columns: {df.columns.tolist()}"
    )

# Retain Subject Code when available; otherwise use the row index.
id_column = "Subject Code" if "Subject Code" in df.columns else None
id_vars = [id_column] if id_column else []

feedback_df = df.melt(
    id_vars=id_vars,
    value_vars=FEEDBACK_COLUMNS,
    var_name="Support_Category",
    value_name="Feedback",
)

feedback_df["Feedback"] = feedback_df["Feedback"].fillna("").astype(str).str.strip()
feedback_df = feedback_df[feedback_df["Feedback"] != ""].copy()
feedback_df.reset_index(drop=True, inplace=True)

print("Number of non-empty feedback responses:", len(feedback_df))
print("\nSample feedback:")
print(feedback_df[["Support_Category", "Feedback"]].head(10).to_string(index=False))


# ---------------------------------------------------------------------
# 5. TEXT CLEANING AND PREPROCESSING
# ---------------------------------------------------------------------

print("\n[5/11] Cleaning and preprocessing text...")

def clean_text(text):
    """Lowercase text and remove URLs, HTML, punctuation and extra spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


feedback_df["Cleaned_Feedback"] = feedback_df["Feedback"].apply(clean_text)

# Tokenisation and stopword removal are included for inspection.
# Keep negation words such as no, not, nor and never.
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words("english"))
    stop_words -= {"no", "not", "nor", "never"}

    def tokenize_and_remove_stopwords(text):
        tokens = word_tokenize(text)
        return [word for word in tokens if word not in stop_words]

    feedback_df["Tokens"] = feedback_df["Cleaned_Feedback"].apply(
        tokenize_and_remove_stopwords
    )
    feedback_df["Processed_Text"] = feedback_df["Tokens"].apply(
        lambda tokens: " ".join(tokens)
    )
except LookupError:
    print("NLTK tokenizer resource unavailable; using cleaned text only.")
    feedback_df["Tokens"] = feedback_df["Cleaned_Feedback"].apply(str.split)
    feedback_df["Processed_Text"] = feedback_df["Cleaned_Feedback"]

print("\nText preprocessing examples:")
print(
    feedback_df[["Feedback", "Cleaned_Feedback", "Processed_Text"]]
    .head(10)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# 6. VADER SENTIMENT ANALYSIS
# ---------------------------------------------------------------------

print("\n[6/11] Running VADER sentiment analysis...")

sia = SentimentIntensityAnalyzer()

# Score original feedback so punctuation, capitalisation and negation
# are not discarded before VADER evaluates the text.
feedback_df["Sentiment_Scores"] = feedback_df["Feedback"].apply(
    lambda text: sia.polarity_scores(str(text))
)

feedback_df["Negative_Score"] = feedback_df["Sentiment_Scores"].apply(
    lambda score: score["neg"]
)
feedback_df["Neutral_Score"] = feedback_df["Sentiment_Scores"].apply(
    lambda score: score["neu"]
)
feedback_df["Positive_Score"] = feedback_df["Sentiment_Scores"].apply(
    lambda score: score["pos"]
)
feedback_df["Compound_Score"] = feedback_df["Sentiment_Scores"].apply(
    lambda score: score["compound"]
)


def classify_sentiment(compound_score):
    """Apply the standard VADER compound-score thresholds."""
    if compound_score >= 0.05:
        return "Positive"
    if compound_score <= -0.05:
        return "Negative"
    return "Neutral"


feedback_df["Sentiment"] = feedback_df["Compound_Score"].apply(
    classify_sentiment
)

print("\nVADER sentiment examples:")
print(
    feedback_df[["Feedback", "Compound_Score", "Sentiment"]]
    .head(20)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# 7. SENTIMENT SUMMARY AND VISUALISATIONS
# ---------------------------------------------------------------------

print("\n[7/11] Summarising sentiment...")

sentiment_counts = feedback_df["Sentiment"].value_counts().reindex(
    VALID_LABELS, fill_value=0
)
sentiment_percentages = (
    feedback_df["Sentiment"].value_counts(normalize=True)
    .reindex(VALID_LABELS, fill_value=0) * 100
)

print("\nSentiment counts:")
print(sentiment_counts)

print("\nSentiment percentages:")
print(sentiment_percentages.round(2))

# Overall sentiment distribution.
plt.figure(figsize=(8, 5))
sns.countplot(
    data=feedback_df,
    x="Sentiment",
    order=VALID_LABELS,
)
plt.title("Distribution of Student Sentiment (VADER)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Responses")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sentiment_distribution.png", dpi=150)
plt.close()

# Sentiment by support category.
category_sentiment = pd.crosstab(
    feedback_df["Support_Category"],
    feedback_df["Sentiment"],
).reindex(columns=VALID_LABELS, fill_value=0)

print("\nSentiment by support category:")
print(category_sentiment)

category_sentiment.plot(
    kind="bar",
    stacked=True,
    figsize=(11, 6),
)
plt.title("Sentiment by Support Category (VADER)")
plt.xlabel("Support Category")
plt.ylabel("Number of Responses")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sentiment_by_support_category.png", dpi=150)
plt.close()

# Display examples for qualitative review.
print("\nPositive feedback examples:")
print(
    feedback_df.loc[
        feedback_df["Sentiment"] == "Positive",
        ["Feedback", "Compound_Score"],
    ].head(10).to_string(index=False)
)

print("\nNegative feedback examples:")
print(
    feedback_df.loc[
        feedback_df["Sentiment"] == "Negative",
        ["Feedback", "Compound_Score"],
    ].head(10).to_string(index=False)
)


# ---------------------------------------------------------------------
# 8. EXPORT ANNOTATION TEMPLATE FOR HUMAN LABELS
# ---------------------------------------------------------------------

print("\n[8/11] Creating a human-labelling spreadsheet...")

annotation_data = feedback_df[
    ["Feedback", "Support_Category"]
].copy()
annotation_data["Human_Sentiment"] = ""

annotation_data.to_excel(ANNOTATION_FILE, index=False)

print(f"Annotation template saved to: {ANNOTATION_FILE.resolve()}")
print(
    "Open this spreadsheet and manually fill Human_Sentiment with "
    "Positive, Negative or Neutral. Save it before enabling "
    "TRAIN_SUPERVISED_MODEL."
)


# ---------------------------------------------------------------------
# 9. OPTIONAL SUPERVISED MACHINE LEARNING
# ---------------------------------------------------------------------

model = None
tfidf = None

if TRAIN_SUPERVISED_MODEL:
    print("\n[9/11] Training supervised sentiment classifier...")

    if not LABELLED_FILE.exists():
        raise FileNotFoundError(
            f"Labelled spreadsheet not found: {LABELLED_FILE.resolve()}"
        )

    labelled_df = pd.read_excel(LABELLED_FILE)

    required_annotation_columns = ["Feedback", "Human_Sentiment"]
    missing_annotation_columns = [
        c for c in required_annotation_columns if c not in labelled_df.columns
    ]
    if missing_annotation_columns:
        raise KeyError(
            f"Missing annotation columns: {missing_annotation_columns}"
        )

    labelled_df = labelled_df.dropna(
        subset=["Feedback", "Human_Sentiment"]
    ).copy()
    labelled_df["Feedback"] = labelled_df["Feedback"].astype(str).str.strip()
    labelled_df["Human_Sentiment"] = (
        labelled_df["Human_Sentiment"].astype(str).str.strip().str.title()
    )
    labelled_df = labelled_df[
        labelled_df["Human_Sentiment"].isin(VALID_LABELS)
        & (labelled_df["Feedback"] != "")
    ].copy()

    class_counts = labelled_df["Human_Sentiment"].value_counts()
    print("\nHuman-labelled class counts:")
    print(class_counts)

    if labelled_df["Human_Sentiment"].nunique() < 2:
        raise ValueError(
            "At least two sentiment classes are required for classification."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Each sentiment class needs at least two labelled examples "
            "for a stratified train/test split. Label more examples."
        )

    X = labelled_df["Feedback"]
    y = labelled_df["Human_Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Preserve negation words: stop_words=None.
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=None,
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=VALID_LABELS,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=VALID_LABELS,
    )

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=VALID_LABELS,
        yticklabels=VALID_LABELS,
    )
    plt.title("Confusion Matrix – Human-Labelled Test Data")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    import joblib

    joblib.dump(model, OUTPUT_DIR / "sentiment_model.pkl")
    joblib.dump(tfidf, OUTPUT_DIR / "tfidf_vectorizer.pkl")

    print("\nSaved supervised model and vectorizer.")


# ---------------------------------------------------------------------
# 10. PREDICT NEW FEEDBACK
# ---------------------------------------------------------------------

print("\n[10/11] Example predictions...")

new_feedback = [
    "The lecturer explains concepts very well",
    "The teaching assistant is not helpful",
    "Classes take place every Monday",
    "I really appreciate the support",
]

if model is not None and tfidf is not None:
    new_features = tfidf.transform(new_feedback)
    predictions = model.predict(new_features)
    probabilities = model.predict_proba(new_features)

    prediction_results = pd.DataFrame({
        "Feedback": new_feedback,
        "Predicted_Sentiment": predictions,
        "Model_Confidence": probabilities.max(axis=1),
    })
else:
    # VADER predictions are used when no human-labelled ML model is trained.
    scores = [
        sia.polarity_scores(text)["compound"]
        for text in new_feedback
    ]
    prediction_results = pd.DataFrame({
        "Feedback": new_feedback,
        "Compound_Score": scores,
        "Predicted_Sentiment": [
            classify_sentiment(score) for score in scores
        ],
    })

print(prediction_results.to_string(index=False))


# ---------------------------------------------------------------------
# 11. EXPORT FINAL RESULTS
# ---------------------------------------------------------------------

print("\n[11/11] Exporting final results...")

results_file = OUTPUT_DIR / "Student_Sentiment_Analysis_Results.xlsx"
feedback_df.to_excel(results_file, index=False)

summary_file = OUTPUT_DIR / "Sentiment_Summary.xlsx"
with pd.ExcelWriter(summary_file, engine="openpyxl") as writer:
    sentiment_counts.rename("Count").to_frame().to_excel(
        writer, sheet_name="Sentiment Counts"
    )
    sentiment_percentages.rename("Percentage").to_frame().to_excel(
        writer, sheet_name="Sentiment Percentages"
    )
    category_sentiment.to_excel(
        writer, sheet_name="By Support Category"
    )
    prediction_results.to_excel(
        writer, sheet_name="Example Predictions", index=False
    )

print("\nPROJECT COMPLETED!")
print(f"Detailed results: {results_file.resolve()}")
print(f"Summary workbook: {summary_file.resolve()}")
print(f"Charts and other outputs: {OUTPUT_DIR.resolve()}")
