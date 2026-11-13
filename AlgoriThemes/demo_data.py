"""Real-data demonstration datasets for AlgoriThemes.

Place your own CSV/XLSX/XLS demo datasets in ``demo_datasets/``.
Change the paths, sheet names, and column names in DEMO_CONFIG to match
actual files. Both demos use the normal application analysis pipeline.
No synthetic or precomputed results are used.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

DEMO_CONFIG = {
    "topic": {
        "dataset_name": "Faculty Support Responses",
        "path": "demo_datasets/TopicModellingStudentFeedbackData.xlsx",
        "sheet_name": 0,
        "response_column": "FacultySupportResponse",
        "id_column": "Subject Code",
        "duplicate_policy": "Remove duplicate response text",
        "survey_question": (
            "What one action can professors take to best support students in their classes? (please be as specific as possible). "
            
        ),
    },
    "sentiment": {
        "dataset_name": "Online Employee Reviews",
        "path": "demo_datasets/online employee review dataset.xlsx",
        "sheet_name": 0,
        "response_column": "Text",
        "id_column": None,
        "duplicate_policy": "Keep all responses",
        "survey_question": "Employee reviews about their companies",
    },
}


def get_demo_settings(route: str) -> dict:
    """Return a copy of the configured settings for one demo."""
    if route not in DEMO_CONFIG:
        raise ValueError(f"Unknown demo route: {route!r}")
    return DEMO_CONFIG[route].copy()


def get_demo_dataset(route: str) -> tuple[pd.DataFrame, str]:
    """Load a real CSV/Excel dataset and return it with its survey question.

    Paths are resolved relative to this file, not the shell's current
    working directory, so local and hosted runs behave consistently.
    """
    config = get_demo_settings(route)
    configured_path = Path(config["path"]).expanduser()
    path = (
        configured_path if configured_path.is_absolute()
        else PROJECT_DIR / configured_path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"{config['dataset_name']} demo file was not found: {path}. "
            "Place the real dataset there, or update its path in "
            "demo_data.py (DEMO_CONFIG)."
        )

    extension = path.suffix.lower()
    if extension == ".csv":
        df = pd.read_csv(path)
    elif extension in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=config.get("sheet_name", 0))
    else:
        raise ValueError(
            f"Unsupported demo file type {extension!r}. "
            "Use .csv, .xlsx, or .xls."
        )

    response_column = config["response_column"]
    id_column = config.get("id_column")
    required = [response_column] + ([id_column] if id_column else [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{config['dataset_name']} demo is missing configured "
            f"column(s): {', '.join(map(str, missing))}. "
            f"Available columns: {', '.join(map(str, df.columns))}. "
            "Update DEMO_CONFIG in demo_data.py to match the file."
        )

    if df.empty:
        raise ValueError(f"{config['dataset_name']} demo dataset is empty.")

    return df, config["survey_question"]
