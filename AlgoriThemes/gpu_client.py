"""Private connection to the deployed AlgoriThemes Modal functions.

USE_MODAL and MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are stored in
Streamlit Community Cloud > App settings > Secrets, not in GitHub.
"""

import os
import streamlit as st


def use_modal():
    """Choose Modal GPU (True) or the existing local CPU implementation (False)."""
    value = st.secrets.get("USE_MODAL", False)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def _deployed_function(name):
    """Authenticate with a private Modal token, then locate a deployed function."""
    if not use_modal():
        raise RuntimeError("Modal is disabled. Set USE_MODAL = true to use the GPU.")

    token_id = st.secrets.get("MODAL_TOKEN_ID") or os.environ.get("MODAL_TOKEN_ID")
    token_secret = st.secrets.get("MODAL_TOKEN_SECRET") or os.environ.get("MODAL_TOKEN_SECRET")
    if not token_id or not token_secret:
        raise RuntimeError(
            "Modal credentials are missing. Set MODAL_TOKEN_ID and "
            "MODAL_TOKEN_SECRET in Streamlit Secrets."
        )

    # Modal's SDK reads these environment variables when authenticating.
    # Never print or log their values.
    os.environ["MODAL_TOKEN_ID"] = str(token_id)
    os.environ["MODAL_TOKEN_SECRET"] = str(token_secret)
    import modal

    return modal.Function.from_name("algorithemes-gpu-test", name)


def run_bertopic_gpu(docs, original_responses):
    """Run the deployed BERTopic workflow; return topic metadata, not the model."""
    return _deployed_function("analyse_bertopic_gpu").remote(
        list(docs), list(original_responses)
    )


def run_sentiment_gpu(texts):
    """Run deployed RoBERTa; return one classification per response."""
    return _deployed_function("analyse_sentiment_gpu").remote(list(texts))
