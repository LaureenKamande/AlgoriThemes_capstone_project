
"""Private connection to the deployed AlgoriThemes Modal GPU backend."""

import streamlit as st


def use_modal():
    """Select Modal GPU or the existing local CPU implementation."""
    value = st.secrets.get("USE_MODAL", False)

    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")

    return bool(value)


def _deployed_function(name):
    """Connect to Modal using explicit credentials."""

    if not use_modal():
        raise RuntimeError(
            "Modal is disabled. Set USE_MODAL = true to use the GPU."
        )

    token_id = str(
        st.secrets.get("MODAL_TOKEN_ID", "")
    ).strip()

    token_secret = str(
        st.secrets.get("MODAL_TOKEN_SECRET", "")
    ).strip()

    if not token_id or not token_secret:
        raise RuntimeError(
            "Modal credentials are missing from Streamlit Secrets."
        )

    import modal

    # Authenticate directly with the credentials from Streamlit.
    client = modal.Client.from_credentials(
        token_id,
        token_secret
    )

    # Verify authentication before attempting GPU processing.
    client.hello()

    # Locate the deployed AlgoriThemes function.
    return modal.Function.from_name(
        "algorithemes-gpu-test",
        name,
        client=client
    )


def run_bertopic_gpu(docs, original_responses):
    """Run BERTopic on Modal's GPU."""

    function = _deployed_function("analyse_bertopic_gpu")

    return function.remote(
        list(docs),
        list(original_responses)
    )


def run_sentiment_gpu(texts):
    """Run RoBERTa sentiment analysis on Modal's GPU."""

    function = _deployed_function("analyse_sentiment_gpu")

    return function.remote(list(texts))
