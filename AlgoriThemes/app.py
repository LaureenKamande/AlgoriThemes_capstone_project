
"""AlgoriThemes — Streamlit application.

Presentation and interface changes only.
The original BERTopic, Gemini and export modules remain separate; RoBERTa provides sentiment labels.
"""

from __future__ import annotations

import io
import html
import hashlib
import base64
from pathlib import Path
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


ASSETS_DIR = Path(__file__).resolve().parent / "assets"


st.set_page_config(
    page_title="AlgoriThemes | From Responses to Insights",
    page_icon=str(ASSETS_DIR / "algorithemes_icon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* AlgoriThemes — white-first interface with deliberate card hierarchy. Presentation only. */
    :root {
        --at-purple: #604582;
        --at-purple-hover: #4E356E;
        --at-teal: #147E7A;
        --at-ink: #252A3C;
        --at-muted: #62687A;
        --at-line: #E0E2E9;
        --at-bg: #FFFFFF;
        --at-sidebar: #F1F2F5;
        --at-card: #FFFFFF;
        --at-input: #FFFFFF;
    }

    /* Shared page grid: same left/right alignment on every route. */
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--at-bg);
        color: var(--at-ink);
    }
    [data-testid="stHeader"] { background: rgba(255, 255, 255, .97); }
    [data-testid="stMainBlockContainer"], .block-container {
        width: 100%;
        max-width: 1160px !important;
        margin-left: auto;
        margin-right: auto;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
        padding-left: 2.25rem;
        padding-right: 2.25rem;
    }
    h1, h2, h3 { color: var(--at-ink); letter-spacing: -.025em; }
    h1 { font-weight: 750; }
    p, label { color: inherit; }
    [data-testid="stCaptionContainer"] { color: var(--at-muted); }

    

    /* Sidebar: clear contrast from the workspace and fixed icon/text columns. */
    section[data-testid="stSidebar"] {
        background: var(--at-sidebar);
        border-right: 1px solid #E5E7EF;
    }
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {
        background: var(--at-sidebar);
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: .36rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stImage"] {
        margin: .35rem 0 .75rem;
    }
    .at-workspace-label {
        margin: .6rem 0 .55rem;
        color: #6D6B7B;
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .115em;
    }
    [data-testid="stSidebar"] .stButton,
    [data-testid="stSidebar"] [data-testid="stButton"] {
        width: 100%;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        width: 100%;
        min-height: 2.85rem;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: .8rem !important;
        padding: .58rem .85rem !important;
        text-align: left !important;
        border-radius: 9px;
        box-shadow: none;
        font-size: .89rem;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton > button [data-testid="stIconMaterial"],
    [data-testid="stSidebar"] [data-testid="stButton"] > button [data-testid="stIconMaterial"] {
        display: inline-flex !important;
        flex: 0 0 21px !important;
        width: 21px !important;
        min-width: 21px !important;
        align-items: center;
        justify-content: center;
        font-size: 1.13rem;
        line-height: 1;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stButton"] > button [data-testid="stMarkdownContainer"] {
        flex: 1 1 auto;
        min-width: 0;
        text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] [data-testid="stButton"] > button p {
        text-align: left !important;
        margin: 0;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"],
    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="secondary"] {
        background: transparent;
        border: 1px solid transparent;
        color: #41465A;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="secondary"]:hover {
        background: #F4F0FA;
        border-color: #E4D9F0;
        color: #583A83;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
        background: var(--at-purple);
        border-color: var(--at-purple);
        color: #fff;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:hover {
        background: var(--at-purple-hover);
        border-color: var(--at-purple-hover);
    }
    [data-testid="stSidebar"] hr { border-color: #E2DCED; margin: 1.1rem 0 .75rem; }
    .at-sidebar-meta { padding: .4rem .1rem .2rem; }
    .at-sidebar-meta-label {
        color: #777487;
        font-size: .67rem;
        font-weight: 750;
        letter-spacing: .09em;
        text-transform: uppercase;
        margin-bottom: .28rem;
    }
    .at-sidebar-meta-value {
        color: #4A4A5C;
        font-size: .81rem;
        line-height: 1.5;
        overflow-wrap: anywhere;
        margin-bottom: .9rem;
    }

    /* White surfaces and clean controls. */
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px;
        min-height: 2.65rem;
        font-weight: 600;
        transition: background-color .15s ease, border-color .15s ease;
    }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: var(--at-purple);
        border-color: var(--at-purple);
        color: #fff;
    }
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover {
        background: var(--at-purple-hover);
        border-color: var(--at-purple-hover);
    }
    .stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {
        background: var(--at-input);
        border-color: var(--at-line);
        color: #3B4154;
    }
    .stButton > button[kind="secondary"]:hover,
    .stDownloadButton > button[kind="secondary"]:hover {
        background: #FCFAFF;
        border-color: #BCA5D7;
        color: var(--at-purple-hover);
    }
    [data-testid="stMetric"] {
        background: var(--at-card);
        border: 1px solid var(--at-line);
        border-radius: 12px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--at-muted); }
    [data-baseweb="tab-list"] { border-bottom: 1px solid var(--at-line); }
    [data-baseweb="tab"] { color: #5B5B6C; }
    [data-testid="stExpander"] { border-color: var(--at-line); }
    [data-baseweb="input"] > div, [data-baseweb="textarea"] > div,
    [data-baseweb="select"] > div {
        background-color: var(--at-input);
    }

    /* Home: same shared content column as Results, Upload and About. */
    .at-home-head { padding: .15rem 0 .95rem; margin-bottom: .25rem; }
    .at-brand-logo {
        width: min(100%, 265px);
        height: auto;
        display: block;
        margin: 0 0 1.55rem;
    }
    .at-kicker {
        color: var(--at-purple);
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: 0 0 .65rem;
    }
    .at-home-head h1 {
        font-size: clamp(1.7rem, 2.7vw, 2.15rem);
        line-height: 1.2;
        margin: 0 0 .65rem;
        color: var(--at-ink);
        font-weight: 750;
    }
    .at-home-head p {
        color: #5A6071;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 680px;
        margin: 0;
    }
    .at-section-label {
        color: #444A5E;
        font-size: .86rem;
        font-weight: 700;
        margin: .7rem 0 .8rem;
    }
    .at-route-marker { height: 0; overflow: hidden; }
    .at-route-icon {
        display: inline-flex;
        width: 38px;
        height: 38px;
        align-items: center;
        justify-content: center;
        background: #F4F0FA;
        color: var(--at-purple);
        border-radius: 10px;
        font-size: .84rem;
        font-weight: 750;
        letter-spacing: .02em;
        margin: .15rem 0 .75rem;
    }
    .at-route-icon.teal { color: #106F6B; background: #E7F6F3; }
    .at-route-title {
        font-size: 1.12rem;
        font-weight: 750;
        color: var(--at-ink);
        letter-spacing: -.02em;
        margin: 0 0 .45rem;
    }
    .at-route-description {
        font-size: .92rem;
        color: #5B6172;
        line-height: 1.6;
        min-height: 4.3rem;
        margin: 0 0 .5rem;
    }
    .at-footnote { color: #686D7F; font-size: .81rem; margin: 1rem 0 0; }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"], .block-container {
            padding-top: 1.1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .at-brand-logo { width: min(100%, 245px); margin-bottom: 1.1rem; }
        .at-route-description { min-height: 0; }
    }


    /* Homepage onboarding: compact, clearly sequenced, and subordinate to route choice. */
    .at-sentiment-guidance {
        display: flex;
        gap: .85rem;
        align-items: flex-start;
        margin: 1.15rem 0 1.55rem;
        padding: .95rem 1.1rem;
        background: #F8F9FC;
        border: 1px solid #E2DCED;
        border-left: 3px solid var(--at-purple);
        border-radius: 10px;
        color: var(--at-ink);
    }
    .at-guidance-icon {
        display: inline-flex;
        flex: 0 0 24px;
        width: 24px;
        height: 24px;
        justify-content: center;
        align-items: center;
        border: 1.5px solid #6E568D;
        border-radius: 50%;
        font-size: .82rem;
        font-weight: 750;
        color: #604582;
    }
    .at-guidance-heading {
        margin: 0 0 .2rem;
        font-size: .89rem;
        font-weight: 750;
        color: #302B42;
    }
    .at-guidance-copy {
        margin: 0;
        font-size: .86rem;
        line-height: 1.6;
        color: #4D4A5C;
    }
    .at-steps {
        margin: .4rem 0 1.1rem;
        padding: 1rem 1.15rem 1.05rem;
        border: 1px solid #E5DFEB;
        border-radius: 12px;
        background: #FFFFFF;
    }
    .at-steps-heading {
        margin: 0 0 .95rem;
        font-size: .88rem;
        font-weight: 750;
        color: #303246;
    }
    .at-steps-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1.15rem;
    }
    .at-step { min-width: 0; }
    .at-step-top {
        display: flex;
        align-items: center;
        gap: .7rem;
        margin-bottom: .6rem;
    }
    .at-step-number {
        display: inline-flex;
        width: 30px;
        height: 30px;
        flex: 0 0 30px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #F4F0FA;
        border: 1px solid #DAC9EB;
        color: #563B79;
        font-size: .81rem;
        font-weight: 750;
    }
    .at-step-rule { height: 1px; flex: 1; background: #DED5E9; }
    .at-step-title {
        margin: 0 0 .25rem;
        color: #2B2D40;
        font-size: .89rem;
        font-weight: 750;
    }
    .at-step-copy {
        margin: 0;
        color: #555669;
        font-size: .82rem;
        line-height: 1.55;
    }
    @media (max-width: 700px) {
        .at-steps-grid { grid-template-columns: 1fr; gap: .95rem; }
        .at-step-top { margin-bottom: .35rem; }
        .at-step-rule { display: none; }
        .at-sentiment-guidance { margin: 1rem 0 1.2rem; }
    }

    /* NAV FIX v2: Streamlit may centre the INNER button content wrapper.
       Align the outer button AND its nested icon/text wrapper. */
    section[data-testid="stSidebar"] [data-testid="stButton"] button,
    section[data-testid="stSidebar"] .stButton button,
    section[data-testid="stSidebar"] button[data-testid^="stBaseButton"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
        width: 100% !important;
        padding-left: 14px !important;
        padding-right: 12px !important;
        gap: 11px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button > div,
    section[data-testid="stSidebar"] .stButton button > div {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
        gap: 11px !important;
        width: 100% !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button [data-testid="stIconMaterial"],
    section[data-testid="stSidebar"] .stButton button [data-testid="stIconMaterial"] {
        flex: 0 0 20px !important;
        width: 20px !important;
        min-width: 20px !important;
        margin: 0 !important;
        text-align: center !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"] {
        flex: 0 1 auto !important;
        width: auto !important;
        text-align: left !important;
        margin: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button p,
    section[data-testid="stSidebar"] .stButton button p {
        margin: 0 !important;
        text-align: left !important;
        white-space: nowrap;
    }
    /* Native Streamlit surfaces match the bright theme. */
    [data-testid="stAlert"], [data-testid="stNotification"] {
        border-color: var(--at-line);
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--at-card);
        border-color: var(--at-line);
    }
    /* v4: Accessible, distinct status notices without the original neon tints.
       Streamlit uses stAlertContentInfo / Warning / Success / Error test IDs
       in its alert body. :has() identifies the semantic type reliably without
       relying on optional data-type or BaseWeb kind attributes. */
    [data-testid="stAlert"] {
        --at-notice-bg: #E2E4EB;
        --at-notice-border: #A9B0C1;
        --at-notice-accent: #5E6B87;
        --at-notice-ink: #26354B;
    }
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]),
    [data-testid="stAlert"][data-type="info"],
    [data-testid="stAlert"]:has([data-baseweb="notification"][kind="info"]) {
        --at-notice-bg: #D7E8F2;
        --at-notice-border: #7FA9C3;
        --at-notice-accent: #387CA2;
        --at-notice-ink: #253A4A;
    }
    [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]),
    [data-testid="stAlert"][data-type="warning"],
    [data-testid="stAlert"]:has([data-baseweb="notification"][kind="warning"]) {
        --at-notice-bg: #F3E5C9;
        --at-notice-border: #C69B50;
        --at-notice-accent: #A66B20;
        --at-notice-ink: #4B3A22;
    }
    [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]),
    [data-testid="stAlert"][data-type="success"],
    [data-testid="stAlert"]:has([data-baseweb="notification"][kind="positive"]) {
        --at-notice-bg: #DCEBE1;
        --at-notice-border: #85B49B;
        --at-notice-accent: #358363;
        --at-notice-ink: #264336;
    }
    [data-testid="stAlert"]:has([data-testid="stAlertContentError"]),
    [data-testid="stAlert"][data-type="error"],
    [data-testid="stAlert"]:has([data-baseweb="notification"][kind="negative"]) {
        --at-notice-bg: #F1DDDF;
        --at-notice-border: #CB8F99;
        --at-notice-accent: #B44E61;
        --at-notice-ink: #502D35;
    }
    /* Style the actual alert surface. The fallback avoids double borders
       on Streamlit versions whose notification element is absent. */
    [data-testid="stAlert"] [data-baseweb="notification"],
    [data-testid="stAlert"]:not(:has([data-baseweb="notification"])) > div {
        background-color: var(--at-notice-bg) !important;
        border: 1px solid var(--at-notice-border) !important;
        border-left: 5px solid var(--at-notice-accent) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] strong {
        color: var(--at-notice-ink) !important;
        opacity: 1 !important;
    }
    /* Preserve the status icon, if the Streamlit version renders one. */
    [data-testid="stAlert"] svg {
        color: var(--at-notice-accent) !important;
    }
    /* Keep low-priority descriptions readable, including the Home privacy note. */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span {
        color: #53576A !important;
        opacity: 1 !important;
    }
    .at-footnote { color: #53576A !important; }

    /* WHITE-FIRST DESIGN: preserve the white workspace. Contrast belongs to
       interactive surfaces, not a tinted full-page background. */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #E4E5EC;
    }
    .at-steps {
        background: #FFFFFF;
        border: 1px solid #E1DDE9;
        box-shadow: 0 1px 3px rgba(34, 29, 52, .025);
    }
    .at-step-number {
        background: #F7F2FC;
        border-color: #DACBE9;
    }

    /* Named Streamlit containers: target the actual rendered card surface.
       The old :has(.at-route-marker) rule did not reliably style the card
       background, leaving cards visually identical to the canvas. */
    .st-key-at_topic_card,
    .st-key-at_sentiment_card {
        background: #FFFFFF !important;
        border: 1px solid #D5CEDF !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 16px rgba(38, 30, 60, .07) !important;
        padding: 1.1rem 1.15rem !important;
        transition: box-shadow .18s ease, border-color .18s ease, transform .18s ease;
    }
    .st-key-at_topic_card {
        border-top: 3px solid #8063A5 !important;
    }
    .st-key-at_sentiment_card {
        border-top: 3px solid #27958F !important;
    }
    .st-key-at_topic_card:hover,
    .st-key-at_topic_card:focus-within {
        border-color: #AD9BC6 !important;
        box-shadow: 0 8px 22px rgba(48, 36, 74, .11) !important;
        transform: translateY(-1px);
    }
    .st-key-at_sentiment_card:hover,
    .st-key-at_sentiment_card:focus-within {
        border-color: #8CBFBA !important;
        box-shadow: 0 8px 22px rgba(28, 90, 88, .10) !important;
        transform: translateY(-1px);
    }
    /* Avoid a doubled border when Streamlit supplies its own bordered
       container wrapper inside the keyed element. */
    .st-key-at_topic_card [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-at_sentiment_card [data-testid="stVerticalBlockBorderWrapper"] {
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    /* Crisp, distinct call-to-action hierarchy. */
    [data-testid="stMain"] .stButton > button[kind="primary"],
    [data-testid="stMain"] .stDownloadButton > button[kind="primary"] {
        background: var(--at-purple);
        border: 1px solid var(--at-purple);
        color: #FFFFFF;
        box-shadow: 0 2px 5px rgba(96, 69, 130, .14);
    }
    [data-testid="stMain"] .stButton > button[kind="primary"]:hover,
    [data-testid="stMain"] .stDownloadButton > button[kind="primary"]:hover {
        background: var(--at-purple-hover);
        border-color: var(--at-purple-hover);
        box-shadow: 0 4px 10px rgba(96, 69, 130, .18);
    }
    [data-testid="stMain"] .stButton > button[kind="secondary"],
    [data-testid="stMain"] .stDownloadButton > button[kind="secondary"] {
        background: #FFFFFF;
        border: 1px solid #BBA9D0;
        color: #563A7C;
    }
    [data-testid="stMain"] .stButton > button[kind="secondary"]:hover,
    [data-testid="stMain"] .stDownloadButton > button[kind="secondary"]:hover {
        background: #F8F4FC;
        border-color: var(--at-purple);
        color: var(--at-purple-hover);
    }
    [data-testid="stMain"] .stButton > button:focus-visible,
    [data-testid="stMain"] .stDownloadButton > button:focus-visible {
        outline: 3px solid #B8A0DB;
        outline-offset: 2px;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border-color: #E0DCE8;
        box-shadow: 0 2px 8px rgba(37, 42, 60, .035);
    }
    /* Results page actions: give each action a clear visual role, while
       leaving the white workspace, sidebar, and homepage cards unchanged. */
    .st-key-at_results_new_analysis button {
        background: #F0EEF4 !important;
        border: 1px solid #C9C2D4 !important;
        color: #38324A !important;
        box-shadow: 0 1px 2px rgba(32, 26, 48, .04);
        font-weight: 650;
    }
    .st-key-at_results_new_analysis button:hover {
        background: #E8E3EF !important;
        border-color: #A996BF !important;
        color: #48315F !important;
    }

    .st-key-download_interactive_html_top button {
    background: #F3F0F7 !important;
    border: 1px solid #C7B8DA !important;
    color: #604582 !important;
    box-shadow: none !important;
    font-weight: 600;
    }

    .st-key-download_interactive_html_top button:hover {
    background: #E9E2F1 !important;
    border-color: #AA95C7 !important;
    color: #4E356E !important;
    box-shadow: 0 2px 5px rgba(72, 48, 105, 0.08);
    }


    /* Downloads page only: restrained contrast for all five export actions.
       Keep the Results header's solid-purple HTML CTA untouched. */
    :is(.st-key-at_export_coded_csv,
        .st-key-at_export_dictionary_csv,
        .st-key-at_export_excel,
        .st-key-download_interactive_html,
        .st-key-at_export_report_word) button {
        background: #F3F0F7 !important;
        border: 1px solid #C7B8DA !important;
        color: #604582 !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    :is(.st-key-at_export_coded_csv,
        .st-key-at_export_dictionary_csv,
        .st-key-at_export_excel,
        .st-key-download_interactive_html,
        .st-key-at_export_report_word) button:hover {
        background: #E9E2F1 !important;
        border-color: #AA95C7 !important;
        color: #4E356E !important;
        box-shadow: 0 2px 5px rgba(72, 48, 105, .08) !important;
    }

    /* Results navigation: editorial, full-width tab row. Keep the active
       destination unmistakable without a rounded, nested pill container. */
    [data-testid="stTabs"] :is([role="tablist"], [data-baseweb="tab-list"]),
    .stTabs :is([role="tablist"], [data-baseweb="tab-list"]) {
        display: flex !important;
        flex-wrap: nowrap !important;
        align-items: end !important;
        gap: .35rem !important;
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto !important;
        padding: 0 !important;
        margin: .85rem 0 1.35rem !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid #DDD9E4 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    [data-testid="stTabs"] :is([role="tab"], [data-baseweb="tab"]),
    .stTabs :is([role="tab"], [data-baseweb="tab"]) {
        display: inline-flex !important;
        justify-content: center !important;
        align-items: center !important;
        min-height: 2.65rem !important;
        padding: .66rem 1.15rem !important;
        border: 1px solid transparent !important;
        border-bottom: 3px solid transparent !important;
        border-radius: 6px 6px 0 0 !important;
        background: transparent !important;
        color: #444055 !important;
        font-weight: 550 !important;
        white-space: nowrap !important;
        box-shadow: none !important;
        transition: color .15s ease, background-color .15s ease !important;
    }
    [data-testid="stTabs"] :is([role="tab"], [data-baseweb="tab"]):hover:not([aria-selected="true"]),
    .stTabs :is([role="tab"], [data-baseweb="tab"]):hover:not([aria-selected="true"]) {
        background: #F7F4FA !important;
        color: var(--at-purple) !important;
    }
    [data-testid="stTabs"] :is([role="tab"], [data-baseweb="tab"])[aria-selected="true"],
    .stTabs :is([role="tab"], [data-baseweb="tab"])[aria-selected="true"] {
        background: var(--at-purple) !important;
        border-color: var(--at-purple) !important;
        border-bottom-color: var(--at-purple) !important;
        color: #FFFFFF !important;
        font-weight: 650 !important;
        box-shadow: none !important;
    }
    [data-testid="stTabs"] :is([role="tab"], [data-baseweb="tab"])[aria-selected="true"] *,
    .stTabs :is([role="tab"], [data-baseweb="tab"])[aria-selected="true"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stTabs"] :is([role="tab"], [data-baseweb="tab"]):focus-visible,
    .stTabs :is([role="tab"], [data-baseweb="tab"]):focus-visible {
        outline: 2px solid #B8A0DB !important;
        outline-offset: -2px !important;
    }
    /* Streamlit's native underline conflicts with the custom selected tab. */
    [data-testid="stTabs"] :is([data-baseweb="tab-highlight"], [data-testid="stTabHighlight"]),
    [data-testid="stTabs"] :is([data-baseweb="tab-border"], [data-testid="stTabBorder"]),
    .stTabs :is([data-baseweb="tab-highlight"], [data-baseweb="tab-border"]) {
        display: none !important;
    }
    @media (max-width: 700px) {
        [data-testid="stTabs"] :is([role="tab"], [data-baseweb="tab"]),
        .stTabs :is([role="tab"], [data-baseweb="tab"]) {
            padding: .6rem .8rem !important;
            min-height: 2.45rem !important;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        .st-key-at_topic_card, .st-key-at_sentiment_card {
            transition: none;
            transform: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Home",
    "Upload & Configure",
    "Results",
    "About",
]

NAV_ICONS = {
    "Home": ":material/home:",
    "Upload & Configure": ":material/upload_file:",
    "Results": ":material/analytics:",
    "About": ":material/info:",
}

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
        "upload_signature": None,
        "analysis_route_selector": "topic",
        "demo_loaded": False,
        "demo_error": None,
        "pending_demo_analysis": None,
        "demo_precomputed": False,
        "demo_cache_warning": None,
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


def set_page(page):
    """Set the active page without modifying a rendered Streamlit widget."""
    st.session_state.page = page


def navigate(page):
    set_page(page)
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
    """Open a saved demo immediately, or run the real pipeline if not yet saved."""
    from demo_snapshots import build_mode, load_demo_snapshot
    from demo_data import get_demo_dataset, get_demo_settings

    st.session_state.demo_cache_warning = None
    if not build_mode():
        try:
            snapshot = load_demo_snapshot(route)
        except Exception as exc:
            # A broken or incompatible cache must never prevent live analysis.
            snapshot = None
            st.session_state.demo_cache_warning = (
                f"Saved demo could not be loaded ({type(exc).__name__}: {exc}). "
                "Running a fresh analysis instead."
            )
        if snapshot is not None:
            clear_preparation()
            st.session_state.pop("dataset_upload", None)
            st.session_state.update(snapshot["state"])
            st.session_state.update({
                "analysis_route_selector": route,
                "page": "Results",
                "upload_signature": None,
                "demo_loaded": True,
                "demo_precomputed": True,
                "demo_error": None,
                "pending_demo_analysis": None,
            })
            return

    # First run (or explicit build mode): use the original real dataset.

    # Validate and prepare before touching the existing session, so a
    # missing/misconfigured demo file does not discard the user's work.
    try:
        settings = get_demo_settings(route)
        raw, question = get_demo_dataset(route)
        response_column = settings["response_column"]
        # A small, reproducible demonstration sample. Keep the full dataset
        # unchanged on disk for research and larger-scale evaluation.
        id_column = settings.get("id_column")
        duplicate_policy = settings.get("duplicate_policy", "Keep all responses")
        working, model_df, short_df, summary = build_prepared_data(
            raw, response_column, id_column, duplicate_policy
        )
        if len(model_df) < 30:
            raise ValueError(
                f"Only {len(model_df)} eligible responses are available; "
                "the demo needs at least 30."
            )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        st.session_state.demo_error = f"Unable to load demo: {exc}"
        st.session_state.page = "Home"
        return

    clear_preparation()
    # Clear a previously uploaded file so it cannot replace the demo on rerun.
    st.session_state.pop("dataset_upload", None)
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
        "page": "Results",
        "upload_signature": None,
        "demo_loaded": True,
        "demo_precomputed": False,
        "demo_error": None,
        # The next Streamlit run starts analysis automatically. No extra click.
        "pending_demo_analysis": route,
    })


def select_own_data_route(route):
    st.session_state.pending_demo_analysis = None
    st.session_state.analysis_route_selector = route
    if st.session_state.demo_loaded:
        # An own-data route should not silently reuse the sample dataset.
        clear_preparation()
        st.session_state.raw_df = None
        st.session_state.dataset_name = None
        st.session_state.sheet_name = None
        st.session_state.survey_question = ""
        st.session_state.upload_signature = None
    st.session_state.demo_loaded = False
    st.session_state.demo_precomputed = False
    st.session_state.demo_cache_warning = None
    st.session_state.demo_error = None
    set_page("Upload & Configure")


# ============================================================
# HOME
# ============================================================


def show_home():
    # A compact, cohesive light homepage; demo and upload callbacks are unchanged.
    logo_path = ASSETS_DIR / "algorithemes_logo_dark.png"
    logo_uri = "data:image/png;base64," + base64.b64encode(logo_path.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <section class="at-home-head">
            <img class="at-brand-logo" src="{logo_uri}" alt="AlgoriThemes — From Responses to Insights" />
            <div class="at-kicker">AI-powered feedback analysis</div>
            <h1>Explore your feedback</h1>
            <p>Find recurring themes, understand sentiment and turn written survey responses
            into insights you can explore and share.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    # The homepage selects the route first; upload and configuration come next.
    st.markdown(
        """<section class="at-steps" aria-label="How AlgoriThemes works">
            <div class="at-steps-heading">How it works</div>
            <div class="at-steps-grid">
                <div class="at-step">
                    <div class="at-step-top"><span class="at-step-number">1</span><span class="at-step-rule"></span></div>
                    <div class="at-step-title">Choose your analysis</div>
                    <p class="at-step-copy">Select topics only, or topics and sentiment.</p>
                </div>
                <div class="at-step">
                    <div class="at-step-top"><span class="at-step-number">2</span><span class="at-step-rule"></span></div>
                    <div class="at-step-title">Upload &amp; configure</div>
                    <p class="at-step-copy">Upload a CSV or Excel file and configure the column to analyse.</p>
                </div>
                <div class="at-step">
                    <div class="at-step-top"><span class="at-step-number">3</span></div>
                    <div class="at-step-title">Explore your insights</div>
                    <p class="at-step-copy">Explore topics, review sentiment if selected, and export results.</p>
                </div>
            </div>
        </section>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="at-section-label">Choose your analysis</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=False, key="at_topic_card"):
            st.markdown(
                """<div class="at-route-marker"></div>
                <div class="at-route-title">Explore topics</div>
                <p class="at-route-description">Identify the main topics in your responses
                and group related topics into broader themes.</p>""",
                unsafe_allow_html=True,
            )
            st.button(
                "Upload my data", key="own_topics", type="primary",
                on_click=select_own_data_route, args=("topic",),
                use_container_width=True,
            )
            st.button(
                "Try topic analysis demo", key="demo_topics", type="secondary",
                on_click=select_demo_route, args=("topic",),
                use_container_width=True,
            )

    with right:
        with st.container(border=False, key="at_sentiment_card"):
            st.markdown(
                """<div class="at-route-marker at-route-marker--sentiment"></div>
                <div class="at-route-title">Explore topics and sentiment</div>
                <p class="at-route-description">Identify what people discuss and compare
                positive, neutral and negative responses within each topic.</p>""",
                unsafe_allow_html=True,
            )
            st.button(
                "Upload my data", key="own_sentiment", type="primary",
                on_click=select_own_data_route, args=("sentiment",),
                use_container_width=True,
            )
            st.button(
                "Try sentiment analysis demo", key="demo_sentiment", type="secondary",
                on_click=select_demo_route, args=("sentiment",),
                use_container_width=True,
            )

    st.markdown(
        '<p class="at-footnote">Try a demo using sample survey responses, or upload your own dataset.</p>',
        unsafe_allow_html=True,
    )
    # Explain route suitability at the point of choice; do not make it a warning.
    st.markdown(
        """<aside class="at-sentiment-guidance" aria-label="When to choose sentiment analysis">
            <span class="at-guidance-icon" aria-hidden="true">i</span>
            <div>
                <div class="at-guidance-heading">When should you choose sentiment analysis?</div>
                <p class="at-guidance-copy">Choose topics and sentiment when responses express
                how people feel, such as reviews, comments about an experience, or complaints. Topic analysis is
                more appropriate when people give suggestions, requests or facts, such as "What one action could help?".</p>
            </div>
        </aside>""",
        unsafe_allow_html=True,
    )

    if st.session_state.get("demo_cache_warning"):
        st.warning(st.session_state.demo_cache_warning)
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
    st.caption(
        "Privacy: Remove identifying information before analysing "
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
            "**When to use this analysis**\n"
            "- Most useful for open-ended responses containing opinions, evaluations, "
            "satisfaction, dissatisfaction, or descriptions of experiences.\n"
            "- For mainly factual or descriptive responses, topic analysis alone "
            "is often more informative.\n"
            "- Sentiment is a prediction for the entire response, not a measure "
            "of sentiment toward each individual topic. Review mixed or "
            "context-dependent responses before drawing conclusions."
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
# UPLOAD + CONFIGURE — ONE STEP
# ============================================================

def show_upload_and_configure():
    st.title("Upload & Configure")
    st.caption("Upload your survey data, choose the response column and click Analyse. ")

    route = st.session_state.analysis_route_selector
    route_label = "Topic analysis" if route == "topic" else "Topic + sentiment analysis"
    st.info(f"Selected on Home: **{route_label}**. To change the analysis route, return to Home.")
    if route == "sentiment":
        st.caption(
            "Sentiment analysis is most useful for opinion-based or evaluative responses. "
            "For mainly factual answers, consider choosing topic analysis on Home."
        )

    st.subheader("1. Upload data")
    upload = st.file_uploader(
        "Choose a CSV or Excel dataset",
        type=["csv", "xlsx", "xls"],
        key="dataset_upload",
    )

    if upload is not None:
        contents = upload.getvalue()
        filename = upload.name
        sheet = 0
        if filename.lower().endswith((".xlsx", ".xls")):
            try:
                sheets = pd.ExcelFile(io.BytesIO(contents)).sheet_names
                sheet = st.selectbox("Excel worksheet", sheets, key="selected_sheet")
            except Exception as exc:
                st.error(f"Could not read Excel worksheets: {exc}")
                return

        signature = hashlib.sha256(contents + filename.encode("utf-8") + str(sheet).encode("utf-8")).hexdigest()
        if signature != st.session_state.upload_signature:
            try:
                preview = read_uploaded_file(contents, filename, sheet)
            except Exception as exc:
                st.error(f"Could not read dataset: {exc}")
                return
            clear_preparation()
            st.session_state.update({
                "raw_df": preview.copy(),
                "dataset_name": filename,
                "sheet_name": sheet,
                "survey_question": "",
                "demo_loaded": False,
                "demo_precomputed": False,
                "upload_signature": signature,
            })

    raw = st.session_state.raw_df
    if raw is None:
        st.info("Upload a dataset above, or choose a sample dataset from Home.")
        return

    if st.session_state.demo_loaded:
        st.success(f"Demo dataset loaded: {st.session_state.dataset_name} (up to 100 responses).")
    else:
        st.success(f"Dataset loaded: {st.session_state.dataset_name}")

    show_metrics([("Rows", f"{len(raw):,}"), ("Columns", len(raw.columns))])
    with st.expander("Preview uploaded data"):
        st.dataframe(raw.head(15), use_container_width=True)

    st.subheader("2. Configure analysis")
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

    st.caption("Likely open-ended response columns are suggested automatically; you can select any column.")
    response_choice = st.selectbox(
        "Open-ended response column", response_options,
        index=response_options.index(initial_choice),
    )
    if response_choice == other_label:
        other_columns = [c for c in columns if c not in suggested] or columns
        response_column = st.selectbox(
            "Select another column", other_columns,
            index=(other_columns.index(current_response)
                   if current_response in other_columns else 0),
        )
    else:
        response_column = response_choice

    with st.form("analyse_form"):
        id_options = ["(None)"] + [c for c in columns if c != response_column]
        selected_id = st.session_state.id_column
        id_selection = st.selectbox(
            "Respondent identifier (optional)", id_options,
            index=(id_options.index(selected_id) if selected_id in id_options else 0),
        )
        id_column = None if id_selection == "(None)" else id_selection

        with st.expander("Advanced options — duplicate handling"):
            policies = [
                "Keep all responses",
                "Remove duplicate response text",
                "Keep first response per respondent ID",
            ]
            default_policy = st.session_state.duplicate_policy
            duplicate_policy = st.selectbox(
                "Duplicate handling", policies,
                index=policies.index(default_policy) if default_policy in policies else 0,
            )

        st.caption("Responses with fewer than three cleaned words are excluded from analysis.")
        submitted = st.form_submit_button("Analyse", type="primary", use_container_width=True)

    if not submitted:
        return
    if duplicate_policy == "Keep first response per respondent ID" and not id_column:
        st.error("Select a respondent identifier, or choose a different duplicate-handling option.")
        return

    try:
        with st.spinner("Preparing survey responses..."):
            working, model_df, short_df, summary = build_prepared_data(
                raw, response_column, id_column, duplicate_policy,
            )
    except Exception as exc:
        st.error(f"Preparation failed: {exc}")
        return

    if len(model_df) == 0:
        st.warning(
            "AlgoriThemes only analyses open-ended survey responses. "
            "Please select a column containing open-ended responses"
        )
        return
    if len(model_df) < 30:
        st.warning(
            f"Only {len(model_df)} eligible open-ended responses remain after preparation. "
            "Topic discovery requires at least 30. Select another response column "
            "or review the duplicate-handling settings."
        )
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
        # Preserve the backend's optional survey_question argument without
        # asking users to supply a question.
        "survey_question": "",
    })
    # No intermediate Analyse page: run the same pipeline and go to Results.
    if execute_analysis(route):
        navigate("Results")


# ============================================================
# ANALYSIS EXECUTION — ORIGINAL MODEL PIPELINE
# ============================================================

def execute_analysis(analysis_type):
    """Run existing BERTopic/RoBERTa/Gemini workflows; return True on success."""
    # A checkpoint is valid only for the exact prepared inputs and question.
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

    try:
        with st.spinner("Running the analytical pipeline..."):
            # Do not load a local transformer when Modal is enabled.
            embedding_model = None if use_modal() else get_embedding_model()
            if analysis_type == "topic":
                results = analyse_topics(
                    model_df=st.session_state.model_df,
                    response_column=st.session_state.response_column,
                    survey_question=st.session_state.survey_question,
                    embedding_model=embedding_model,
                    checkpoint=st.session_state.topic_checkpoint,
                    save_checkpoint=lambda value: st.session_state.update(topic_checkpoint=value),
                )
            else:
                results = analyse_sentiment(
                    clean_df=st.session_state.working_df,
                    model_df=st.session_state.model_df,
                    response_column=st.session_state.response_column,
                    survey_question=st.session_state.survey_question,
                    embedding_model=embedding_model,
                    checkpoint=st.session_state.topic_checkpoint,
                    save_checkpoint=lambda value: st.session_state.update(topic_checkpoint=value),
                )

        st.session_state.analysis_type = analysis_type
        st.session_state.analysis_results = results

        # Recommendation generation is optional; its failure must not
        # discard successfully completed topic/sentiment results.
        try:
            with st.spinner("Preparing evidence-based recommendations..."):
                recommendation_evidence = prepare_recommendation_evidence(results, analysis_type)
                recommendations = generate_recommendations(recommendation_evidence)
            st.session_state.recommendation_evidence = recommendation_evidence
            st.session_state.recommendations = recommendations
        except Exception as rec_exc:
            st.session_state.recommendation_error = (
                f"{type(rec_exc).__name__}: {rec_exc}"
            )
            st.warning("Analysis completed, but recommendations were unavailable. "
                       "You can still view and export your findings.")
        return True
    except Exception as exc:
        st.session_state.analysis_error = f"{type(exc).__name__}: {exc}"
        st.error(f"Analysis failed: {st.session_state.analysis_error}")
        return False


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
# SENTIMENT PRESENTATION HELPERS
# ============================================================

SENTIMENT_COLORS = {
    "Positive": "#4D9B78",
    "Neutral": "#8794A3",
    "Negative": "#C16B6B",
}


def wrap_topic_label(value, width=34):
    """Wrap long Plotly category labels without changing actual topic names."""
    import textwrap

    pieces = textwrap.wrap(
        str(value), width=width, break_long_words=False,
        break_on_hyphens=False,
    )
    return "<br>".join(pieces) if pieces else str(value)


def sentiment_by_topic_chart(distribution):
    """Reuse the same within-topic sentiment figure across results tabs."""
    if distribution is None or distribution.empty:
        return None
    chart_data = distribution.copy()
    names_per_id = (
        chart_data[["topic", "topic_name"]].drop_duplicates()
        if "topic" in chart_data.columns else pd.DataFrame()
    )
    if not names_per_id.empty:
        duplicate_names = names_per_id["topic_name"].duplicated(keep=False)
        duplicate_ids = set(names_per_id.loc[duplicate_names, "topic"])
        chart_data["Topic label"] = chart_data.apply(
            lambda row: (
                f"{row['topic_name']} (topic {row['topic']})"
                if row["topic"] in duplicate_ids else str(row["topic_name"])
            ), axis=1,
        )
    else:
        chart_data["Topic label"] = chart_data["topic_name"].astype(str)

    topic_order = (
        chart_data.groupby("Topic label")["Topic Total"]
        .max().sort_values(ascending=True).index.tolist()
    )
    fig = px.bar(
        chart_data,
        x="Within Topic (%)", y="Topic label", color="sentiment",
        orientation="h", barmode="stack",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment": SENTIMENT_ORDER},
        custom_data=["Count", "Topic Total"],
        title="Sentiment composition of identified topics (%)",
        labels={"Within Topic (%)": "Within topic (%)", "sentiment": "Sentiment"},
    )
    fig.update_traces(
        hovertemplate=(
            "%{y}<br>%{fullData.name}: %{x:.1f}% of topic"
            "<br>n=%{customdata[0]} of %{customdata[1]}<extra></extra>"
        )
    )
    fig.update_yaxes(
        categoryorder="array", categoryarray=topic_order,
        tickmode="array", tickvals=topic_order,
        ticktext=[wrap_topic_label(label) for label in topic_order],
        automargin=True, title=None, tickfont=dict(size=11),
    )
    fig.update_layout(
        height=max(540, 66 * len(topic_order) + 140),
        xaxis=dict(range=[0, 100], ticksuffix="%", title="Within topic (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=310, r=35, t=75, b=50),
    )
    return fig


def render_theme_sentiment(theme_rows, topic_counts, topic_id_col, section_key, theme_index):
    """Show sentiment percentages at theme and constituent-topic levels.

    Uses existing response-level RoBERTa labels; no model is rerun.
    """
    if "sentiment" not in theme_rows.columns or theme_rows["sentiment"].dropna().empty:
        return

    # Theme-level percentages use every classified response in THIS theme.
    theme_total = len(theme_rows)
    counts = theme_rows["sentiment"].value_counts().reindex(
        SENTIMENT_ORDER, fill_value=0
    )
    theme_data = pd.DataFrame({
        "sentiment": SENTIMENT_ORDER,
        "Count": [int(counts[label]) for label in SENTIMENT_ORDER],
    })
    theme_data["Percentage"] = 100 * theme_data["Count"] / max(theme_total, 1)
    theme_data["Group"] = "Theme"
    theme_data["Bar label"] = theme_data["Percentage"].apply(
        lambda value: f"{value:.1f}%" if value >= 8 else ""
    )

    st.markdown("**Sentiment distribution within this theme**")
    theme_fig = px.bar(
        theme_data,
        x="Percentage", y="Group", color="sentiment", orientation="h",
        barmode="stack", text="Bar label", custom_data=["Count"],
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment": SENTIMENT_ORDER},
        labels={"Percentage": "Within theme (%)", "sentiment": "Sentiment"},
    )
    theme_fig.update_traces(
        textposition="inside",
        hovertemplate=(
            "%{fullData.name}: %{x:.1f}% of this theme"
            "<br>n=%{customdata[0]}<extra></extra>"
        ),
    )
    theme_fig.update_layout(
        height=165,
        xaxis=dict(range=[0, 100], ticksuffix="%", title=None),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.14, x=0),
        margin=dict(l=8, r=12, t=45, b=15),
    )
    st.plotly_chart(
        theme_fig, use_container_width=True,
        key=f"{section_key}_theme_sentiment_{theme_index}",
    )
    st.caption(
        " · ".join(
            f"{label}: {int(counts[label]):,} "
            f"({100 * counts[label] / max(theme_total, 1):.1f}%)"
            for label in SENTIMENT_ORDER
        )
    )

    # Topic-level percentages use the number of responses in EACH topic.
    topic_sentiment = (
        theme_rows.groupby([topic_id_col, "topic_name", "sentiment"], dropna=False)
        .size().reset_index(name="Count")
    )
    if topic_sentiment.empty:
        return

    totals = theme_rows.groupby(topic_id_col).size()
    topic_sentiment["Topic Total"] = topic_sentiment[topic_id_col].map(totals)
    topic_sentiment["Within Topic (%)"] = (
        100 * topic_sentiment["Count"] / topic_sentiment["Topic Total"]
    )

    # Use a unique category per topic ID, even if Gemini gives two topics the same name.
    display_labels = {
        row[topic_id_col]: (
            f"{row['topic_name']} (topic {row[topic_id_col]})"
            if topic_counts["topic_name"].eq(row["topic_name"]).sum() > 1
            else str(row["topic_name"])
        )
        for _, row in topic_counts.iterrows()
    }
    topic_sentiment["Topic label"] = topic_sentiment[topic_id_col].map(display_labels)
    ordered_ids = topic_counts.sort_values("Count", ascending=True)[topic_id_col].tolist()
    ordered_labels = [display_labels[topic_id] for topic_id in ordered_ids]

    st.markdown("**Sentiment within topics in this theme**")
    topic_fig = px.bar(
        topic_sentiment,
        x="Within Topic (%)", y="Topic label", color="sentiment",
        orientation="h", barmode="stack", custom_data=["Count", "Topic Total"],
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment": SENTIMENT_ORDER},
        labels={"Within Topic (%)": "Within topic (%)", "sentiment": "Sentiment"},
    )
    topic_fig.update_traces(
        hovertemplate=(
            "%{y}<br>%{fullData.name}: %{x:.1f}% of topic"
            "<br>n=%{customdata[0]} of %{customdata[1]}<extra></extra>"
        )
    )
    topic_fig.update_yaxes(
        categoryorder="array", categoryarray=ordered_labels,
        tickmode="array", tickvals=ordered_labels,
        ticktext=[wrap_topic_label(label) for label in ordered_labels],
        automargin=True, title=None, tickfont=dict(size=11),
    )
    topic_fig.update_layout(
        height=max(300, 68 * len(ordered_labels) + 125),
        xaxis=dict(range=[0, 100], ticksuffix="%", title="Within topic (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=300, r=25, t=55, b=45),
    )
    st.plotly_chart(
        topic_fig, use_container_width=True,
        key=f"{section_key}_topic_sentiment_{theme_index}",
    )
    st.caption("Each topic bar totals 100% of responses assigned to that topic.")


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

        fig.update_yaxes(
            tickmode="array",
            tickvals=topic_counts["topic_name"].tolist(),
            ticktext=[wrap_topic_label(name) for name in topic_counts["topic_name"]],
            automargin=True,
            title=None,
            tickfont=dict(size=11),
        )
        fig.update_layout(
            xaxis=dict(
                range=[0, 105],
                ticksuffix="%",
            ),
            yaxis_title=None,
            margin=dict(l=300, r=55, t=40, b=35),
            height=max(300, len(topic_counts) * 68 + 110),
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"{section_key}_theme_chart_{theme_index}",
        )

        # The sentiment route also shows Positive / Neutral / Negative
        # composition for this theme and for each topic within it.
        render_theme_sentiment(
            theme_rows=theme_rows,
            topic_counts=topic_counts,
            topic_id_col=topic_id_col,
            section_key=section_key,
            theme_index=theme_index,
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

                    # Include existing sentiment predictions when present.
                    for sentiment_column in (
                        "sentiment", "confidence_score", "review_flag"
                    ):
                        if sentiment_column in examples.columns:
                            display_columns.append(sentiment_column)

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
            st.info("Theme grouping skipped: 8 or fewer topics identified.")
            st.subheader("Topic distribution")
            fig = topic_distribution_chart(explorer, len(explorer))
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key="topic_only_distribution")
            st.caption("Percentages use all modelled responses as the denominator.")
            render_topic_details(explorer, mapping, len(explorer), "topic")
            

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
            fig = sentiment_by_topic_chart(dist)
            if fig is not None:
                st.plotly_chart(
                    fig, use_container_width=True, key="sentiment_by_topic_chart"
                )
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


def display_report_distribution_charts(results, analysis_type):
    """Show full-analysis distribution charts in the Survey Analysis Report.

    Charts describe all analysed responses, rather than a filtered subset.
    """
    explorer = results["response_explorer"]
    st.subheader("Distribution of analysed responses")

    if results.get("themes_generated", True):
        view = st.radio(
            "View distribution by", ["Themes", "Topics"],
            horizontal=True, key="report_distribution_view",
        )
        if view == "Themes":
            if analysis_type == "topic":
                theme_dist = get_theme_distribution(results)
            else:
                # The combined route stores response-level themes; derive the
                # same counts without depending on topic-only summary fields.
                assigned = explorer.loc[
                    explorer["theme_name"].notna()
                    & explorer["theme_name"].astype(str).ne("Unassigned")
                ]
                theme_dist = (
                    assigned.groupby("theme_name").size()
                    .reset_index(name="Count")
                )
                theme_dist["Percentage"] = (
                    100 * theme_dist["Count"] / max(len(explorer), 1)
                )
            fig = percentage_bar_chart(theme_dist, "Theme distribution (%)")
        else:
            fig = topic_distribution_chart(
                explorer, len(explorer), "Topic distribution (%)"
            )
    else:
        fig = topic_distribution_chart(
            explorer, len(explorer), "Topic distribution (%)"
        )

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key="report_topic_distribution")
    else:
        st.info("No assigned topics are available to chart.")
    st.caption(
        "Percentages use all analysed responses as the denominator."
    )

    if analysis_type == "sentiment":
        sentiment_dist = get_sentiment_distribution(results)
        if sentiment_dist is not None and not sentiment_dist.empty:
            st.plotly_chart(
                percentage_sentiment_chart(sentiment_dist),
                use_container_width=True, key="report_sentiment_distribution",
            )
        topic_sentiment = sentiment_by_topic_chart(
            results.get("sentiment_by_topic", pd.DataFrame())
        )
        if topic_sentiment is not None:
            st.plotly_chart(
                topic_sentiment, use_container_width=True,
                key="report_sentiment_by_topic",
            )
            st.caption(
                "Within-topic percentages use the responses assigned to each topic "
                "as their denominator; unassigned responses are excluded."
            )
    st.divider()


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

    # Visual summaries belong in the compiled Report, not Response Explorer.
    display_report_distribution_charts(results, analysis_type)

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

def recommendation_coverage_chart(results, recommendations, evidence):
    """Count distinct analysed responses supporting each recommendation.

    Each recommendation can cite multiple topics (and, for combined analyses,
    optionally sentiment-specific evidence). Count response IDs only once
    within a recommendation; the same response can inform several actions.
    """
    explorer = results.get("response_explorer")
    if explorer is None or explorer.empty:
        return None
    topic_column = "topic" if "topic" in explorer.columns else "sentiment_topic_id"
    if topic_column not in explorer.columns:
        return None

    lookup = {str(t["evidence_id"]): t for t in evidence.get("topics", [])}
    topic_codes = pd.to_numeric(explorer[topic_column], errors="coerce")
    response_ids = (
        explorer["_studio_response_id"]
        if "_studio_response_id" in explorer.columns
        else pd.Series(range(len(explorer)), index=explorer.index)
    )
    rows = []
    for index, rec in enumerate(recommendations, 1):
        matched_ids = set()
        for evidence_id in rec.get("supporting_evidence_ids", []):
            item = lookup.get(str(evidence_id))
            if item is None:
                continue
            try:
                topic_id = int(item["topic_code"])
            except (TypeError, ValueError, KeyError):
                continue
            mask = topic_codes.eq(topic_id)
            sentiment = item.get("sentiment")
            if sentiment and "sentiment" in explorer.columns:
                mask = mask & explorer["sentiment"].eq(sentiment)
            matched_ids.update(response_ids.loc[mask].tolist())
        count = len(matched_ids)
        rows.append({
            "Recommendation": f"{index}. {rec['title']}",
            "Responses": count,
            "Share": 100 * count / len(explorer),
            "Priority": str(rec.get("priority", "Other")),
        })

    data = pd.DataFrame(rows)
    if data.empty or data["Responses"].sum() == 0:
        return None
    data = data.sort_values("Responses", ascending=True)
    fig = px.bar(
        data, x="Responses", y="Recommendation", orientation="h",
        color="Priority", text="Responses", custom_data=["Share"],
        title="Survey responses informing each recommendation",
        color_discrete_map={
            "High": "#C16B6B", "Medium": "#7655B8",
            "Low": "#4D9B78", "Other": "#8794A3",
        },
        labels={"Responses": "Distinct supporting responses"},
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "%{y}<br>%{x:,} distinct supporting responses"
            "<br>%{customdata[0]:.1f}% of all analysed responses<extra></extra>"
        ),
        cliponaxis=False,
    )
    order = data["Recommendation"].tolist()
    fig.update_yaxes(
        categoryorder="array", categoryarray=order,
        tickmode="array", tickvals=order,
        ticktext=[wrap_topic_label(name, width=38) for name in order],
        automargin=True, title=None,
    )
    # fig.update_layout(
    #     height=max(350, 66 * len(data) + 145),
    #     margin=dict(l=300, r=65, t=65, b=45),
    #     legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    #     xaxis=dict(range=[0, max(1, data["Responses"].max() * 1.17)]),
    # )
    # return fig
    fig.update_layout(
        height=max(350, 66 * len(data) + 195),
        margin=dict(l=300, r=65, t=115, b=45),
        legend=dict(
            title_text="Priority",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
        ),
        xaxis=dict(range=[0, max(1, data["Responses"].max() * 1.17)]),
        )
    return fig

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
    coverage_fig = recommendation_coverage_chart(results, recommendations, evidence)
    if coverage_fig is not None:
        st.plotly_chart(
            coverage_fig, use_container_width=True, key="recommendation_coverage_chart"
        )
        st.caption(
            "Each bar counts distinct responses assigned to topics cited as evidence "
            "for that recommendation. A response may support more than one recommendation, "
            "so bars should not be added together. Priority labels are AI-generated "
            "and should be reviewed against the original responses."
        )
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
        st.info("Run an analysis first.")

        st.button(
            "Upload & Configure",
            on_click=navigate,
            args=("Upload & Configure",),
        )
        return

    analysis_type = st.session_state.analysis_type
    st.caption(
        "Topic modelling"
        if analysis_type == "topic"
        else "Topic-first sentiment analysis"
    )
    # Keep the complete interactive HTML export accessible from the Results header.
    # Build it once and reuse the same payload in the Downloads tab.
    html_bytes = None
    try:
        coded_for_html, dictionary_for_html = build_coded_outputs(results, analysis_type)
        html_bytes = build_interactive_html(
            results=results,
            analysis_type=analysis_type,
            coded=coded_for_html,
            dictionary=dictionary_for_html,
            recommendations=st.session_state.recommendations,
            recommendation_evidence=st.session_state.recommendation_evidence,
            survey_question=st.session_state.survey_question,
            response_column=st.session_state.response_column,
            id_column=st.session_state.id_column,
        )
    except Exception as exc:
        st.warning(f"Interactive HTML export unavailable: {exc}")

    # Two intentionally sized actions: secondary restart and primary export.
    # The spacer prevents the download CTA from becoming an oversized banner.
    new_analysis_col, html_export_col, _action_spacer = st.columns(
        [1.2, 2.2, 1.1], vertical_alignment="center"
    )
    with new_analysis_col:
        st.button(
            "Start a new analysis",
            on_click=navigate,
            args=("Home",),
            use_container_width=True,
            key="at_results_new_analysis",
        )
    with html_export_col:
        if html_bytes is not None:
            st.download_button(
                "Download Interactive HTML — complete analysis",
                data=html_bytes,
                file_name="algorithemes_results.html",
                mime="text/html",
                use_container_width=True,
                type="primary",
                key="download_interactive_html_top",
            )

    # The first tab reflects whether broader theme grouping was performed.
    tabs = st.tabs([
        "Themes" if results.get("themes_generated", True) else "Topics",
        "Response Explorer",
        "Recommendations",
        "Report",
        "Downloads",
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

    with tabs[4]:
        show_export(embedded=True, prebuilt_html=html_bytes)


# ============================================================
# EXPORT — EXISTING FUNCTIONALITY
# ============================================================

def show_export(embedded=False, prebuilt_html=None):
    if not embedded:
        st.title("Export")
    else:
        st.subheader("Download your analysis")
    results = st.session_state.analysis_results
    if results is None:
        st.warning("Complete an analysis before exporting.")
        return

    analysis_type = st.session_state.analysis_type
    coded, dictionary = build_coded_outputs(results, analysis_type)
    st.write("Download the coded survey dataset and its coding dictionary.")

    st.download_button("Coded responses — CSV", dataframe_to_csv(coded),
                       "coded_responses.csv", "text/csv", use_container_width=True,
                       key="at_export_coded_csv")
    st.download_button("Coding dictionary — CSV", dataframe_to_csv(dictionary),
                       "coding_dictionary.csv", "text/csv", use_container_width=True,
                       key="at_export_dictionary_csv")
    st.download_button("Coded responses + coding dictionary — Excel",
                       coded_workbook(coded, dictionary), "coded_survey_data.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True, key="at_export_excel")

    st.divider()
    st.subheader("Interactive HTML — complete results")
    st.caption("Open the downloaded HTML in your browser to explore charts, topics, coded responses and recommendations without Streamlit or an internet connection.")
    try:
        html_bytes = prebuilt_html
        if html_bytes is None:
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
            key="at_export_report_word",
        )
    except Exception as exc:
        st.warning(f"Report export unavailable: {exc}")


# ============================================================
# APPLICATION NAVIGATION
# ============================================================

def main():
    initialize_state()

    with st.sidebar:
        st.image(str(ASSETS_DIR / "algorithemes_logo_dark.png"), width=205)
        st.markdown('<div class="at-workspace-label">WORKSPACE</div>', unsafe_allow_html=True)
        for page in PAGES:
            st.button(
                page,
                icon=NAV_ICONS[page],
                key=f"nav_{page}",
                use_container_width=True,
                type="primary" if st.session_state.page == page else "secondary",
                on_click=set_page,
                args=(page,),
            )
        st.divider()
        if st.session_state.dataset_name:
            st.markdown(
                '<div class="at-sidebar-meta"><div class="at-sidebar-meta-label">Dataset</div>'
                f'<div class="at-sidebar-meta-value">{html.escape(str(st.session_state.dataset_name))}</div></div>',
                unsafe_allow_html=True,
            )
        if st.session_state.analysis_type:
            st.markdown(
                '<div class="at-sidebar-meta"><div class="at-sidebar-meta-label">Latest analysis</div>'
                f'<div class="at-sidebar-meta-value">{html.escape(str(st.session_state.analysis_type.title()))}</div></div>',
                unsafe_allow_html=True,
            )

    # Demo buttons bypass Upload & Configure: run the existing pipeline
    # once, then display Results automatically on successful completion.
    pending_demo = st.session_state.pending_demo_analysis
    if pending_demo is not None:
        # Clear before executing so Streamlit reruns never repeat the analysis.
        st.session_state.pending_demo_analysis = None
        st.info("Running the sample analysis. Results will appear automatically when it finishes.")
        if not execute_analysis(pending_demo):
            st.session_state.demo_error = (
                "The demo analysis could not be completed: "
                + str(st.session_state.analysis_error or "Unknown error")
            )
            st.session_state.page = "Home"
            st.rerun()
        # Snapshot creation is an explicit local developer action only.
        # Normal visitors never write result files to the server.
        from demo_snapshots import build_mode, save_demo_snapshot
        if build_mode():
            try:
                path = save_demo_snapshot(pending_demo, st.session_state)
                st.success(
                    f"Saved the {pending_demo} demo snapshot to {path}. "
                    "Commit this file to GitHub to enable instant demos online."
                )
            except Exception as exc:
                st.warning(
                    "Analysis succeeded, but the demo snapshot could not be saved: "
                    f"{type(exc).__name__}: {exc}"
                )
        # Keep Results selected and show the newly generated findings below.
        st.session_state.page = "Results"

    pages = {
        "Home": show_home,
        "Upload & Configure": show_upload_and_configure,
        "Results": show_results,
        "About": show_about,
    }
    pages[st.session_state.page]()


if __name__ == "__main__":
    main()
