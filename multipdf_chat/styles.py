from __future__ import annotations

import streamlit as st

APP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0a1220;
        --text: #e5eefb;
        --muted: #93a4bc;
        --border: rgba(148, 163, 184, 0.16);
    }

    html, body, [class*="css"] {
        font-family: "Manrope", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(34, 197, 94, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(79, 209, 197, 0.14), transparent 24%),
            linear-gradient(180deg, #08111f 0%, #0b1324 46%, #0e1728 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 15, 28, 0.98), rgba(13, 23, 39, 0.98));
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div {
        color: var(--text);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.85);
        border: 1px dashed rgba(148, 163, 184, 0.35);
        border-radius: 20px;
    }

    .stTextInput > div > div {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
    }

    .stTextInput input {
        color: var(--text);
        min-height: 3.5rem;
        padding-left: 1rem !important;
    }

    .stTextInput input::placeholder {
        color: #8fa5c0;
    }

    [data-testid="InputInstructions"] {
        display: none;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0f766e, #0891b2);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 16px 32px rgba(8, 145, 178, 0.22);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0d9488, #0284c7);
    }

    .hero-card,
    .info-card,
    .response-card,
    .sidebar-card,
    .result-shell {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.80));
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(2, 6, 23, 0.28);
    }

    .hero-card {
        padding: 2rem;
        margin-bottom: 1.2rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(79, 209, 197, 0.12);
        border: 1px solid rgba(79, 209, 197, 0.25);
        color: #9be7df;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .hero-title {
        font-size: clamp(2rem, 4vw, 3.1rem);
        font-weight: 800;
        line-height: 1.05;
        margin: 1rem 0 0.7rem;
        color: #f8fbff;
    }

    .hero-copy,
    .section-copy,
    .sidebar-copy {
        color: var(--muted);
        line-height: 1.72;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 1.4rem;
    }

    .stat-chip {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    .stat-label {
        display: block;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8da2bd;
        margin-bottom: 0.35rem;
    }

    .stat-value {
        color: #f8fbff;
        font-size: 1.05rem;
        font-weight: 700;
    }

    .info-card,
    .response-card,
    .sidebar-card,
    .result-shell {
        padding: 1.2rem 1.25rem;
    }

    .section-title,
    .sidebar-title {
        color: #f8fbff;
        font-weight: 800;
        font-size: 1.08rem;
        margin-bottom: 0.4rem;
    }

    .question-helper {
        margin-top: 0.85rem;
        color: #7f93ad;
        font-size: 0.92rem;
    }

    .response-label,
    .result-kicker {
        color: #8fb7d9;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .result-question {
        color: #f8fbff;
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.6;
        margin: 0;
    }

    .result-meta {
        margin-top: 0.8rem;
        color: #7f93ad;
        font-size: 0.9rem;
    }

    .sidebar-list {
        margin: 0;
        padding-left: 1rem;
        color: #d6e1ef;
        line-height: 1.7;
    }

    @media (max-width: 900px) {
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
"""


def inject_styles():
    st.markdown(APP_CSS, unsafe_allow_html=True)
