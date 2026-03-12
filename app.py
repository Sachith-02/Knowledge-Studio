from __future__ import annotations

import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.chunking import chunk_documents
from src.config import FALLBACK_RESPONSE, get_app_config, get_environment_status
from src.embeddings import get_embedding_model
from src.pdf_processor import extract_pdf_pages
from src.qa_chain import answer_question, get_llm
from src.retriever import retrieve_for_qa, retrieve_for_summary
from src.summarizer import generate_summary, get_summary_query
from src.utils import (
    AppError,
    build_document_id,
    build_source_label,
    save_uploaded_pdf,
    serialize_documents,
    sort_serialized_sources,
    validate_uploaded_pdf,
)
from src.vectordb import (
    create_or_load_vectorstore,
    get_index_dir,
    index_exists,
    list_indexed_documents,
    load_index_by_document_id,
)

st.set_page_config(
    page_title="Knowledge Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

config = get_app_config()
PAGES = ["Chat", "Documents", "Sources", "Analytics", "Settings"]


@st.cache_resource(show_spinner=False)
def load_embeddings_resource(model_name: str, device: str):
    return get_embedding_model(model_name=model_name, device=device)


@st.cache_resource(show_spinner=False)
def load_llm_resource(api_key: str, model_name: str):
    return get_llm(api_key=api_key, model_name=model_name)


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;700&display=swap');
        :root {
            --bg-base: #08090c;
            --bg-surface: #0f1117;
            --bg-elevated: #161922;
            --bg-hover: #1d2130;
            --border: rgba(255,255,255,0.08);
            --border-bright: rgba(255,255,255,0.14);
            --accent: #5e6ad2;
            --accent-glow: rgba(94,106,210,0.35);
            --accent2: #2ed3b7;
            --amber: #f5a623;
            --red: #e5484d;
            --green: #30a46c;
            --text-primary: #f0f2f8;
            --text-secondary: #b8bfd2;
            --text-muted: #7e879f;
            --font-display: 'Syne', sans-serif;
            --font-body: 'DM Sans', sans-serif;
            --radius-sm: 10px;
            --radius-md: 14px;
            --radius-lg: 18px;
            --radius-xl: 24px;
            --shadow-sm: 0 4px 16px rgba(0,0,0,0.35);
            --shadow-md: 0 10px 28px rgba(0,0,0,0.45);
        }
        html, body, [class*="css"] {
            font-family: var(--font-body);
            color: var(--text-primary) !important;
        }
        .stApp {
            background: radial-gradient(circle at top left, rgba(94,106,210,0.09), transparent 25%), var(--bg-base);
        }
        [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
            background: transparent !important;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--border);
            background: var(--bg-surface) !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background: var(--bg-surface);
        }
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span, label, div, small {
            color: var(--text-primary) !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: var(--font-display) !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.02em;
        }
        .topbar {
            display:flex; align-items:center; justify-content:space-between; gap:1rem;
            background: rgba(15,17,23,0.85); border:1px solid var(--border);
            padding: 0.85rem 1rem; border-radius: 18px; box-shadow: var(--shadow-sm);
            backdrop-filter: blur(14px); margin-bottom: 1rem;
        }
        .brand { display:flex; align-items:center; gap:0.8rem; }
        .brand-icon {
            width: 40px; height: 40px; border-radius: 12px;
            display:flex; align-items:center; justify-content:center; color:white !important;
            background: linear-gradient(135deg, var(--accent), var(--accent2)); box-shadow: 0 0 24px var(--accent-glow);
            font-weight:700;
        }
        .brand-name { font-family: var(--font-display); font-size: 1rem; font-weight: 700; }
        .brand-tag { color: var(--text-muted) !important; font-size: 0.78rem; }
        .card, .metric-card, .source-card, .answer-card, .empty-card, .chat-shell {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
        }
        .hero {
            padding: 1.5rem 1.6rem;
            background: linear-gradient(180deg, rgba(94,106,210,0.12) 0%, rgba(22,25,34,1) 100%);
            margin-bottom: 1rem;
        }
        .eyebrow {
            display:inline-flex; align-items:center; gap:0.4rem; padding:0.38rem 0.7rem;
            border-radius:999px; background: rgba(94,106,210,0.14); border:1px solid rgba(94,106,210,0.25);
            color: #cbd3ff !important; font-size:0.76rem; font-weight:700; margin-bottom:0.8rem;
        }
        .hero-title { font-size:2rem; font-weight:800; margin-bottom:0.45rem; }
        .hero-title span { color: #b8c0ff !important; }
        .hero-sub { color: var(--text-secondary) !important; font-size: 0.97rem; line-height: 1.65; max-width: 900px; }
        .badge-row { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.9rem; }
        .pill {
            display:inline-flex; align-items:center; gap:0.35rem; padding:0.38rem 0.72rem;
            border-radius:999px; font-size:0.74rem; font-weight:700; border:1px solid var(--border);
            background: var(--bg-surface); color: var(--text-secondary) !important;
        }
        .pill.primary { color:#cbd3ff !important; border-color: rgba(94,106,210,0.32); background: rgba(94,106,210,0.12); }
        .pill.green { color:#8af0be !important; border-color: rgba(48,164,108,0.32); background: rgba(48,164,108,0.12); }
        .pill.amber { color:#ffd28f !important; border-color: rgba(245,166,35,0.32); background: rgba(245,166,35,0.12); }
        .metric-card { padding: 1rem; min-height: 120px; }
        .metric-label { color: var(--text-muted) !important; font-size: 0.77rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; margin-bottom: 0.6rem; }
        .metric-value { font-family: var(--font-display); font-size: 1.8rem; font-weight: 800; margin-bottom: 0.3rem; }
        .metric-note { color: var(--text-muted) !important; font-size: 0.82rem; }
        .section-title { font-family: var(--font-display); font-size: 1.05rem; font-weight: 700; margin-bottom: 0.25rem; }
        .section-sub { color: var(--text-secondary) !important; font-size: 0.9rem; margin-bottom: 0.85rem; line-height: 1.6; }
        .empty-card { padding: 2rem 1.25rem; text-align:center; }
        .empty-icon { font-size: 2rem; margin-bottom: 0.6rem; }
        .empty-title { font-weight: 700; margin-bottom: 0.35rem; }
        .empty-text { color: var(--text-secondary) !important; font-size: 0.92rem; line-height: 1.6; }
        .source-card { padding: 1rem; margin-bottom: 0.85rem; }
        .source-meta { color: var(--text-muted) !important; font-size: 0.8rem; margin: 0.3rem 0 0.65rem 0; }
        .source-snippet {
            background: var(--bg-surface); border:1px solid var(--border); border-radius: 12px;
            padding: 0.85rem; color: var(--text-secondary) !important; line-height: 1.6; font-size: 0.9rem;
        }
        .sidebar-panel {
            background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-lg);
            padding: 0.95rem; margin-bottom: 0.9rem;
        }
        .status-line { display:flex; align-items:center; justify-content:space-between; gap:0.5rem; }
        .status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:0.4rem; }
        .answer-card { padding: 1rem; }
        .toolbar-note { color: var(--text-muted) !important; font-size: 0.78rem; }
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px !important; border: 1px solid var(--border) !important;
            background: var(--bg-surface) !important; color: var(--text-primary) !important;
            font-weight: 700 !important; box-shadow: none !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--border-bright) !important; background: var(--bg-hover) !important;
        }
        .primary-btn button {
            background: linear-gradient(135deg, var(--accent), #7280f4) !important;
            border-color: transparent !important; color: white !important;
            box-shadow: 0 0 24px rgba(94,106,210,0.25) !important;
        }
        .stTextInput input, .stTextArea textarea {
            background: var(--bg-surface) !important; color: var(--text-primary) !important; border-radius: 14px !important;
        }
        .stTextArea textarea::placeholder, .stTextInput input::placeholder { color: var(--text-muted) !important; }
        .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div, .stMultiSelect > div > div, .stFileUploader > div {
            background: transparent !important; color: var(--text-primary) !important;
        }
        .stFileUploader > div {
            border: 1.5px dashed var(--border-bright) !important; border-radius: 18px !important;
            background: var(--bg-elevated) !important;
        }
        .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {
            background: var(--bg-surface) !important; border:1px solid var(--border) !important; color: var(--text-primary) !important;
        }
        .stSlider [data-baseweb="slider"] { color: var(--accent) !important; }
        [data-testid="stChatMessage"] {
            background: transparent !important;
        }
        [data-testid="stChatMessageContent"] {
            background: var(--bg-elevated) !important; border:1px solid var(--border) !important;
            border-radius: 16px !important; padding: 0.9rem 1rem !important;
        }
        .nav-note { color: var(--text-muted) !important; font-size:0.78rem; }
        .mini-chip { display:inline-block; padding:0.26rem 0.55rem; border-radius:999px; background: var(--bg-surface); border:1px solid var(--border); color: var(--text-secondary) !important; font-size:0.74rem; margin-right:0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "active_document_id": None,
        "active_source_filename": None,
        "active_pdf_path": None,
        "active_index_metadata": None,
        "active_vectorstore": None,
        "chat_messages": [],
        "summary_markdown": None,
        "summary_sources": [],
        "page": "Chat",
        "pending_question": "",
        "selected_saved_index": None,
        "recent_uploads": [],
        "query_log": [],
        "latency_log": [],
        "feedback_log": [],
        "last_sources": [],
        "last_answer": "",
        "status_message": "",
        "selected_upload_name": None,
        "cfg_top_k": config.qa_k,
        "cfg_search_type": "MMR",
        "settings_temperature": 0.0,
        "cfg_rerank": False,
        "cfg_citations": True,
        "settings_theme": "Dark",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def switch_page(page: str) -> None:
    st.session_state.page = page


def set_status(message: str) -> None:
    st.session_state.status_message = message


def clear_chat_history() -> None:
    st.session_state.chat_messages = []
    st.session_state.last_answer = ""
    st.session_state.last_sources = []
    set_status("Chat history cleared.")


def clear_summary() -> None:
    st.session_state.summary_markdown = None
    st.session_state.summary_sources = []


def activate_document(*, document_id: str, source_filename: str, pdf_path: Path | None, index_metadata: dict[str, Any], vector_store: Any, reset_history: bool) -> None:
    if reset_history:
        clear_chat_history()
        clear_summary()
    st.session_state.active_document_id = document_id
    st.session_state.active_source_filename = source_filename
    st.session_state.active_pdf_path = str(pdf_path) if pdf_path else None
    st.session_state.active_index_metadata = index_metadata
    st.session_state.active_vectorstore = vector_store


def find_saved_pdf(document_id: str) -> Path | None:
    document_dir = config.data_dir / document_id
    if not document_dir.exists():
        return None
    pdf_files = sorted(document_dir.glob("*.pdf"))
    return pdf_files[0] if pdf_files else None


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def add_recent_upload(file_name: str, size_bytes: int, status: str, chunk_count: int = 0) -> None:
    rows = [item for item in st.session_state.recent_uploads if item.get("name") != file_name]
    rows.insert(
        0,
        {
            "name": file_name,
            "size_bytes": size_bytes,
            "status": status,
            "chunk_count": chunk_count,
            "updated_at": time.strftime("%H:%M"),
        },
    )
    st.session_state.recent_uploads = rows[:10]


def render_topbar() -> None:
    st.markdown(
        """
        <div class="topbar">
            <div class="brand">
                <div class="brand-icon">✦</div>
                <div>
                    <div class="brand-name">Knowledge Studio</div>
                    <div class="brand-tag">AI document intelligence</div>
                </div>
            </div>
            <div class="nav-note">Upload knowledge · Retrieve context · Generate grounded answers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1, 1, 1, 1, 3, 1])
    for idx, page in enumerate(PAGES):
        with cols[idx]:
            button_type = "primary" if st.session_state.page == page else "secondary"
            if st.button(page, key=f"topnav_{page}", use_container_width=True, type=button_type):
                switch_page(page)
                st.rerun()
    with cols[5]:
        if st.session_state.status_message:
            st.markdown(f"<div class='nav-note'>{st.session_state.status_message}</div>", unsafe_allow_html=True)
    with cols[6]:
        if st.button("Clear", key="clear_top", use_container_width=True):
            clear_chat_history()
            st.rerun()


def render_sidebar(indexed_documents: list[dict[str, Any]], env_status: dict[str, bool]) -> bool:
    load_saved_clicked = False
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-panel">
                <div class="brand" style="gap:0.7rem;">
                    <div class="brand-icon" style="width:36px;height:36px;">✦</div>
                    <div>
                        <div class="brand-name" style="font-size:0.96rem;">Knowledge Studio</div>
                        <div class="brand-tag">AI document intelligence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
        st.selectbox("Workspace", ["Default Workspace", "Research Lab", "Investor Demo"], key="workspace_selector")
        st.selectbox("Knowledge Base", ["Lecture Collection", "Uploaded Docs", "All Documents"], key="kb_selector")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Navigation</div>", unsafe_allow_html=True)
        for page in PAGES:
            button_type = "primary" if st.session_state.page == page else "secondary"
            if st.button(page, key=f"side_{page}", use_container_width=True, type=button_type):
                switch_page(page)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Retrieval</div>", unsafe_allow_html=True)
        st.slider("Top-K chunks", 3, 8, key="sidebar_top_k", on_change=update_cfg_from_sidebar)
        st.selectbox("Search type", ["MMR", "Similarity"], key="sidebar_search_type", on_change=update_cfg_from_sidebar)
        st.toggle("Re-rank retrieved chunks", key="sidebar_rerank", on_change=update_cfg_from_sidebar)
        st.toggle("Show citations", key="sidebar_citations", on_change=update_cfg_from_sidebar)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Saved indexes</div>", unsafe_allow_html=True)
        if indexed_documents:
            options = {f"{row.get('source_filename','document.pdf')} · {row.get('document_id','')[:10]}": row.get("document_id", "") for row in indexed_documents}
            label = st.selectbox("Load index", ["Select a saved index"] + list(options.keys()), key="saved_index_select")
            if label != "Select a saved index":
                st.session_state.selected_saved_index = options[label]
            load_saved_clicked = st.button("Load saved index", key="load_saved_btn", use_container_width=True)
        else:
            st.caption("No saved indexes yet.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Environment</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='status-line'><span><span class='status-dot' style='background:{'#30a46c' if env_status['groq_api_key_present'] else '#e5484d'};'></span>Groq API Key</span><span class='nav-note'>{'Active' if env_status['groq_api_key_present'] else 'Missing'}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='status-line'><span><span class='status-dot' style='background:{'#f5a623' if env_status['env_file_exists'] else '#e5484d'};'></span>.env File</span><span class='nav-note'>{'Detected' if env_status['env_file_exists'] else 'Missing'}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    return load_saved_clicked


def render_hero() -> None:
    meta = st.session_state.active_index_metadata or {}
    active_name = meta.get("source_filename", "No active document")
    st.markdown(
        f"""
        <div class="card hero">
            <div class="eyebrow">⚡ Retrieval-grounded AI</div>
            <div class="hero-title">Ask your <span>knowledge base</span></div>
            <div class="hero-sub">Upload documents, retrieve the most relevant context, and generate grounded answers with full source transparency. The visual system is based on your dark Knowledge Studio UI reference.</div>
            <div class="badge-row">
                <span class="pill primary">Active document: {active_name}</span>
                <span class="pill">Transparent sources</span>
                <span class="pill green">Grounded answers</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty(icon: str, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="empty-card">
            <div class="empty-icon">{icon}</div>
            <div class="empty-title">{title}</div>
            <div class="empty-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_cards(serialized_sources: list[dict[str, Any]]) -> None:
    if not serialized_sources:
        render_empty("🔎", "No retrieved sources yet", "Ask a question or generate a summary to inspect the retrieval evidence behind the answer.")
        return
    for idx, source in enumerate(sort_serialized_sources(serialized_sources), start=1):
        metadata = source.get("metadata", {})
        label = build_source_label(metadata)
        preview = source.get("content", "")
        st.markdown(
            f"""
            <div class="source-card">
                <div class="section-title">Source {idx} · {label}</div>
                <div class="source-meta">Document ID {str(metadata.get('document_id', ''))[:12]} · Page {metadata.get('page_number', '?')} · Chunk {metadata.get('chunk_index', '?')}</div>
                <div class="source-snippet">{preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.cfg_citations:
            with st.expander(f"Metadata for source {idx}"):
                st.json(metadata)


def render_right_panel() -> None:
    with st.container(border=False):
        st.markdown("<div class='card' style='padding:1rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Retrieval insights</div><div class='section-sub'>Inspect source evidence, active document metadata, and answer grounding.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Assistant answer</div>", unsafe_allow_html=True)
        if st.session_state.last_answer:
            source_count = len(st.session_state.last_sources)
            confidence = "High" if source_count >= 4 else "Medium" if source_count >= 2 else "Low"
            color_cls = "green" if confidence == "High" else "amber" if confidence == "Medium" else ""
            st.markdown(
                f"<div class='badge-row'><span class='pill primary'>Answer generated from {source_count} source(s)</span><span class='pill {color_cls}'>Confidence: {confidence}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(st.session_state.last_answer)
            tool_cols = st.columns(4)
            with tool_cols[0]:
                st.download_button("Export answer", st.session_state.last_answer, file_name="answer.md", key="dl_answer", use_container_width=True)
            with tool_cols[1]:
                if st.button("Show sources", key="btn_show_sources", use_container_width=True):
                    switch_page("Sources")
                    st.rerun()
            with tool_cols[2]:
                if st.button("Helpful", key="btn_helpful", use_container_width=True):
                    st.session_state.feedback_log.append("up")
                    set_status("Marked the latest answer as helpful.")
                    st.rerun()
            with tool_cols[3]:
                if st.button("Needs work", key="btn_down", use_container_width=True):
                    st.session_state.feedback_log.append("down")
                    set_status("Recorded feedback for the latest answer.")
                    st.rerun()
        else:
            render_empty("✨", "No answer yet", "Ask a grounded question to see the answer, confidence, and source-backed evidence here.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card' style='padding:1rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Latest sources</div><div class='section-sub'>Retrieved chunks from your last response.</div>", unsafe_allow_html=True)
        render_source_cards(st.session_state.last_sources)
        st.markdown("</div>", unsafe_allow_html=True)


def handle_saved_index_load(document_id: str) -> None:
    embeddings = load_embeddings_resource(config.embedding_model, config.embedding_device)
    current_document_id = st.session_state.active_document_id
    with st.status("Loading saved knowledge base...", expanded=True) as status:
        st.write("Opening the embedding model cache...")
        st.write("Loading the saved Chroma vector database...")
        vector_store, index_metadata = load_index_by_document_id(document_id=document_id, vectordb_dir=config.vectordb_dir, embedding_function=embeddings)
        pdf_path = find_saved_pdf(document_id)
        activate_document(
            document_id=document_id,
            source_filename=index_metadata.get("source_filename", "uploaded_document.pdf"),
            pdf_path=pdf_path,
            index_metadata=index_metadata,
            vector_store=vector_store,
            reset_history=current_document_id != document_id,
        )
        status.update(label="Knowledge base loaded successfully.", state="complete")
    set_status(f"Loaded saved index for {index_metadata.get('source_filename', 'uploaded_document.pdf')}.")


def handle_uploaded_pdf(uploaded_file, force_rebuild: bool = False) -> None:
    file_bytes = uploaded_file.getvalue()
    validate_uploaded_pdf(uploaded_file.name, file_bytes)
    document_id = build_document_id(file_bytes)
    saved_pdf_path = save_uploaded_pdf(file_bytes=file_bytes, original_filename=uploaded_file.name, data_dir=config.data_dir, document_id=document_id)
    current_document_id = st.session_state.active_document_id

    with st.status("Processing document...", expanded=True) as status:
        st.write("Validating uploaded file...")
        st.write("Loading embedding model...")
        embeddings = load_embeddings_resource(config.embedding_model, config.embedding_device)

        if index_exists(config.vectordb_dir, document_id) and not force_rebuild:
            st.write("Existing document hash found. Loading persistent vector database...")
            vector_store, index_metadata, loaded_existing = create_or_load_vectorstore(
                document_id=document_id,
                source_filename=uploaded_file.name,
                file_size_bytes=len(file_bytes),
                page_count=0,
                chunks=None,
                embedding_function=embeddings,
                embedding_model=config.embedding_model,
                vectordb_dir=config.vectordb_dir,
                collection_name=config.collection_name,
                force_rebuild=False,
            )
        else:
            st.write("Extracting content and building a new vector database...")
            page_documents = extract_pdf_pages(file_path=saved_pdf_path, source_filename=uploaded_file.name, document_id=document_id)
            page_count = int(page_documents[0].metadata.get("page_count", len(page_documents)))
            st.write(f"Extracted content from {page_count} page(s).")
            chunks = chunk_documents(documents=page_documents, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
            st.write(f"Created {len(chunks)} retrieval chunks.")
            vector_store, index_metadata, loaded_existing = create_or_load_vectorstore(
                document_id=document_id,
                source_filename=uploaded_file.name,
                file_size_bytes=len(file_bytes),
                page_count=page_count,
                chunks=chunks,
                embedding_function=embeddings,
                embedding_model=config.embedding_model,
                vectordb_dir=config.vectordb_dir,
                collection_name=config.collection_name,
                force_rebuild=force_rebuild,
            )

        activate_document(
            document_id=document_id,
            source_filename=index_metadata.get("source_filename", uploaded_file.name),
            pdf_path=saved_pdf_path,
            index_metadata=index_metadata,
            vector_store=vector_store,
            reset_history=force_rebuild or current_document_id != document_id,
        )
        add_recent_upload(uploaded_file.name, len(file_bytes), "Loaded" if loaded_existing else "Indexed", int(index_metadata.get("chunk_count", 0)))
        status.update(label="Document is ready for grounded chat.", state="complete")
    set_status(f"{'Loaded existing index for' if loaded_existing else 'Indexed'} {uploaded_file.name}.")


def delete_index(document_id: str) -> None:
    index_dir = get_index_dir(config.vectordb_dir, document_id)
    data_dir = config.data_dir / document_id
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    if st.session_state.active_document_id == document_id:
        st.session_state.active_document_id = None
        st.session_state.active_source_filename = None
        st.session_state.active_pdf_path = None
        st.session_state.active_index_metadata = None
        st.session_state.active_vectorstore = None
        clear_chat_history()
    set_status("Deleted the selected index and its cached document.")


def ask_question(question: str) -> None:
    if st.session_state.active_vectorstore is None:
        raise AppError("Upload or load a document before asking questions.")
    env_status = get_environment_status(config)
    if not env_status["groq_api_key_present"]:
        raise AppError("GROQ_API_KEY is missing. Add it to your .env file to enable Q&A.")

    st.session_state.chat_messages.append({"role": "user", "content": question})
    llm = load_llm_resource(config.groq_api_key, config.groq_model)
    start = time.time()
    with st.status("Thinking with retrieved context...", expanded=True) as status:
        st.write("Searching the most relevant chunks...")
        search_type = st.session_state.cfg_search_type
        retrieved = retrieve_for_qa(
            st.session_state.active_vectorstore,
            question,
            k=st.session_state.cfg_top_k,
            fetch_k=max(config.qa_fetch_k, st.session_state.cfg_top_k * 2),
            lambda_mult=0.7 if search_type == "MMR" else 1.0,
        )
        if st.session_state.cfg_rerank:
            retrieved = sorted(retrieved, key=lambda d: len(d.page_content), reverse=True)
        st.write(f"Retrieved {len(retrieved)} chunk(s).")
        st.write("Generating a grounded answer with Groq...")
        answer = answer_question(question, retrieved, llm)
        if not retrieved and not answer.strip():
            answer = FALLBACK_RESPONSE
        sources = serialize_documents(retrieved)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer or FALLBACK_RESPONSE, "sources": sources})
        st.session_state.last_answer = answer or FALLBACK_RESPONSE
        st.session_state.last_sources = sources
        latency = round(time.time() - start, 2)
        st.session_state.query_log.append({"time": time.strftime("%H:%M"), "question": question, "source_count": len(sources)})
        st.session_state.latency_log.append(latency)
        status.update(label="Answer ready.", state="complete")
    set_status(f"Answered using {len(sources)} source chunk(s) in {latency:.2f}s.")


def generate_summary_action() -> None:
    if st.session_state.active_vectorstore is None:
        raise AppError("Upload or load a document before generating a summary.")
    env_status = get_environment_status(config)
    if not env_status["groq_api_key_present"]:
        raise AppError("GROQ_API_KEY is missing. Add it to your .env file to enable summary generation.")

    llm = load_llm_resource(config.groq_api_key, config.groq_model)
    with st.status("Generating structured summary...", expanded=True) as status:
        st.write("Retrieving broad context for synthesis...")
        docs = retrieve_for_summary(
            st.session_state.active_vectorstore,
            get_summary_query(),
            k=max(config.summary_k, 6),
            fetch_k=max(config.summary_fetch_k, 12),
            lambda_mult=0.5,
        )
        st.write(f"Retrieved {len(docs)} chunk(s).")
        st.write("Synthesizing summary with Groq...")
        summary_text = generate_summary(docs, llm)
        st.session_state.summary_markdown = summary_text
        st.session_state.summary_sources = serialize_documents(docs)
        st.session_state.last_answer = summary_text
        st.session_state.last_sources = st.session_state.summary_sources
        status.update(label="Summary ready.", state="complete")
    set_status("Generated a structured summary from retrieved context.")


def render_chat_page() -> None:
    render_hero()
    metrics = st.columns(4)
    metadata = st.session_state.active_index_metadata or {}
    with metrics[0]:
        render_metric_card("Documents indexed", str(len(list_indexed_documents(config.vectordb_dir))), "Across saved knowledge bases")
    with metrics[1]:
        render_metric_card("Total chunks", str(int(metadata.get("chunk_count", 0))), "Indexed retrieval units")
    with metrics[2]:
        avg_latency = f"{sum(st.session_state.latency_log)/len(st.session_state.latency_log):.2f}s" if st.session_state.latency_log else "—"
        render_metric_card("Avg latency", avg_latency, "Observed this session")
    with metrics[3]:
        render_metric_card("Queries", str(len(st.session_state.query_log)), "This session")

    action_cols = st.columns([1, 1, 1, 2])
    with action_cols[0]:
        if st.button("Upload documents", key="hero_upload", use_container_width=True):
            switch_page("Documents")
            st.rerun()
    with action_cols[1]:
        if st.button("Start new chat", key="hero_new_chat", use_container_width=True):
            clear_chat_history()
            st.rerun()
    with action_cols[2]:
        if st.button("View sources", key="hero_sources", use_container_width=True):
            switch_page("Sources")
            st.rerun()
    with action_cols[3]:
        st.markdown("<span class='mini-chip'>Upload knowledge</span><span class='mini-chip'>Retrieve context</span><span class='mini-chip'>Ground answers</span>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    chat_col, right_col = st.columns([1.7, 1], gap="large")

    with chat_col:
        st.markdown("<div class='chat-shell' style='padding:1rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Conversation</div><div class='section-sub'>Ask questions about your active knowledge base. Answers are grounded in retrieved document chunks.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        prompt_cols = st.columns(4)
        prompts = [
            "Summarize the core ideas",
            "What are the key definitions?",
            "What does page 3 explain?",
            "List the main takeaways",
        ]
        for idx, prompt in enumerate(prompts):
            with prompt_cols[idx]:
                if st.button(prompt, key=f"prompt_{idx}", use_container_width=True):
                    st.session_state.pending_question = prompt
                    st.rerun()

        if not st.session_state.chat_messages:
            render_empty("💬", "No conversation yet", "Start with a question like “Summarize the core ideas” or “What does the document say about topic X?”")
        else:
            for idx, message in enumerate(st.session_state.chat_messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and message.get("sources"):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if st.button(f"Show sources · {len(message['sources'])}", key=f"show_src_{idx}", use_container_width=True):
                                st.session_state.last_sources = message["sources"]
                                switch_page("Sources")
                                st.rerun()
                        with c2:
                            if st.button("Helpful", key=f"up_{idx}", use_container_width=True):
                                st.session_state.feedback_log.append("up")
                                set_status("Recorded positive feedback.")
                                st.rerun()
                        with c3:
                            if st.button("Needs work", key=f"down_{idx}", use_container_width=True):
                                st.session_state.feedback_log.append("down")
                                set_status("Recorded improvement feedback.")
                                st.rerun()

        question = st.chat_input("Ask a grounded question about your knowledge base…")
        if not question and st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = ""
        if question:
            try:
                ask_question(question)
                st.rerun()
            except AppError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected failure during Q&A: {exc}")

    with right_col:
        render_right_panel()


def render_documents_page(indexed_documents: list[dict[str, Any]]) -> None:
    st.markdown("<div class='section-title'>Documents & knowledge bases</div><div class='section-sub'>Upload PDFs, build persistent vector indexes, and manage your knowledge library from one workspace.</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop files to upload or click to browse",
        type=["pdf", "docx", "txt", "csv", "pptx"],
        accept_multiple_files=True,
        help="PDF indexing is fully wired to the backend. Other file types are surfaced in the UI for future ingestion support.",
    )
    if uploaded_files:
        for idx, uploaded in enumerate(uploaded_files):
            is_pdf = uploaded.name.lower().endswith(".pdf")
            st.markdown("<div class='source-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='section-title'>📄 {uploaded.name}</div><div class='source-meta'>Size {format_size(len(uploaded.getvalue()))} · Type {uploaded.name.split('.')[-1].upper()} · {'Ready to index' if is_pdf else 'UI only'}</div>", unsafe_allow_html=True)
            cols = st.columns([1, 1, 4])
            with cols[0]:
                if is_pdf:
                    if st.button("Process", key=f"process_file_{idx}", use_container_width=True):
                        try:
                            handle_uploaded_pdf(uploaded, force_rebuild=False)
                            switch_page("Chat")
                            st.rerun()
                        except AppError as exc:
                            st.error(str(exc))
                        except Exception as exc:
                            st.error(f"Unexpected error while processing the PDF: {exc}")
                else:
                    st.button("Unsupported", key=f"unsupported_{idx}", disabled=True, use_container_width=True)
            with cols[1]:
                if st.button("Preview", key=f"preview_file_{idx}", use_container_width=True):
                    st.session_state.selected_upload_name = uploaded.name
                    set_status(f"Selected {uploaded.name} for preview.")
                    st.rerun()
            with cols[2]:
                if st.session_state.selected_upload_name == uploaded.name:
                    st.info(f"{uploaded.name} is selected. PDF processing builds a persistent Chroma index and activates the document for chat.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Indexed documents</div><div class='section-sub'>Load, inspect, and delete existing indexes.</div>", unsafe_allow_html=True)
    if not indexed_documents:
        render_empty("📚", "No indexed documents yet", "Upload a PDF to create your first persistent knowledge base.")
        return

    for idx, row in enumerate(indexed_documents):
        doc_id = row.get("document_id", "")
        st.markdown("<div class='source-card'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='section-title'>📄 {row.get('source_filename', 'uploaded_document.pdf')}</div>"
            f"<div class='source-meta'>Pages {row.get('page_count', 0)} · Chunks {row.get('chunk_count', 0)} · Updated {row.get('updated_at', row.get('created_at', '—'))}</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns([1, 1, 1, 1])
        with cols[0]:
            if st.button("Load", key=f"load_doc_{idx}", use_container_width=True):
                try:
                    handle_saved_index_load(doc_id)
                    switch_page("Chat")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to load saved index: {exc}")
        with cols[1]:
            if st.button("View sources", key=f"viewsrc_doc_{idx}", use_container_width=True):
                if st.session_state.active_document_id != doc_id:
                    try:
                        handle_saved_index_load(doc_id)
                    except Exception as exc:
                        st.error(f"Failed to load saved index: {exc}")
                        return
                st.session_state.last_sources = st.session_state.summary_sources or st.session_state.last_sources
                switch_page("Sources")
                st.rerun()
        with cols[2]:
            if st.button("Activate for chat", key=f"activate_doc_{idx}", use_container_width=True):
                try:
                    handle_saved_index_load(doc_id)
                    set_status(f"Activated {row.get('source_filename', 'document')} for chat.")
                    switch_page("Chat")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to activate document: {exc}")
        with cols[3]:
            if st.button("Delete", key=f"delete_doc_{idx}", use_container_width=True):
                delete_index(doc_id)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_sources_page() -> None:
    st.markdown("<div class='section-title'>Retrieved sources</div><div class='section-sub'>Inspect retrieved document chunks and verify the evidence behind every answer.</div>", unsafe_allow_html=True)
    sources = st.session_state.last_sources or st.session_state.summary_sources
    render_source_cards(sources)


def render_analytics_page(indexed_documents: list[dict[str, Any]]) -> None:
    st.markdown("<div class='section-title'>Analytics & observability</div><div class='section-sub'>Track query volume, latency trends, and feedback signals across your Knowledge Studio session.</div>", unsafe_allow_html=True)
    docs_df = pd.DataFrame(indexed_documents)
    total_docs = len(indexed_documents)
    total_chunks = int(docs_df["chunk_count"].sum()) if not docs_df.empty and "chunk_count" in docs_df.columns else 0
    avg_latency = sum(st.session_state.latency_log) / len(st.session_state.latency_log) if st.session_state.latency_log else 0.0
    metrics = st.columns(4)
    with metrics[0]:
        render_metric_card("Documents indexed", str(total_docs), "Across knowledge bases")
    with metrics[1]:
        render_metric_card("Total chunks", str(total_chunks), "Indexed retrieval units")
    with metrics[2]:
        render_metric_card("Avg latency", f"{avg_latency:.2f}s" if avg_latency else "—", "Session average")
    with metrics[3]:
        render_metric_card("Queries", str(len(st.session_state.query_log)), "This session")

    charts = st.columns(2, gap="large")
    with charts[0]:
        st.markdown("<div class='card' style='padding:1rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Query volume</div><div class='section-sub'>Activity distribution across time buckets this session.</div>", unsafe_allow_html=True)
        if st.session_state.query_log:
            qdf = pd.DataFrame(st.session_state.query_log)
            chart_df = qdf.groupby("time").size().reset_index(name="queries").set_index("time")
            st.bar_chart(chart_df)
        else:
            render_empty("📈", "No queries yet", "Start chatting to populate the analytics dashboard.")
        st.markdown("</div>", unsafe_allow_html=True)
    with charts[1]:
        st.markdown("<div class='card' style='padding:1rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Feedback signals</div><div class='section-sub'>Answer quality signals from user ratings.</div>", unsafe_allow_html=True)
        if st.session_state.feedback_log:
            counts = Counter(st.session_state.feedback_log)
            fdf = pd.DataFrame({"count": [counts.get("up", 0), counts.get("down", 0)]}, index=["Helpful", "Needs work"])
            st.bar_chart(fdf)
        else:
            render_empty("🧭", "No feedback yet", "Rate a few answers to see quality signals here.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_settings_page(env_status: dict[str, bool]) -> None:
    st.markdown("<div class='section-title'>Configuration & settings</div><div class='section-sub'>Fine-tune retrieval, generation, and system behavior for your RAG pipeline.</div>", unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    with cols[0]:
        st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Model configuration</div>", unsafe_allow_html=True)
        st.text_input("LLM model", value=config.groq_model, disabled=True)
        st.text_input("Embedding model", value=config.embedding_model, disabled=True)
        st.number_input("Chunk size", value=config.chunk_size, disabled=True)
        st.number_input("Chunk overlap", value=config.chunk_overlap, disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Retrieval controls</div>", unsafe_allow_html=True)
        st.selectbox("Search type", ["MMR", "Similarity"], key="page_search_type")
        st.slider("QA Top-K", 3, 8, key="page_top_k")
        st.toggle("Re-rank retrieved chunks", key="page_rerank")
        st.toggle("Show source citations", key="page_citations")
        st.markdown("</div>", unsafe_allow_html=True)

    action_apply = st.button("Apply retrieval settings", key="apply_settings_btn", use_container_width=False)
    if action_apply:
        st.session_state.cfg_search_type = st.session_state.page_search_type
        st.session_state.cfg_top_k = st.session_state.page_top_k
        st.session_state.cfg_rerank = st.session_state.page_rerank
        st.session_state.cfg_citations = st.session_state.page_citations
        set_status("Updated retrieval settings.")
        st.rerun()

    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button("Generate summary", key="gen_summary_btn", use_container_width=True):
            try:
                generate_summary_action()
                switch_page("Sources")
                st.rerun()
            except AppError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected error while generating summary: {exc}")
    with action_cols[1]:
        if st.button("Clear chat", key="settings_clear_chat", use_container_width=True):
            clear_chat_history()
            st.rerun()
    with action_cols[2]:
        export_text = st.session_state.last_answer or st.session_state.summary_markdown or "No answer generated yet."
        st.download_button("Export conversation", export_text, file_name="knowledge_studio_export.md", key="dl_conversation", use_container_width=True)
    with action_cols[3]:
        if st.button("Delete all indexes", key="delete_all_indexes", use_container_width=True):
            for row in list_indexed_documents(config.vectordb_dir):
                delete_index(row.get("document_id", ""))
            set_status("Deleted all indexes.")
            st.rerun()

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='card' style='padding:1rem;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Environment health</div><div class='section-sub'>A transparent product experience starts with clear system readiness.</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='badge-row'><span class='pill {'green' if env_status['groq_api_key_present'] else 'amber'}'>GROQ_API_KEY {'active' if env_status['groq_api_key_present'] else 'missing'}</span><span class='pill {'amber' if env_status['env_file_exists'] else ''}'>.env {'detected' if env_status['env_file_exists'] else 'missing'}</span><span class='pill'>Theme: {st.session_state.settings_theme}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def update_cfg_from_sidebar() -> None:
    st.session_state.cfg_top_k = st.session_state.sidebar_top_k
    st.session_state.cfg_search_type = st.session_state.sidebar_search_type
    st.session_state.cfg_rerank = st.session_state.sidebar_rerank
    st.session_state.cfg_citations = st.session_state.sidebar_citations


def sync_widget_state_from_cfg() -> None:
    st.session_state.sidebar_top_k = st.session_state.cfg_top_k
    st.session_state.sidebar_search_type = st.session_state.cfg_search_type
    st.session_state.sidebar_rerank = st.session_state.cfg_rerank
    st.session_state.sidebar_citations = st.session_state.cfg_citations
    st.session_state.page_top_k = st.session_state.cfg_top_k
    st.session_state.page_search_type = st.session_state.cfg_search_type
    st.session_state.page_rerank = st.session_state.cfg_rerank
    st.session_state.page_citations = st.session_state.cfg_citations


def main() -> None:
    inject_global_css()
    init_session_state()
    sync_widget_state_from_cfg()

    env_status = get_environment_status(config)
    indexed_documents = list_indexed_documents(config.vectordb_dir)

    load_saved_clicked = render_sidebar(indexed_documents, env_status)
    if load_saved_clicked and st.session_state.selected_saved_index:
        try:
            handle_saved_index_load(st.session_state.selected_saved_index)
            switch_page("Chat")
            st.rerun()
        except AppError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Failed to load saved index: {exc}")

    render_topbar()

    if st.session_state.page == "Chat":
        render_chat_page()
    elif st.session_state.page == "Documents":
        render_documents_page(indexed_documents)
    elif st.session_state.page == "Sources":
        render_sources_page()
    elif st.session_state.page == "Analytics":
        render_analytics_page(indexed_documents)
    elif st.session_state.page == "Settings":
        render_settings_page(env_status)


if __name__ == "__main__":
    main()
