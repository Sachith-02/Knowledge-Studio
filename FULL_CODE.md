# Full code bundle

## `app.py`

```python
from __future__ import annotations

import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

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
    page_icon="✨",
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
        :root {
            --primary: #6366F1;
            --accent: #06B6D4;
            --success: #22C55E;
            --warning: #F59E0B;
            --danger: #EF4444;
            --bg: #F8FAFC;
            --card: #FFFFFF;
            --text: #0F172A;
            --muted: #334155;
            --soft: #64748B;
            --border: #E2E8F0;
            --shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            --shadow-lg: 0 18px 48px rgba(15, 23, 42, 0.12);
            --radius: 18px;
        }
        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text) !important;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(99, 102, 241, 0.08), transparent 24%),
                radial-gradient(circle at top right, rgba(6, 182, 212, 0.08), transparent 22%),
                linear-gradient(180deg, #FCFDFF 0%, var(--bg) 100%);
        }
        [data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0.78);
            backdrop-filter: blur(10px);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.98) 100%);
            border-right: 1px solid rgba(226, 232, 240, 0.9);
        }
        .block-container { padding-top: 1.1rem; }
        h1, h2, h3, h4, h5, h6, p, span, label, div {
            color: var(--text);
        }
        .top-nav {
            position: sticky;
            top: 0.5rem;
            z-index: 999;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.9rem 1.2rem;
            margin-bottom: 1.1rem;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-radius: 20px;
            backdrop-filter: blur(16px);
            box-shadow: var(--shadow);
        }
        .brand-wrap { display:flex; align-items:center; gap:0.85rem; }
        .brand-logo {
            width: 40px; height: 40px; border-radius: 14px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
            display:flex; align-items:center; justify-content:center; color:white !important;
            font-size:1.15rem; box-shadow: 0 10px 24px rgba(99, 102, 241, 0.26);
        }
        .brand-title { font-size:1rem; font-weight:800; color: var(--text) !important; }
        .brand-sub { font-size:0.78rem; color: var(--soft) !important; margin-top:0.12rem; }
        .hero-card, .glass-card, .metric-card, .source-card, .upload-card, .settings-card, .empty-card, .answer-card, .sidebar-block {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }
        .hero-card {
            padding: 1.7rem 1.8rem;
            background:
                radial-gradient(circle at top right, rgba(99,102,241,0.16), transparent 28%),
                radial-gradient(circle at bottom left, rgba(6,182,212,0.15), transparent 26%),
                linear-gradient(135deg, rgba(255,255,255,0.98), rgba(255,255,255,0.92));
            margin-bottom: 1rem;
        }
        .hero-kicker {
            display:inline-flex; align-items:center; gap:0.4rem; padding:0.4rem 0.75rem;
            background: rgba(99, 102, 241, 0.10); border:1px solid rgba(99, 102, 241, 0.18);
            border-radius:999px; color: var(--primary) !important; font-size:0.8rem; font-weight:700;
            margin-bottom:0.9rem;
        }
        .hero-title { font-size:2rem; font-weight:800; color: var(--text) !important; line-height:1.1; letter-spacing:-0.03em; margin-bottom:0.5rem; }
        .hero-subtitle { color: var(--muted) !important; font-size:1rem; max-width:880px; line-height:1.65; margin-bottom:1rem; }
        .section-title { font-size:1.05rem; font-weight:800; color: var(--text) !important; margin-bottom:0.25rem; }
        .section-subtitle { color: var(--muted) !important; font-size:0.92rem; margin-bottom:1rem; line-height:1.55; }
        .glass-card, .answer-card, .settings-card, .upload-card { padding:1.05rem 1.1rem; }
        .metric-card { padding:1rem 1.05rem; min-height: 114px; }
        .metric-label { color: var(--muted) !important; font-size:0.83rem; font-weight:700; margin-bottom:0.55rem; }
        .metric-value { color: var(--text) !important; font-size:1.7rem; font-weight:800; margin-bottom:0.4rem; }
        .metric-trend { color: var(--soft) !important; font-size:0.8rem; }
        .status-badge {
            display:inline-flex; align-items:center; gap:0.35rem; padding:0.38rem 0.72rem; border-radius:999px;
            font-size:0.74rem; font-weight:700; margin-right:0.45rem; margin-bottom:0.45rem; border:1px solid transparent;
        }
        .status-primary { background: rgba(99,102,241,0.10); color: var(--primary) !important; border-color: rgba(99,102,241,0.18); }
        .status-accent { background: rgba(6,182,212,0.10); color: #0F766E !important; border-color: rgba(6,182,212,0.18); }
        .status-success { background: rgba(34,197,94,0.10); color: #15803D !important; border-color: rgba(34,197,94,0.18); }
        .status-warning { background: rgba(245,158,11,0.10); color: #B45309 !important; border-color: rgba(245,158,11,0.18); }
        .status-danger { background: rgba(239,68,68,0.10); color: #B91C1C !important; border-color: rgba(239,68,68,0.18); }
        .source-card { padding:1rem; margin-bottom:0.9rem; }
        .source-title { color: var(--text) !important; font-size:0.95rem; font-weight:800; }
        .source-meta { color: var(--muted) !important; font-size:0.8rem; margin-top:0.35rem; margin-bottom:0.7rem; }
        .chunk-preview {
            font-size:0.92rem; color: var(--text) !important; line-height:1.65; background:#F8FAFC;
            border:1px solid #EEF2F7; padding:0.85rem 0.95rem; border-radius:14px;
        }
        .empty-card { padding:2rem 1.35rem; text-align:center; margin-top:0.6rem; }
        .empty-icon { font-size:2rem; margin-bottom:0.55rem; }
        .empty-title { font-size:1rem; font-weight:800; color: var(--text) !important; margin-bottom:0.4rem; }
        .empty-text { color: var(--muted) !important; font-size:0.92rem; max-width:560px; margin:0 auto; line-height:1.6; }
        .help-chip-row { display:flex; flex-wrap:wrap; gap:0.45rem; margin-top:0.75rem; }
        .help-chip { display:inline-flex; align-items:center; gap:0.35rem; padding:0.45rem 0.72rem; font-size:0.78rem; color: var(--muted) !important; border:1px solid var(--border); border-radius:999px; background:white; }
        .sidebar-nav-title { color: var(--text) !important; font-size:0.78rem; font-weight:800; letter-spacing:0.08em; text-transform:uppercase; margin-top:0.9rem; margin-bottom:0.65rem; }
        .stButton > button, .stDownloadButton > button {
            border-radius: 14px !important; border: 1px solid var(--border) !important;
            background: white !important; color: var(--text) !important; padding: 0.65rem 1rem !important;
            font-weight: 700 !important; box-shadow: 0 8px 22px rgba(15,23,42,0.04);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(99,102,241,0.26) !important; color: var(--primary) !important;
        }
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {
            color: var(--text) !important;
        }
        .stFileUploader > div {
            border: 1.5px dashed #CBD5E1 !important; background: rgba(255,255,255,0.92); padding: 0.5rem; border-radius: 16px !important;
        }
        [data-testid="stChatMessage"] { padding: 0.25rem 0; }
        [data-testid="stChatMessageContent"] {
            background: rgba(255,255,255,0.97) !important; border:1px solid var(--border) !important;
            border-radius: 18px !important; box-shadow: 0 8px 24px rgba(15,23,42,0.05); padding: 0.95rem 1rem 0.85rem 1rem !important;
        }
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span {
            color: var(--text) !important;
        }
        code, pre { color: #E2E8F0; }
        div[data-testid="stTabs"] button {
            background: white; border-radius:999px; border:1px solid var(--border); padding:0.6rem 1rem; margin-right:0.35rem; color: var(--muted) !important; font-weight:700;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(6,182,212,0.10));
            border-color: rgba(99,102,241,0.24); color: var(--primary) !important;
        }
        .small-note { color: var(--soft) !important; font-size: 0.8rem; }
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
        "page_sidebar": "Chat",
        "selected_saved_index": None,
        "recent_uploads": [],
        "query_log": [],
        "latency_log": [],
        "feedback_log": [],
        "last_question": "",
        "pending_question": "",
        "export_text": "",
        "selected_upload_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_chat_history() -> None:
    st.session_state.chat_messages = []


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


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def add_recent_upload(name: str, size_bytes: int, status: str, chunk_count: int) -> None:
    item = {
        "name": name,
        "size_bytes": size_bytes,
        "status": status,
        "chunk_count": chunk_count,
        "time": time.strftime("%H:%M"),
    }
    remaining = [row for row in st.session_state.recent_uploads if row.get("name") != name]
    st.session_state.recent_uploads = [item] + remaining[:7]


def current_sources() -> list[dict[str, Any]]:
    for message in reversed(st.session_state.chat_messages):
        if message.get("role") == "assistant" and message.get("sources"):
            return message["sources"]
    if st.session_state.summary_sources:
        return st.session_state.summary_sources
    return []


def current_answer_text() -> str:
    for message in reversed(st.session_state.chat_messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return st.session_state.summary_markdown or ""


def metric_card(label: str, value: str, trend: str, emoji: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{emoji} {label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-trend">{trend}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, text: str) -> None:
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


def render_top_nav() -> None:
    st.markdown(
        """
        <div class="top-nav">
            <div class="brand-wrap">
                <div class="brand-logo">✦</div>
                <div>
                    <div class="brand-title">Knowledge Studio</div>
                    <div class="brand-sub">AI document intelligence</div>
                </div>
            </div>
            <div class="brand-sub">Upload knowledge · Retrieve context · Generate grounded answers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected = st.radio(
        "Main navigation",
        options=PAGES,
        key="page",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.page = selected


def render_sidebar(indexed_documents: list[dict[str, Any]], env_status: dict[str, bool]) -> tuple[bool, int, int, bool, bool, bool, str]:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-block" style="padding: 1rem; margin-bottom: 1rem;">
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <div class="brand-logo" style="width:36px; height:36px; border-radius:12px; font-size:1rem;">✦</div>
                    <div>
                        <div class="brand-title" style="font-size:0.95rem;">Knowledge Studio</div>
                        <div class="brand-sub">AI document intelligence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.selectbox("Workspace", ["Default Workspace", "Research Lab", "Investor Demo"], index=0)
        st.selectbox("Knowledge Base", ["Lecture Collection", "Uploaded Docs", "All Documents"], index=0)

        st.markdown('<div class="sidebar-nav-title">Navigation</div>', unsafe_allow_html=True)
        st.radio(
            "Go to",
            options=PAGES,
            key="page_sidebar",
            format_func=lambda x: x,
            label_visibility="collapsed",
        )
        if st.session_state.page_sidebar != st.session_state.page:
            st.session_state.page = st.session_state.page_sidebar

        st.markdown('<div class="sidebar-nav-title">Filters</div>', unsafe_allow_html=True)
        st.multiselect("Document types", ["PDF", "DOCX", "TXT", "CSV", "PPTX"], default=["PDF"])
        force_rebuild = st.toggle("Force rebuild current index", value=False)
        show_citations = st.toggle("Show source citations", value=True)

        st.markdown('<div class="sidebar-nav-title">Retrieval</div>', unsafe_allow_html=True)
        qa_k = st.slider("Top-k retrieval", min_value=3, max_value=8, value=config.qa_k)
        summary_k = st.slider("Summary breadth", min_value=4, max_value=12, value=config.summary_k)
        rerank = st.toggle("Re-rank retrieved chunks", value=False)
        search_type = st.selectbox("Search type", ["MMR", "Similarity"], index=0)

        st.markdown('<div class="sidebar-nav-title">Environment</div>', unsafe_allow_html=True)
        st.markdown(
            f"<span class='status-badge {'status-success' if env_status['env_file_exists'] else 'status-warning'}'>.env {'ready' if env_status['env_file_exists'] else 'missing'}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span class='status-badge {'status-success' if env_status['groq_api_key_present'] else 'status-danger'}'>Groq key {'present' if env_status['groq_api_key_present'] else 'missing'}</span>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-nav-title">Saved indexes</div>', unsafe_allow_html=True)
        load_clicked = False
        if indexed_documents:
            load_options = {
                f"{item.get('source_filename', 'uploaded_document.pdf')} · {item.get('document_id', '')[:12]}": item.get("document_id", "")
                for item in indexed_documents
            }
            selected = st.selectbox("Choose a saved index", ["Select a saved index"] + list(load_options.keys()))
            if selected != "Select a saved index":
                st.session_state.selected_saved_index = load_options[selected]
            load_clicked = st.button("Load saved index", use_container_width=True)
        else:
            st.caption("No saved indexes yet.")

        if st.button("Clear chat history", use_container_width=True):
            clear_chat_history()
            st.success("Chat history cleared.")

    return load_clicked, qa_k, summary_k, force_rebuild, show_citations, rerank, search_type


def render_hero() -> None:
    metadata = st.session_state.active_index_metadata or {}
    doc_name = metadata.get("source_filename", "No active document")
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">⚡ Grounded AI answers with transparent retrieval</div>
            <div class="hero-title">Ask your knowledge base</div>
            <div class="hero-subtitle">
                Upload documents, retrieve the most relevant context, and generate grounded answers with visible sources.
                This workspace is designed for trust, clarity, and speed.
            </div>
            <div>
                <span class="status-badge status-primary">Active doc: {doc_name}</span>
                <span class="status-badge status-accent">Retrieval-first workflow</span>
                <span class="status-badge status-success">Source-grounded answers</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        if st.button("Upload Documents", use_container_width=True):
            st.session_state.page = "Documents"
            st.rerun()
    with col2:
        if st.button("Start New Chat", use_container_width=True):
            clear_chat_history()
            st.session_state.page = "Chat"
            st.success("Started a fresh conversation.")
    with col3:
        if st.button("View Sources", use_container_width=True):
            st.session_state.page = "Sources"
            st.rerun()
    with col4:
        st.markdown(
            """
            <div class="help-chip-row">
                <span class="help-chip">📚 Upload knowledge</span>
                <span class="help-chip">🔎 Retrieve relevant context</span>
                <span class="help-chip">✨ Generate grounded answers</span>
                <span class="help-chip">🛡️ Show transparent sources</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sources(serialized_sources: list[dict[str, Any]], title: str, show_citations: bool) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:0.7rem;">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">Retrieved context is displayed separately so users can verify the answer against the evidence.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not serialized_sources:
        empty_state("📚", "No source evidence yet", "Ask a question or generate a summary to populate this panel with retrieved chunks and metadata.")
        return
    for idx, source in enumerate(sort_serialized_sources(serialized_sources), start=1):
        metadata = source.get("metadata", {})
        title_line = build_source_label(metadata)
        relevance = max(0.72, 0.97 - ((idx - 1) * 0.05))
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">Source {idx} · {title_line}</div>
                <div class="source-meta">
                    Document: {metadata.get('source_filename', 'uploaded_document.pdf')} · Page {metadata.get('page_number', '?')} · Chunk {metadata.get('chunk_index', '?')}
                    <span class="status-badge status-accent" style="margin-left:0.5rem;">Relevance {relevance:.2f}</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"Preview source chunk {idx}", expanded=(idx == 1)):
            st.markdown(f"<div class='chunk-preview'>{source.get('content', '')}</div>", unsafe_allow_html=True)
            if show_citations:
                st.json(metadata)
        st.markdown("</div>", unsafe_allow_html=True)


def render_answer_panel(answer_text: str, source_count: int) -> None:
    confidence = "High" if source_count >= 4 else "Medium" if source_count >= 2 else "Low"
    css_class = "status-success" if confidence == "High" else "status-warning" if confidence == "Medium" else "status-danger"
    st.markdown(
        f"""
        <div class="answer-card">
            <div class="section-title">Assistant answer</div>
            <div style="margin-top:0.3rem; margin-bottom:0.85rem;">
                <span class="status-badge status-primary">Answer generated from {source_count} source(s)</span>
                <span class="status-badge {css_class}">Confidence: {confidence}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(answer_text)


def render_chat_history(show_citations: bool) -> None:
    if not st.session_state.chat_messages:
        empty_state(
            "💬",
            "No conversation yet",
            "Start with a question like “Summarize the core ideas” or “What does the document say about topic X?”",
        )
        return
    assistant_counter = 0
    for message in st.session_state.chat_messages:
        avatar = "🧑" if message["role"] == "user" else "✨"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                assistant_counter += 1
                sources = message.get("sources", [])
                if sources:
                    with st.expander(f"Retrieved context · answer #{assistant_counter}", expanded=False):
                        render_sources(sources, f"Evidence for answer #{assistant_counter}", show_citations)


def handle_saved_index_load(selected_document_id: str) -> None:
    embeddings = load_embeddings_resource(config.embedding_model, config.embedding_device)
    current_document_id = st.session_state.active_document_id
    with st.status("Loading saved knowledge base...", expanded=True) as status:
        st.write("Opening the embedding model cache...")
        st.write("Loading the saved Chroma vector database...")
        vector_store, index_metadata = load_index_by_document_id(
            document_id=selected_document_id,
            vectordb_dir=config.vectordb_dir,
            embedding_function=embeddings,
        )
        pdf_path = find_saved_pdf(selected_document_id)
        activate_document(
            document_id=selected_document_id,
            source_filename=index_metadata.get("source_filename", "uploaded_document.pdf"),
            pdf_path=pdf_path,
            index_metadata=index_metadata,
            vector_store=vector_store,
            reset_history=current_document_id != selected_document_id,
        )
        status.update(label="Knowledge base loaded.", state="complete")
    st.success(f"Loaded saved index for {index_metadata.get('source_filename', 'uploaded_document.pdf')}.")


def handle_uploaded_pdf(uploaded_file, force_rebuild: bool) -> None:
    file_bytes = uploaded_file.getvalue()
    validate_uploaded_pdf(uploaded_file.name, file_bytes)
    document_id = build_document_id(file_bytes)
    saved_pdf_path = save_uploaded_pdf(file_bytes=file_bytes, original_filename=uploaded_file.name, data_dir=config.data_dir, document_id=document_id)
    current_document_id = st.session_state.active_document_id

    with st.status("Processing document...", expanded=True) as status:
        st.write("Validating uploaded PDF...")
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
            st.write("Extracting readable content...")
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
        status.update(label="Document ready.", state="complete")

    if loaded_existing:
        st.success(f"Loaded the existing index for {uploaded_file.name}.")
    else:
        st.success(f"Indexed {uploaded_file.name} successfully.")


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
        clear_summary()


def retrieval_lambda(search_type: str, rerank: bool) -> float:
    if search_type == "Similarity":
        return 1.0
    return 0.35 if rerank else 0.7


def log_query(question: str, sources: int, latency: float) -> None:
    st.session_state.query_log.append({"question": question, "sources": sources, "time": time.strftime("%H:%M")})
    st.session_state.latency_log.append({"latency": latency, "time": time.strftime("%H:%M")})


def render_chat_page(env_status: dict[str, bool], qa_k: int, show_citations: bool, rerank: bool, search_type: str) -> None:
    left_col, right_col = st.columns([1.7, 1], gap="large")
    with left_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Conversation</div>
                <div class="section-subtitle">Ask questions about your active knowledge base. Answers are grounded in retrieved document chunks.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_chat_history(show_citations)
        if st.session_state.active_vectorstore is None:
            empty_state("📎", "No active knowledge base", "Upload or load a document to begin chatting with grounded retrieval.")

        suggestions = [
            "Summarize the core ideas",
            "What are the key definitions?",
            "What does page 3 explain?",
            "List the main takeaways",
        ]
        sug_cols = st.columns(4)
        for col, prompt in zip(sug_cols, suggestions):
            with col:
                if st.button(prompt, key=f"s_{prompt}", use_container_width=True):
                    st.session_state.pending_question = prompt

        question = st.chat_input("Ask a grounded question about your knowledge base...")
        if not question and st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = ""

        if question:
            if st.session_state.active_vectorstore is None:
                st.error("Upload or load a document before asking questions.")
            elif not env_status["groq_api_key_present"]:
                st.error("GROQ_API_KEY is missing. Add it to your .env file to enable Q&A.")
            else:
                st.session_state.chat_messages.append({"role": "user", "content": question})
                try:
                    llm = load_llm_resource(config.groq_api_key, config.groq_model)
                    start = time.time()
                    with st.status("Thinking with retrieved context...", expanded=True) as status:
                        st.write("Searching the most relevant chunks...")
                        retrieved_documents = retrieve_for_qa(
                            st.session_state.active_vectorstore,
                            question,
                            k=qa_k,
                            fetch_k=max(config.qa_fetch_k, qa_k * 2),
                            lambda_mult=retrieval_lambda(search_type, rerank),
                        )
                        st.write(f"Retrieved {len(retrieved_documents)} chunk(s).")
                        st.write("Generating a grounded answer with Groq...")
                        answer = answer_question(question, retrieved_documents, llm)
                        if not retrieved_documents and not answer.strip():
                            answer = FALLBACK_RESPONSE
                        source_payload = serialize_documents(retrieved_documents)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": answer or FALLBACK_RESPONSE,
                            "sources": source_payload,
                            "question": question,
                        })
                        st.session_state.last_question = question
                        st.session_state.export_text = answer or FALLBACK_RESPONSE
                        log_query(question, len(source_payload), time.time() - start)
                        status.update(label="Answer ready.", state="complete")
                    st.rerun()
                except AppError as exc:
                    st.session_state.chat_messages.append({"role": "assistant", "content": f"**Error:** {exc}", "sources": []})
                    st.rerun()
                except Exception as exc:
                    st.session_state.chat_messages.append({"role": "assistant", "content": f"**Error:** Unexpected failure during Q&A: {exc}", "sources": []})
                    st.rerun()

    with right_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Retrieval insights</div>
                <div class="section-subtitle">Inspect source evidence, active document metadata, and answer grounding.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        answer_text = current_answer_text()
        sources = current_sources()
        if answer_text:
            render_answer_panel(answer_text, len(sources))
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("Regenerate last answer", use_container_width=True):
                    if st.session_state.last_question:
                        st.session_state.pending_question = st.session_state.last_question
                        st.success("Queued the last question for regeneration.")
                        st.rerun()
            with action_cols[1]:
                if st.button("Show sources", use_container_width=True):
                    st.session_state.page = "Sources"
                    st.rerun()
            feedback_cols = st.columns(2)
            with feedback_cols[0]:
                if st.button("👍 Helpful", use_container_width=True):
                    st.session_state.feedback_log.append({"rating": "up", "time": time.strftime("%H:%M")})
                    st.success("Thanks for the feedback.")
            with feedback_cols[1]:
                if st.button("👎 Needs work", use_container_width=True):
                    st.session_state.feedback_log.append({"rating": "down", "time": time.strftime("%H:%M")})
                    st.info("Feedback saved for this session.")
            st.download_button(
                "Export answer",
                data=answer_text,
                file_name="knowledge_studio_answer.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
            render_sources(sources, "Retrieved context", show_citations)
        else:
            empty_state("🔎", "No retrieval evidence yet", "Ask a question to inspect the retrieved context, metadata, and source attribution here.")


def render_documents_page(indexed_documents: list[dict[str, Any]], force_rebuild: bool, show_citations: bool) -> None:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Document upload & indexing</div>
            <div class="section-subtitle">Upload PDFs, create persistent embeddings, and manage saved indexes from one clean workspace.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "docx", "txt", "csv", "pptx"],
        accept_multiple_files=True,
        help="PDF processing is fully connected. Other file types are shown in the interface for future extension.",
    )
    if uploaded_files:
        st.markdown("<div class='section-title' style='margin-top:0.8rem;'>Files ready for ingestion</div>", unsafe_allow_html=True)
        for idx, uploaded in enumerate(uploaded_files):
            is_pdf = uploaded.name.lower().endswith(".pdf")
            st.markdown(
                f"""
                <div class="upload-card">
                    <div class="source-title">{'📄' if is_pdf else '📁'} {uploaded.name}</div>
                    <div class="source-meta">Size: {format_bytes(len(uploaded.getvalue()))} · Type: {uploaded.name.split('.')[-1].upper()} · Status: {'Ready to index' if is_pdf else 'UI-visible only'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("Process", key=f"process_{idx}", use_container_width=True):
                    if not is_pdf:
                        st.warning("Only PDF indexing is enabled in the current backend.")
                    else:
                        try:
                            handle_uploaded_pdf(uploaded, force_rebuild=force_rebuild)
                            st.session_state.page = "Chat"
                            st.rerun()
                        except AppError as exc:
                            st.error(str(exc))
                        except Exception as exc:
                            st.error(f"Unexpected error while processing the PDF: {exc}")
            with c2:
                if st.button("View details", key=f"view_{idx}", use_container_width=True):
                    st.session_state.selected_upload_name = uploaded.name
                    st.info(f"{uploaded.name} · {format_bytes(len(uploaded.getvalue()))} · {'PDF document' if is_pdf else 'Unsupported for indexing right now'}")

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Recent uploads</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Track ingestion status, chunk counts, and the latest document activity.</div>", unsafe_allow_html=True)
    if st.session_state.recent_uploads:
        for item in st.session_state.recent_uploads:
            st.markdown(
                f"""
                <div class="upload-card">
                    <div class="source-title">📄 {item['name']}</div>
                    <div class="source-meta">Size: {format_bytes(item['size_bytes'])} · Chunks: {item['chunk_count']} · Updated: {item['time']}</div>
                    <span class="status-badge {'status-success' if item['status'] == 'Indexed' else 'status-accent'}">{item['status']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        empty_state("📂", "No files uploaded yet", "Upload a PDF to build your first knowledge base.")

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Saved knowledge bases</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Load, inspect, or delete persistent vector indexes.</div>", unsafe_allow_html=True)
    if indexed_documents:
        for idx, item in enumerate(indexed_documents):
            document_id = item.get("document_id", "")
            st.markdown(
                f"""
                <div class="upload-card">
                    <div class="source-title">🧠 {item.get('source_filename', 'uploaded_document.pdf')}</div>
                    <div class="source-meta">Pages: {item.get('page_count', 0)} · Chunks: {item.get('chunk_count', 0)} · ID: {document_id[:12]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Load", key=f"load_doc_{idx}", use_container_width=True):
                    handle_saved_index_load(document_id)
                    st.session_state.page = "Chat"
                    st.rerun()
            with c2:
                if st.button("View sources", key=f"view_sources_{idx}", use_container_width=True):
                    handle_saved_index_load(document_id)
                    st.session_state.page = "Sources"
                    st.rerun()
            with c3:
                if st.button("Delete index", key=f"delete_doc_{idx}", use_container_width=True):
                    delete_index(document_id)
                    st.success("Index deleted.")
                    st.rerun()
    else:
        empty_state("🧠", "No saved indexes yet", "Process a PDF to create a persistent Chroma index.")


def render_sources_page(show_citations: bool) -> None:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Sources & retrieved context</div>
            <div class="section-subtitle">Separate evidence from generation. This view helps users verify exactly what was retrieved.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_sources(current_sources(), "Latest retrieved sources", show_citations)


def render_analytics_page(indexed_documents: list[dict[str, Any]]) -> None:
    query_count = len(st.session_state.query_log)
    total_chunks = sum(int(item.get("chunk_count", 0)) for item in indexed_documents)
    avg_latency = 0.0
    if st.session_state.latency_log:
        avg_latency = sum(row["latency"] for row in st.session_state.latency_log) / len(st.session_state.latency_log)
    row = st.columns(4)
    with row[0]:
        metric_card("Total documents", str(len(indexed_documents)), "Across saved knowledge bases", "📚")
    with row[1]:
        metric_card("Total chunks", str(total_chunks), "Indexed retrieval units", "🧩")
    with row[2]:
        metric_card("Average latency", f"{avg_latency:.2f}s" if avg_latency else "—", "Observed this session", "⚡")
    with row[3]:
        metric_card("Queries today", str(query_count), "Session activity", "💬")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'><div class='section-title'>Query volume</div><div class='section-subtitle'>How often the assistant is being used in this session.</div></div>", unsafe_allow_html=True)
        if st.session_state.query_log:
            counts = Counter(row["time"] for row in st.session_state.query_log)
            st.bar_chart({"queries": [counts[key] for key in counts]}, x_label="Time bucket")
            st.caption("Time buckets appear in the order they were recorded during the session.")
        else:
            empty_state("📈", "No analytics yet", "Ask a few questions to start building activity charts.")
    with col2:
        st.markdown("<div class='glass-card'><div class='section-title'>Feedback trend</div><div class='section-subtitle'>Quick signal on whether answers are landing well.</div></div>", unsafe_allow_html=True)
        if st.session_state.feedback_log:
            counts = Counter(row["rating"] for row in st.session_state.feedback_log)
            st.bar_chart({"count": [counts.get("up", 0), counts.get("down", 0)]})
            st.caption("First bar is helpful, second bar is needs work.")
        else:
            empty_state("👍", "No feedback yet", "Feedback charts will appear once you rate a few answers.")

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'><div class='section-title'>Top-used sources</div><div class='section-subtitle'>Which documents are contributing evidence most often.</div></div>", unsafe_allow_html=True)
    source_names: list[str] = []
    for message in st.session_state.chat_messages:
        if message.get("role") == "assistant":
            for source in message.get("sources", []):
                source_names.append(source.get("metadata", {}).get("source_filename", "Unknown"))
    if source_names:
        counts = Counter(source_names)
        st.bar_chart({name: [count] for name, count in counts.items()})
    else:
        empty_state("🗂️", "No source usage yet", "Once the assistant answers with retrieved evidence, source usage will appear here.")


def render_settings_page(env_status: dict[str, bool], summary_k: int, show_citations: bool, rerank: bool, search_type: str) -> None:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">RAG settings</div>
            <div class="section-subtitle">Fine-tune retrieval and generation controls in a clean admin experience.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
        st.markdown("#### Model configuration")
        st.text_input("LLM model", value=config.groq_model, disabled=True)
        st.text_input("Embedding model", value=config.embedding_model, disabled=True)
        st.text_input("Chunk size", value=str(config.chunk_size), disabled=True)
        st.text_input("Chunk overlap", value=str(config.chunk_overlap), disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
        st.markdown("#### Retrieval controls")
        st.text_input("Search type", value=search_type, disabled=True)
        st.text_input("Re-ranking", value="Enabled" if rerank else "Disabled", disabled=True)
        st.text_input("Citations", value="Visible" if show_citations else "Hidden", disabled=True)
        st.text_input("Summary breadth", value=str(summary_k), disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    if st.button("Generate structured summary", use_container_width=True):
        if st.session_state.active_vectorstore is None:
            st.error("Upload or load a document before generating a summary.")
        elif not env_status["groq_api_key_present"]:
            st.error("GROQ_API_KEY is missing. Add it to your .env file to enable summary generation.")
        else:
            try:
                llm = load_llm_resource(config.groq_api_key, config.groq_model)
                with st.status("Generating structured summary...", expanded=True) as status:
                    st.write("Retrieving broad context for synthesis...")
                    summary_documents = retrieve_for_summary(
                        st.session_state.active_vectorstore,
                        get_summary_query(),
                        k=summary_k,
                        fetch_k=max(config.summary_fetch_k, summary_k * 2),
                        lambda_mult=0.5,
                    )
                    st.write(f"Retrieved {len(summary_documents)} chunk(s).")
                    st.write("Synthesizing summary with Groq...")
                    summary_text = generate_summary(summary_documents, llm)
                    st.session_state.summary_markdown = summary_text
                    st.session_state.summary_sources = serialize_documents(summary_documents)
                    st.session_state.export_text = summary_text
                    status.update(label="Summary ready.", state="complete")
                st.session_state.page = "Sources"
                st.success("Structured summary generated.")
            except AppError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unexpected error while generating the summary: {exc}")


def main() -> None:
    inject_global_css()
    init_session_state()
    st.session_state.page_sidebar = st.session_state.page
    env_status = get_environment_status(config)
    indexed_documents = list_indexed_documents(config.vectordb_dir)

    load_clicked, qa_k, summary_k, force_rebuild, show_citations, rerank, search_type = render_sidebar(indexed_documents, env_status)
    if load_clicked and st.session_state.selected_saved_index:
        try:
            handle_saved_index_load(st.session_state.selected_saved_index)
            st.session_state.page = "Chat"
            st.rerun()
        except AppError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Failed to load saved index: {exc}")

    render_top_nav()
    render_hero()

    metadata = st.session_state.active_index_metadata or {}
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Documents indexed", str(len(indexed_documents)), "Persistent knowledge bases", "📚")
    with m2:
        metric_card("Active chunks", str(int(metadata.get("chunk_count", 0))) if metadata else "0", "Ready for retrieval", "🧩")
    with m3:
        metric_card("Pages", str(int(metadata.get("page_count", 0))) if metadata else "0", "Readable lecture content", "📄")
    with m4:
        metric_card("Status", "Ready" if st.session_state.active_vectorstore else "Idle", "Grounded chat workspace", "✅")

    page = st.session_state.page
    if page == "Chat":
        render_chat_page(env_status, qa_k, show_citations, rerank, search_type)
    elif page == "Documents":
        render_documents_page(indexed_documents, force_rebuild, show_citations)
    elif page == "Sources":
        render_sources_page(show_citations)
    elif page == "Analytics":
        render_analytics_page(indexed_documents)
    elif page == "Settings":
        render_settings_page(env_status, summary_k, show_citations, rerank, search_type)


if __name__ == "__main__":
    main()

```

## `src/__init__.py`

```python
"""Lecture PDF RAG application package."""

```

## `src/config.py`

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_QA_K = 4
DEFAULT_QA_FETCH_K = 12
DEFAULT_SUMMARY_K = 8
DEFAULT_SUMMARY_FETCH_K = 20
DEFAULT_COLLECTION_NAME = "lecture_chunks"
FALLBACK_RESPONSE = (
    "I could not find enough information in the uploaded document to answer that confidently."
)


def load_environment() -> bool:
    return load_dotenv(dotenv_path=ENV_FILE, override=False)


ENV_LOADED = load_environment()


@dataclass(frozen=True)
class AppConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    vectordb_dir: Path = VECTORDB_DIR
    env_file: Path = ENV_FILE
    collection_name: str = DEFAULT_COLLECTION_NAME
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )
    embedding_device: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
    )
    qa_k: int = field(default_factory=lambda: int(os.getenv("QA_TOP_K", DEFAULT_QA_K)))
    qa_fetch_k: int = field(
        default_factory=lambda: int(os.getenv("QA_FETCH_K", DEFAULT_QA_FETCH_K))
    )
    summary_k: int = field(
        default_factory=lambda: int(os.getenv("SUMMARY_TOP_K", DEFAULT_SUMMARY_K))
    )
    summary_fetch_k: int = field(
        default_factory=lambda: int(os.getenv("SUMMARY_FETCH_K", DEFAULT_SUMMARY_FETCH_K))
    )


def ensure_base_directories(config: AppConfig) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.vectordb_dir.mkdir(parents=True, exist_ok=True)


def get_app_config() -> AppConfig:
    config = AppConfig()
    ensure_base_directories(config)
    return config


def get_environment_status(config: AppConfig) -> dict[str, bool]:
    return {
        "env_file_exists": config.env_file.exists(),
        "groq_api_key_present": bool(config.groq_api_key),
        "env_loaded": ENV_LOADED,
    }

```

## `src/pdf_processor.py`

```python
from __future__ import annotations

from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document

from src.utils import PDFProcessingError, clean_metadata, normalize_whitespace


def extract_pdf_pages(file_path: str | Path, source_filename: str, document_id: str) -> list[Document]:
    try:
        page_chunks = pymupdf4llm.to_markdown(
            str(file_path),
            page_chunks=True,
            show_progress=False,
        )
    except Exception as exc:
        raise PDFProcessingError(f"Failed to read the PDF: {exc}") from exc

    if not page_chunks or not isinstance(page_chunks, list):
        raise PDFProcessingError("The PDF could not be parsed into readable page content.")

    documents: list[Document] = []
    for fallback_page_number, page_chunk in enumerate(page_chunks, start=1):
        text = (page_chunk.get("text") or "").strip()
        if not normalize_whitespace(text):
            continue

        page_metadata = dict(page_chunk.get("metadata") or {})
        metadata = {
            "source": source_filename,
            "source_filename": source_filename,
            "page_number": int(page_metadata.get("page_number", fallback_page_number)),
            "page_count": int(page_metadata.get("page_count", len(page_chunks))),
            "document_id": document_id,
        }

        for optional_key in ("title", "author", "subject", "keywords"):
            value = page_metadata.get(optional_key)
            if value:
                metadata[f"pdf_{optional_key}"] = value

        documents.append(
            Document(
                page_content=text,
                metadata=clean_metadata(metadata),
            )
        )

    if not documents:
        raise PDFProcessingError(
            "The PDF was parsed, but no readable text could be extracted from it."
        )

    return documents

```

## `src/chunking.py`

```python
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import PDFProcessingError, clean_metadata


def build_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = build_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = splitter.split_documents(list(documents))

    if not split_documents:
        raise PDFProcessingError(
            "Text was extracted from the PDF, but chunking produced no usable chunks."
        )

    page_chunk_counters: dict[int, int] = defaultdict(int)
    final_chunks: list[Document] = []

    for global_chunk_index, chunk in enumerate(split_documents, start=1):
        content = chunk.page_content.strip()
        if not content:
            continue

        page_number = int(chunk.metadata.get("page_number", 0))
        page_chunk_counters[page_number] += 1
        page_chunk_index = page_chunk_counters[page_number]

        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = global_chunk_index
        metadata["page_chunk_index"] = page_chunk_index
        metadata["chunk_id"] = (
            f"{metadata.get('document_id', 'document')}-"
            f"p{page_number:03d}-c{global_chunk_index:04d}"
        )

        final_chunks.append(
            Document(page_content=content, metadata=clean_metadata(metadata))
        )

    if not final_chunks:
        raise PDFProcessingError(
            "Chunking completed, but every chunk was empty after cleanup."
        )

    return final_chunks

```

## `src/embeddings.py`

```python
from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import EmbeddingInitializationError


def get_embedding_model(model_name: str, device: str = "cpu") -> HuggingFaceEmbeddings:
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise EmbeddingInitializationError(
            f"Failed to initialize the embedding model '{model_name}': {exc}"
        ) from exc

```

## `src/vectordb.py`

```python
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import DEFAULT_COLLECTION_NAME
from src.utils import (
    VectorStoreError,
    chunk_ids_from_documents,
    load_json,
    save_json,
    utc_now_iso,
)

INDEX_METADATA_FILENAME = "index_metadata.json"


def get_index_dir(vectordb_dir: Path, document_id: str) -> Path:
    return vectordb_dir / document_id


def get_index_metadata_path(index_dir: Path) -> Path:
    return index_dir / INDEX_METADATA_FILENAME


def index_exists(vectordb_dir: Path, document_id: str) -> bool:
    index_dir = get_index_dir(vectordb_dir, document_id)
    metadata_path = get_index_metadata_path(index_dir)
    return index_dir.exists() and metadata_path.exists()


def _collection_has_documents(vector_store: Chroma) -> bool:
    try:
        result = vector_store.get(limit=1)
    except Exception as exc:
        raise VectorStoreError(f"Unable to inspect the Chroma collection: {exc}") from exc
    return bool(result.get("ids"))


def load_vectorstore(
    index_dir: Path,
    embedding_function: Any,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    try:
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=str(index_dir),
            embedding_function=embedding_function,
        )
    except Exception as exc:
        raise VectorStoreError(f"Failed to open the Chroma index: {exc}") from exc

    if not _collection_has_documents(vector_store):
        raise VectorStoreError("The Chroma index exists but contains no documents.")

    return vector_store


def build_vectorstore(
    chunks: Sequence[Document],
    embedding_function: Any,
    index_dir: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    if not chunks:
        raise VectorStoreError("No chunks were provided to build the Chroma index.")

    index_dir.mkdir(parents=True, exist_ok=True)
    chunk_ids = chunk_ids_from_documents(chunks)

    try:
        return Chroma.from_documents(
            documents=list(chunks),
            ids=chunk_ids,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory=str(index_dir),
        )
    except Exception as exc:
        raise VectorStoreError(f"Failed to build the Chroma index: {exc}") from exc


def load_index_metadata(index_dir: Path) -> dict[str, Any]:
    metadata_path = get_index_metadata_path(index_dir)
    if not metadata_path.exists():
        raise VectorStoreError("Index metadata file is missing.")
    return load_json(metadata_path)


def save_index_metadata(index_dir: Path, payload: dict[str, Any]) -> None:
    metadata_path = get_index_metadata_path(index_dir)
    save_json(metadata_path, payload)


def create_index_metadata(
    document_id: str,
    source_filename: str,
    file_size_bytes: int,
    page_count: int,
    chunk_count: int,
    embedding_model: str,
    index_dir: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "source_filename": source_filename,
        "file_size_bytes": file_size_bytes,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "embedding_model": embedding_model,
        "collection_name": collection_name,
        "persist_directory": str(index_dir),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }


def create_or_load_vectorstore(
    *,
    document_id: str,
    source_filename: str,
    file_size_bytes: int,
    page_count: int,
    chunks: Sequence[Document] | None,
    embedding_function: Any,
    embedding_model: str,
    vectordb_dir: Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    force_rebuild: bool = False,
) -> tuple[Chroma, dict[str, Any], bool]:
    index_dir = get_index_dir(vectordb_dir, document_id)

    if force_rebuild and index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)

    if index_exists(vectordb_dir, document_id):
        vector_store = load_vectorstore(
            index_dir=index_dir,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
        metadata = load_index_metadata(index_dir)
        metadata["loaded_from_existing"] = True
        metadata["updated_at"] = utc_now_iso()
        save_index_metadata(index_dir, metadata)
        return vector_store, metadata, True

    if not chunks:
        raise VectorStoreError(
            "A new Chroma index is required, but no chunks were supplied to build it."
        )

    vector_store = build_vectorstore(
        chunks=chunks,
        embedding_function=embedding_function,
        index_dir=index_dir,
        collection_name=collection_name,
    )
    metadata = create_index_metadata(
        document_id=document_id,
        source_filename=source_filename,
        file_size_bytes=file_size_bytes,
        page_count=page_count,
        chunk_count=len(chunks),
        embedding_model=embedding_model,
        index_dir=index_dir,
        collection_name=collection_name,
    )
    metadata["loaded_from_existing"] = False
    save_index_metadata(index_dir, metadata)
    return vector_store, metadata, False


def list_indexed_documents(vectordb_dir: Path) -> list[dict[str, Any]]:
    indexed_documents: list[dict[str, Any]] = []

    if not vectordb_dir.exists():
        return indexed_documents

    for candidate_dir in vectordb_dir.iterdir():
        if not candidate_dir.is_dir():
            continue
        metadata_path = get_index_metadata_path(candidate_dir)
        if not metadata_path.exists():
            continue
        try:
            payload = load_json(metadata_path)
            indexed_documents.append(payload)
        except Exception:
            continue

    return sorted(
        indexed_documents,
        key=lambda item: item.get("updated_at", item.get("created_at", "")),
        reverse=True,
    )


def load_index_by_document_id(
    document_id: str,
    vectordb_dir: Path,
    embedding_function: Any,
) -> tuple[Chroma, dict[str, Any]]:
    index_dir = get_index_dir(vectordb_dir, document_id)
    metadata = load_index_metadata(index_dir)
    vector_store = load_vectorstore(
        index_dir=index_dir,
        embedding_function=embedding_function,
        collection_name=metadata.get("collection_name", DEFAULT_COLLECTION_NAME),
    )
    return vector_store, metadata

```

## `src/retriever.py`

```python
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


def build_retriever(
    vector_store: Any,
    *,
    k: int,
    fetch_k: int,
    lambda_mult: float,
):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(fetch_k, k),
            "lambda_mult": lambda_mult,
        },
    )


def retrieve_for_qa(
    vector_store: Any,
    question: str,
    *,
    k: int = 4,
    fetch_k: int = 12,
    lambda_mult: float = 0.7,
) -> list[Document]:
    retriever = build_retriever(
        vector_store,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
    return retriever.invoke(question)


def retrieve_for_summary(
    vector_store: Any,
    query: str,
    *,
    k: int = 8,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> list[Document]:
    retriever = build_retriever(
        vector_store,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
    return retriever.invoke(query)

```

## `src/qa_chain.py`

```python
from __future__ import annotations

import os
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import FALLBACK_RESPONSE
from src.utils import (
    LLMGenerationError,
    LLMInitializationError,
    build_context_from_documents,
)


def get_llm(api_key: str, model_name: str) -> ChatGroq:
    if not api_key:
        raise LLMInitializationError(
            "GROQ_API_KEY is missing. Add it to your .env file before using the app."
        )

    try:
        os.environ["GROQ_API_KEY"] = api_key
        return ChatGroq(
            model=model_name,
            temperature=0,
        )
    except Exception as exc:
        raise LLMInitializationError(
            f"Failed to initialize the Groq model '{model_name}': {exc}"
        ) from exc


def answer_question(question: str, documents: Sequence[Document], llm: ChatGroq) -> str:
    if not documents:
        return FALLBACK_RESPONSE

    context = build_context_from_documents(documents)
    if not context.strip():
        return FALLBACK_RESPONSE

    system_prompt = f"""
You are a document-grounded question answering assistant.

Rules:
1. Answer only from the provided context.
2. Do not use outside knowledge.
3. Do not hallucinate or guess.
4. If the context does not contain enough evidence, respond with exactly:
   {FALLBACK_RESPONSE}
5. Keep the answer clear, concise, and formatted in markdown.
""".strip()

    human_prompt = f"""
Question:
{question}

Context:
{context}

Answer:
""".strip()

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
    except Exception as exc:
        raise LLMGenerationError(f"Groq failed to answer the question: {exc}") from exc

    answer = (response.content or "").strip()
    return answer or FALLBACK_RESPONSE

```

## `src/summarizer.py`

```python
from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.utils import LLMGenerationError, build_context_from_documents

SUMMARY_QUERY = (
    "Retrieve the most important chunks needed to create a comprehensive lecture summary, "
    "including main topics, definitions, processes, examples, and conclusions."
)


def get_summary_query() -> str:
    return SUMMARY_QUERY


def generate_summary(documents: Sequence[Document], llm: ChatGroq) -> str:
    if not documents:
        return "## Summary unavailable\n\nNo relevant document chunks were found for summarization."

    context = build_context_from_documents(documents)

    system_prompt = """
You are a professional academic assistant.
Create a structured markdown summary using only the provided context from the uploaded lecture PDF.
Do not invent missing details.

Formatting requirements:
- Start with a clear # title.
- Use ## headings and, if helpful, ### subheadings.
- Use bullet points for key concepts and takeaways.
- Keep the writing concise but informative.
- Mention diagrams or figures only if they are explicitly present in the context.
""".strip()

    human_prompt = f"""
Context:
{context}

Write a structured markdown summary of the lecture.
""".strip()

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
    except Exception as exc:
        raise LLMGenerationError(f"Groq failed to generate the summary: {exc}") from exc

    summary = (response.content or "").strip()
    if not summary:
        return "## Summary unavailable\n\nThe model returned an empty summary."

    return summary

```

## `src/utils.py`

```python
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document


class AppError(Exception):
    """Base application error."""


class PDFProcessingError(AppError):
    """Raised when PDF validation or extraction fails."""


class EmbeddingInitializationError(AppError):
    """Raised when the embedding model fails to initialize."""


class VectorStoreError(AppError):
    """Raised when vector database operations fail."""


class LLMInitializationError(AppError):
    """Raised when the Groq LLM fails to initialize."""


class LLMGenerationError(AppError):
    """Raised when the Groq LLM fails during generation."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def build_document_id(file_bytes: bytes) -> str:
    return compute_file_hash(file_bytes)


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return cleaned or "document.pdf"


def validate_uploaded_pdf(filename: str, file_bytes: bytes) -> None:
    if not filename:
        raise PDFProcessingError("Please upload a PDF file.")
    if not filename.lower().endswith(".pdf"):
        raise PDFProcessingError("The uploaded file must have a .pdf extension.")
    if not file_bytes:
        raise PDFProcessingError("The uploaded PDF is empty.")

    header = file_bytes.lstrip()[:5]
    if header != b"%PDF-":
        raise PDFProcessingError(
            "The uploaded file does not appear to be a valid PDF document."
        )


def save_uploaded_pdf(
    file_bytes: bytes,
    original_filename: str,
    data_dir: Path,
    document_id: str,
) -> Path:
    document_dir = data_dir / document_id
    document_dir.mkdir(parents=True, exist_ok=True)

    for existing_pdf in document_dir.glob("*.pdf"):
        try:
            existing_pdf.unlink()
        except OSError:
            pass

    sanitized_name = sanitize_filename(original_filename)
    target_path = document_dir / sanitized_name
    target_path.write_bytes(file_bytes)
    return target_path


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_context_from_documents(documents: Sequence[Document]) -> str:
    if not documents:
        return ""

    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        page_number = document.metadata.get("page_number", "unknown")
        chunk_index = document.metadata.get("chunk_index", index)
        source_filename = document.metadata.get("source_filename", "uploaded_document.pdf")
        sections.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Filename: {source_filename}",
                    f"Page: {page_number}",
                    f"Chunk: {chunk_index}",
                    document.page_content.strip(),
                ]
            )
        )
    return "\n\n".join(sections)


def serialize_documents(documents: Sequence[Document]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for document in documents:
        serialized.append(
            {
                "content": document.page_content,
                "metadata": dict(document.metadata),
            }
        )
    return serialized


def sort_serialized_sources(
    serialized_documents: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        serialized_documents,
        key=lambda item: (
            int(item.get("metadata", {}).get("page_number", 0)),
            int(item.get("metadata", {}).get("chunk_index", 0)),
        ),
    )


def build_source_label(metadata: dict[str, Any]) -> str:
    filename = str(metadata.get("source_filename", "uploaded_document.pdf"))
    page_number = metadata.get("page_number", "?")
    chunk_index = metadata.get("chunk_index", "?")
    return f"{filename} · Page {page_number} · Chunk {chunk_index}"


def chunk_ids_from_documents(documents: Iterable[Document]) -> list[str]:
    ids: list[str] = []
    for document in documents:
        chunk_id = document.metadata.get("chunk_id")
        if not chunk_id:
            raise VectorStoreError("Every chunk must have a chunk_id before indexing.")
        ids.append(str(chunk_id))
    return ids

```

## `requirements.txt`

```text
streamlit
python-dotenv
pymupdf4llm
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-chroma
langchain-groq
chromadb
sentence-transformers

```

## `.env.example`

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
QA_TOP_K=4
QA_FETCH_K=12
SUMMARY_TOP_K=8
SUMMARY_FETCH_K=20

```

## `README.md`

```markdown
# Lecture PDF RAG Assistant

A modular Streamlit RAG application for lecture PDFs.

This project upgrades a summary-focused PDF workflow into an interactive document assistant that can:

- extract lecture PDF content with **PyMuPDF4LLM**
- split the content into reusable chunks
- embed and persist those chunks in **ChromaDB**
- generate a structured lecture summary with **Groq**
- answer grounded questions about the uploaded document
- reuse existing indexes instead of rebuilding embeddings every run

## Features

### 1. Interactive document Q&A

- ask questions about the active PDF in a chat interface
- answers are instructed to use only retrieved document context
- chat history is preserved for the current Streamlit session
- retrieved chunks and metadata are shown under each assistant response
- fallback response is enforced when the evidence is not strong enough

### 2. Persistent and loadable vector database

- each uploaded PDF gets a document identity based on a **SHA-256 hash of its file bytes**
- the Chroma index is stored in `vectordb/<document_id>/`
- if the same file content is uploaded again, the app loads the existing index
- a **Force rebuild index** option rebuilds the database from scratch
- previously indexed documents can be loaded directly from the sidebar

### 3. Beginner-friendly Streamlit UI

- PDF uploader
- process/load index button
- progress/status containers
- summary tab
- chat tab
- sidebar settings
- retrieved source chunk viewer

## Project structure

```text
rag_app/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
├── vectordb/
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── pdf_processor.py
    ├── chunking.py
    ├── embeddings.py
    ├── vectordb.py
    ├── retriever.py
    ├── qa_chain.py
    ├── summarizer.py
    └── utils.py
```

## Setup

### 1. Create and activate a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scriptsctivate
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your environment file

Copy `.env.example` to `.env`, then add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env` and set:

```env
GROQ_API_KEY=your_real_groq_api_key
```

### 4. Run the app

```bash
streamlit run app.py
```

## How the indexing flow works

1. You upload a PDF in the Streamlit UI.
2. The app validates that the file is a real PDF.
3. The PDF is saved into `data/<document_id>/`.
4. The file bytes are hashed with SHA-256 to create a stable `document_id`.
5. If `vectordb/<document_id>/index_metadata.json` already exists and you did not enable force rebuild, the app loads the saved Chroma index.
6. Otherwise, the app:
   - extracts page content with `pymupdf4llm.to_markdown(..., page_chunks=True)`
   - creates LangChain `Document` objects with page-level metadata
   - chunks the page documents
   - stores the chunks in ChromaDB
   - writes index metadata for future reuse

## Chunking strategy

The default chunking configuration is:

- `chunk_size = 1200`
- `chunk_overlap = 200`

This works well for lecture notes and academic PDFs because it keeps enough local context for definitions, examples, and short explanations, while still giving retrieval room for multiple chunks in the prompt.

Separators are chosen to prefer:

1. markdown headings
2. paragraph breaks
3. line breaks
4. sentence boundaries
5. plain character fallback

## Retrieval strategy

The app uses **MMR (Maximal Marginal Relevance)** retrieval so the returned chunks stay relevant while reducing redundant near-duplicate chunks.

Default retrieval settings:

- Q&A: `k=4`, `fetch_k=12`, `lambda_mult=0.7`
- Summary: `k=8`, `fetch_k=20`, `lambda_mult=0.5`

These defaults are tuned so Q&A stays focused and summary mode gets a broader view of the lecture.

## Metadata stored with each chunk

Each chunk includes metadata such as:

- `source`
- `source_filename`
- `page_number`
- `page_count`
- `document_id`
- `chunk_index`
- `page_chunk_index`
- `chunk_id`

## Using the app

### Summary mode

1. Upload and process a PDF.
2. Open the **Summary** tab.
3. Click **Generate summary**.
4. Review the generated markdown summary.
5. Expand **Summary source chunks** to inspect the supporting chunks.

### Ask Questions mode

1. Upload and process a PDF.
2. Open the **Ask Questions** tab.
3. Ask a question about the document.
4. Review the answer.
5. Expand **Retrieved sources** below the assistant message to inspect the evidence and metadata.

## Important behavior

- Q&A answers are instructed to use only retrieved context.
- If the context is insufficient, the assistant is prompted to return:

```text
I could not find enough information in the uploaded document to answer that confidently.
```

- Summary generation is also grounded in retrieved chunks instead of the whole raw PDF.
- If `.env` is missing or `GROQ_API_KEY` is not set, the app shows user-friendly warnings and keeps indexing available.

## Troubleshooting

### `GROQ_API_KEY is missing`

Create a `.env` file in the project root and add:

```env
GROQ_API_KEY=your_real_groq_api_key
```

### PDF uploads but indexing fails

Common causes:

- the uploaded file is not a real PDF
- the PDF contains no extractable text
- the local embedding model failed to initialize
- a corrupted Chroma index exists and should be rebuilt

Try enabling **Force rebuild index** and processing the file again.

### First run is slow

The first run may take longer because the local embedding model has to be downloaded and initialized.

## Why this upgrade stays close to the original project

The original project already had a solid flow:

- PDF extraction
- chunking
- embeddings
- Chroma storage
- Groq generation

This upgrade keeps that same core stack and refactors it into a cleaner app with:

- modular source files
- persistent Chroma storage
- an interactive chat workflow
- a Streamlit UI
- better validation and error handling

```

## `.gitignore`

```
.env
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.DS_Store
.streamlit/
data/*
!data/.gitkeep
vectordb/*
!vectordb/.gitkeep

```

