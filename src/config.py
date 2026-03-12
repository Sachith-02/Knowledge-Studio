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
