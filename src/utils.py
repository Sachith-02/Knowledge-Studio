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
