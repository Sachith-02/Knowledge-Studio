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
