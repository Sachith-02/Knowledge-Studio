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
