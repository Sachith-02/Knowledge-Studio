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
