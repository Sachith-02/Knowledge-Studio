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
