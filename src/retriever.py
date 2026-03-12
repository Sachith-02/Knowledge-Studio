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
