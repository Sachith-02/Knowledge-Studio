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
