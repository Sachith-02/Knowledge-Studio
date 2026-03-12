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
