# Knowledge Studio

A dark, premium Streamlit RAG app for lecture PDFs and document-grounded Q&A.

## What it does

- Upload a PDF and build a persistent Chroma index
- Reuse saved indexes without rebuilding embeddings every run
- Ask grounded questions about the active document
- Generate a structured summary from retrieved chunks
- Inspect retrieved source chunks and metadata
- Review simple analytics and feedback signals

## Stack

- Streamlit
- LangChain
- ChromaDB
- HuggingFace embeddings
- Groq
- PyMuPDF4LLM

## Python version

Use **Python 3.11 or 3.12**.

Do **not** use Python 3.14 because Chroma and the Pydantic v1 compatibility layer can fail there.

## Run locally

```bash
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# add your GROQ_API_KEY to .env
streamlit run app.py
```

## Environment

Create `.env` from `.env.example` and set:

```env
GROQ_API_KEY=your_real_groq_api_key
```

## Main pages

- Chat
- Documents
- Sources
- Analytics
- Settings

## Notes

- PDF indexing is fully implemented.
- Other file types are surfaced in the UI for future extension.
- The UI styling is adapted from the uploaded `knowledge_studio_ui.html` design reference.
