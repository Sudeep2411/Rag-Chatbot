[file name]: README.md
[file content begin]
# RAG Chatbot (Hugging Face Only) — VS Code + Streamlit + Chroma

This is a **100% Python** Retrieval-Augmented Generation (RAG) chatbot that uses only **Hugging Face** tooling.  
You can run locally in **VS Code** and deploy to **Hugging Face Spaces**.  
It includes:
- Document ingestion (PDF/TXT/URLs) → chunking → Hugging Face **sentence-transformers** embeddings → **Chroma** vectorstore (local).
- Prompt engineering with a clean template that injects retrieved context.
- LLM generation via `transformers` pipelines (text-generation or text2text-generation).
- Streamlit frontend with upload + sources + citations + optional memory.
- **No fine-tuning included** — this project uses pretrained models only.

## Quickstart

1. Create & activate venv
2. pip install -r requirements.txt
3. Add PDFs/TXTs to app/data or URLs to app/data/urls.txt
4. python app/src/ingestion/ingest.py
5. streamlit run app/app.py
[file content end]