# Lab 6 — Retrieval-Augmented Generation

A RAG system over six Wikipedia articles about generative AI (Transformer, LLM, attention,
diffusion models, GANs, RAG). Everything is in `rag_lab6.ipynb`.

## Pipeline

load `docs/*.md` → structure-aware chunking (900 chars, 150 overlap) → embeddings
(`all-MiniLM-L6-v2`, sentence-transformers) → ChromaDB (cosine, persisted in `chroma_db/`)
→ top-k retrieval → Claude generates a cited answer from the retrieved passages only.

## Run

```bash
# inside the project (PyCharm's .venv is fine)
pip install -r requirements.txt
copy .env.example .env      # then paste your ANTHROPIC_API_KEY into .env
jupyter notebook rag_lab6.ipynb
```

Run the cells top to bottom. The first run downloads the embedding model (~80 MB) and builds
the index; later runs reuse `chroma_db/`.

## Files

| Path | Purpose |
|---|---|
| `rag_lab6.ipynb` | the lab: loading, chunking, embedding, vector store, retrieval, generation, evaluation |
| `docs/` | the corpus (cleaned Wikipedia articles, CC BY-SA 4.0, source URL in the first line of each file) |
| `requirements.txt` | Python dependencies |
| `.env.example` | template for the API key (`.env` is git-ignored) |
| `chroma_db/` | created on first run, the persisted vector index |
