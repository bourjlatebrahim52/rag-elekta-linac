# RAG Chatbot — Elekta Linac Maintenance Manuals

A Retrieval-Augmented Generation (RAG) application that lets you ask questions
about your Elekta Linac technical PDFs.  Answers are grounded exclusively in
your documents — no hallucination.

**Stack:** Streamlit · LangChain · FAISS · ChatGroq · HuggingFace Embeddings

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure your Groq API key

Get a **free** key at <https://console.groq.com>, then:

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # macOS / Linux
```

Edit `.env` and paste your key:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Add your PDFs

Place all PDF files inside the `documents/` folder.

### 4. Index the PDFs (run once, or whenever you add new PDFs)

```bash
python ingestion.py
```

This will:
- Load every PDF in `documents/`
- Split them into overlapping chunks
- Embed with `all-MiniLM-L6-v2` (runs locally, free)
- Save a FAISS vector index to `faiss_index/`

> For 6 PDFs totalling ~350 MB expect ~5–15 minutes on first run.

### 5. Launch the app

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## Project structure

```
pfe_rag_groq/
├── app.py            ← Streamlit chatbot (ChatGroq + FAISS retrieval)
├── ingestion.py      ← PDF → chunks → embeddings → FAISS index
├── config.py         ← All tunable constants in one place
├── requirements.txt
├── .env.example      ← Copy to .env and add your Groq key
├── .gitignore
├── documents/        ← Drop your PDFs here (gitignored)
└── faiss_index/      ← Auto-generated after ingestion (gitignored)
```

---

## Configuration (config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 1200 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local HuggingFace embedding model |
| `TOP_K` | 5 | Chunks retrieved per query |
| `DEFAULT_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `DEFAULT_TEMPERATURE` | 0.2 | LLM temperature (0 = deterministic) |

All sidebar sliders in the UI override these at runtime without restarting.

---

## Available Groq models

| Model | Best for |
|-------|----------|
| `llama-3.3-70b-versatile` | Best accuracy, technical documents |
| `llama-3.1-8b-instant` | Fastest responses |
| `mixtral-8x7b-32768` | Long context (32k tokens) |
| `gemma2-9b-it` | Balanced speed / quality |

---

## Tips for large PDFs (100 MB+)

- Ingestion is a one-time step; it takes longer the first time.
- After the FAISS index is built, the app loads in seconds.
- If you run out of RAM during ingestion, reduce `BATCH_SIZE` in `ingestion.py`
  (default: 512 chunks per batch).
- Re-run `python ingestion.py` whenever you add or replace PDF files.
