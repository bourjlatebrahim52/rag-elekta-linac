import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCUMENTS_FOLDER = "documents"
FAISS_INDEX_PATH = "faiss_index"

# ── Embedding ─────────────────────────────────────────────────────────────────
# Free, runs locally, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chunking (tuned for large technical PDFs) ─────────────────────────────────
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",   # best accuracy for technical docs
    "llama-3.1-8b-instant",      # fastest, lighter
    "mixtral-8x7b-32768",        # long-context window
    "gemma2-9b-it",
]
DEFAULT_MODEL       = AVAILABLE_MODELS[0]
DEFAULT_TEMPERATURE = 0.2
