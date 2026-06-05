"""
ingestion.py  –  Load PDFs → chunk → embed → save FAISS index
Run once before launching the app: python ingestion.py
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    DOCUMENTS_FOLDER,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

BATCH_SIZE = 512  # chunks per FAISS batch (keeps RAM manageable for 100 MB PDFs)


def run_ingestion() -> bool:
    # 1 ── Check documents folder ───────────────────────────────────────────
    if not os.path.exists(DOCUMENTS_FOLDER):
        os.makedirs(DOCUMENTS_FOLDER)
        print(f"📁  Created '{DOCUMENTS_FOLDER}/' — place your PDFs there and re-run.")
        return False

    pdf_files = sorted(Path(DOCUMENTS_FOLDER).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️   No PDFs found in '{DOCUMENTS_FOLDER}/'.")
        return False

    print(f"📄  Found {len(pdf_files)} PDF(s):")
    for p in pdf_files:
        size_mb = p.stat().st_size / 1_048_576
        print(f"    • {p.name}  ({size_mb:.1f} MB)")

    # 2 ── Load pages ───────────────────────────────────────────────────────
    print("\n⏳  Loading PDFs (large files may take a few minutes)…")
    loader = PyPDFDirectoryLoader(DOCUMENTS_FOLDER)
    documents = loader.load()
    print(f"    → {len(documents)} page(s) loaded.")

    if not documents:
        print("❌  No content extracted from PDFs.")
        return False

    # 3 ── Chunk ────────────────────────────────────────────────────────────
    print("\n✂️   Splitting into chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"    → {len(chunks)} chunk(s) created  "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # 4 ── Embedding model ──────────────────────────────────────────────────
    print(f"\n🧠  Loading embedding model '{EMBEDDING_MODEL}'…")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 5 ── Build FAISS index in batches ────────────────────────────────────
    print(f"\n💾  Building FAISS index (batch size = {BATCH_SIZE})…")
    vectorstore = None
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        done = min(start + BATCH_SIZE, total)
        pct  = int(done / total * 100)
        bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"    [{bar}] {pct:3d}%  ({done}/{total})", end="\r")

    print()  # newline after progress bar
    vectorstore.save_local(FAISS_INDEX_PATH)

    print(f"\n✅  Ingestion complete! Index saved to '{FAISS_INDEX_PATH}/'")
    print("    Run:  streamlit run app.py")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_ingestion() else 1)
