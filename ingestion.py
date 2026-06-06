"""
ingestion.py  –  PDF → chunks → FAISS index + PDF image extraction
Run once (or whenever you add new PDFs):  python ingestion.py
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
    IMAGES_FOLDER,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

BATCH_SIZE = 512


# ─────────────────────────────────────────────────────────────────────────────
# Image extraction (Level 3 — local only, skipped if pymupdf not installed)
# ─────────────────────────────────────────────────────────────────────────────
def extract_images():
    try:
        import fitz  # pymupdf
    except ImportError:
        print("   ⚠  pymupdf not found — skipping image extraction.")
        return

    pdf_files = list(Path(DOCUMENTS_FOLDER).glob("*.pdf"))
    if not pdf_files:
        return

    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    total_saved = 0
    MIN_SIZE = 10_000   # skip tiny decorative images (< 10 KB)

    print(f"\n🖼️   Extracting figures from {len(pdf_files)} PDF(s)…")
    for pdf_path in pdf_files:
        doc   = fitz.open(str(pdf_path))
        stem  = pdf_path.stem
        for page_num, page in enumerate(doc, start=1):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref    = img[0]
                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                if len(img_bytes) < MIN_SIZE:
                    continue
                ext  = base_img["ext"]
                dest = Path(IMAGES_FOLDER) / f"{stem}_p{page_num:04d}_{img_index}.{ext}"
                dest.write_bytes(img_bytes)
                total_saved += 1
        doc.close()

    print(f"    → {total_saved} figure(s) saved to '{IMAGES_FOLDER}/'")


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion
# ─────────────────────────────────────────────────────────────────────────────
def run_ingestion() -> bool:
    # 1 ── Verify documents folder ──────────────────────────────────────────
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
    print("\n⏳  Loading PDFs…")
    loader    = PyPDFDirectoryLoader(DOCUMENTS_FOLDER)
    documents = loader.load()
    print(f"    → {len(documents)} page(s) loaded.")

    if not documents:
        print("❌  No content extracted.")
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
    print(f"    → {len(chunks)} chunk(s)  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # 4 ── Embedding model (Level 1: BAAI/bge-small-en-v1.5) ───────────────
    print(f"\n🧠  Loading embedding model '{EMBEDDING_MODEL}'…")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 5 ── Build FAISS index in batches ────────────────────────────────────
    print(f"\n💾  Building FAISS index (batches of {BATCH_SIZE})…")
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

    print()
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"\n✅  FAISS index saved to '{FAISS_INDEX_PATH}/'")

    # 6 ── Extract images (Level 3 — optional) ─────────────────────────────
    extract_images()

    print(f"\n   Run:  streamlit run app.py")
    return True


if __name__ == "__main__":
    sys.exit(0 if run_ingestion() else 1)
