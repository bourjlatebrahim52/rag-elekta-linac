"""
app.py — Elekta Linac Maintenance Documentation Query System
Enterprise clinical workstation interface — Internal Use Only

Backend : UNCHANGED (hybrid BM25+FAISS, CrossEncoder, multi-query, decomposition)
UI      : Enterprise clinical redesign — no consumer chatbot aesthetics
"""

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import (
    FAISS_INDEX_PATH, IMAGES_FOLDER, EMBEDDING_MODEL,
    TOP_K_RETRIEVAL, TOP_K_RERANKED, BM25_WEIGHT, FAISS_WEIGHT,
    RERANKER_MODEL, MULTI_QUERY_VARIANTS, AVAILABLE_MODELS,
    DEFAULT_MODEL, DEFAULT_TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP,
)

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND — ALL FUNCTIONS UNCHANGED
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_vectorstore() -> FAISS:
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True)


@st.cache_resource(show_spinner=False)
def load_reranker() -> HuggingFaceCrossEncoder:
    return HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY", ""),
        model_name=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        streaming=True,
    )


@st.cache_resource(show_spinner=False)
def build_bm25(_vectorstore: FAISS) -> BM25Retriever:
    all_docs = list(_vectorstore.docstore._dict.values())
    return BM25Retriever.from_documents(all_docs, k=TOP_K_RETRIEVAL)


def hybrid_retrieve(query: str) -> list:
    faiss_docs = vectorstore.similarity_search(query, k=TOP_K_RETRIEVAL)
    bm25_docs  = bm25.invoke(query)
    seen, merged = set(), []
    for doc in faiss_docs + bm25_docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", ""), doc.page_content[:80])
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged[:TOP_K_RETRIEVAL]


def is_complex_question(question: str) -> bool:
    keywords = ["and","including","after","before","also","then","et","ainsi",
                "après","avant","également","puis","full","complete","entire","overall","complet"]
    return sum(1 for k in keywords if k in question.lower()) >= 2


def decompose_question(question: str) -> list[str]:
    msg = HumanMessage(content=(
        "You are an expert at analyzing technical maintenance questions. "
        "Break this complex question into 3-4 simple, independent sub-questions. "
        "Each sub-question must be answerable on its own from a maintenance manual. "
        "Return ONLY the sub-questions, one per line, no numbering, no explanation.\n\n"
        f"Complex question: {question}"
    ))
    resp   = llm.invoke([msg])
    sub_qs = [q.strip() for q in resp.content.strip().split("\n") if q.strip()]
    return [question] + sub_qs[:4]


def generate_query_variants(question: str) -> list[str]:
    msg = HumanMessage(content=(
        f"Generate {MULTI_QUERY_VARIANTS} alternative phrasings of this question "
        f"using different technical vocabulary for Elekta Linac maintenance manuals. "
        f"Return ONLY the questions, one per line, no numbering.\n\n"
        f"Question: {question}"
    ))
    resp     = llm.invoke([msg])
    variants = [q.strip() for q in resp.content.strip().split("\n") if q.strip()]
    return [question] + variants[:MULTI_QUERY_VARIANTS]


def expand_queries(question: str) -> list[str]:
    if is_complex_question(question):
        return decompose_question(question)
    return generate_query_variants(question)


def rerank_docs(query: str, docs: list, top_n: int = TOP_K_RERANKED):
    if not docs:
        return []
    pairs  = [(query, d.page_content) for d in docs]
    scores = reranker.score(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [(doc, score) for score, doc in ranked[:top_n] if score > -3]


SYSTEM_PROMPT = (
    "You are a technical assistant for Elekta Linac corrective maintenance documentation.\n\n"
    "STRICT RULES:\n"
    "1. Answer ONLY from the excerpts provided. Never use training knowledge.\n"
    "2. For complex multi-part questions: answer each part separately using the available context. "
    "   Clearly label each part. If one part is not in the context, say so for that part only.\n"
    "3. If the context contains NO relevant information at all, respond ONLY with: "
    "   'The provided documents do not contain the information needed to answer this question.'\n"
    "4. FORBIDDEN: 'deduce', 'infer', 'typically', 'usually', 'could be', 'might be'. "
    "   Using these means applying rule 3.\n"
    "5. Structure: Direct answer → Numbered steps → References (doc name + page only).\n"
    "6. Be concise. No preamble. No filler.\n"
    "7. Detect the language of the question and reply in the same language."
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Elekta Linac — Maintenance Query System",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

_defaults = {
    "messages":         [],
    "last_sources":     [],
    "query_timestamps": [],   # HH:MM:SS per query
    "source_counts":    [],   # int per query
    "system_ready":     False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — ENTERPRISE CLINICAL WORKSTATION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── GLOBAL ────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #060F1E !important; }

/* ── SIDEBAR ───────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #040C18 !important;
    border-right: 1px solid #142035 !important;
    min-width: 255px !important;
    max-width: 255px !important;
}
section[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
button[data-testid="stSidebarNavToggle"],
button[data-testid="stSidebarCollapsedControl"] {
    background: #040C18 !important;
    border-color: #142035 !important;
    color: #3A5870 !important;
}

/* ── SIDEBAR: HEADER ───────────────────────────────────────────────── */
.sb-header {
    padding: 16px 18px 13px;
    border-bottom: 1px solid #142035;
}
.sb-brand {
    font-size: 0.6rem;
    font-weight: 700;
    color: #0078D4;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 4px;
}
.sb-product {
    font-size: 0.75rem;
    font-weight: 600;
    color: #B0C8DC;
    line-height: 1.25;
}
.sb-build {
    font-size: 0.58rem;
    color: #1E3550;
    font-family: "Consolas", monospace;
    margin-top: 5px;
}

/* ── SIDEBAR: SECTIONS ─────────────────────────────────────────────── */
.sb-section {
    padding: 10px 18px;
    border-bottom: 1px solid #0B1A28;
}
.sb-section-title {
    font-size: 0.55rem;
    font-weight: 700;
    color: #1E3550;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-bottom: 9px;
}

/* Key-value rows */
.sb-kv {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 2px 0;
    gap: 6px;
}
.sb-k {
    font-size: 0.65rem;
    color: #3A5870;
    white-space: nowrap;
    flex-shrink: 0;
}
.sb-v {
    font-size: 0.62rem;
    color: #8AAEC8;
    font-family: "Consolas", monospace;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Status rows */
.sb-status-row {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 3px 0;
}
.sb-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-ok   { background: #00B87A; }
.dot-warn { background: #C08800; }
.dot-err  { background: #B02030; }
.sb-dot-label { font-size: 0.65rem; color: #3A5870; flex: 1; }
.sb-dot-val   { font-size: 0.62rem; color: #607888; font-family: "Consolas", monospace; }

/* Document list */
.sb-doc {
    display: flex;
    gap: 7px;
    padding: 4px 0;
    border-bottom: 1px solid #0A1825;
    align-items: flex-start;
}
.sb-doc:last-child { border-bottom: none; }
.sb-doc-idx  { font-size: 0.6rem; color: #1E3550; font-family: "Consolas", monospace;
               min-width: 18px; flex-shrink: 0; padding-top: 1px; }
.sb-doc-name { font-size: 0.65rem; color: #5A7890; line-height: 1.35; }

/* ── SIDEBAR: BUTTON ───────────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    color: #3A5870 !important;
    border: 1px solid #142035 !important;
    border-radius: 2px !important;
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 5px 14px !important;
    transition: all 0.12s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #0C1C30 !important;
    color: #8AAEC8 !important;
    border-color: #2A4060 !important;
}

/* ── WORKLOG HEADER STRIP ──────────────────────────────────────────── */
.wl-strip {
    background: #040C18;
    border-bottom: 1px solid #142035;
    padding: 7px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.wl-title {
    font-size: 0.58rem;
    font-weight: 700;
    color: #0078D4;
    text-transform: uppercase;
    letter-spacing: 2.5px;
}
.wl-meta {
    font-size: 0.58rem;
    color: #1E3550;
    font-family: "Consolas", monospace;
}

/* ── QUERY ENTRY ───────────────────────────────────────────────────── */
.qe-q-block {
    padding: 14px 16px 10px;
    border-bottom: 1px solid #0B1A28;
}
.qe-q-head {
    font-size: 0.55rem;
    font-weight: 700;
    color: #1E3550;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: "Consolas", monospace;
    margin-bottom: 7px;
}
.qe-q-text {
    font-size: 0.84rem;
    color: #A0BCCC;
    padding: 5px 0 2px 12px;
    border-left: 2px solid #0078D4;
    line-height: 1.5;
    font-weight: 400;
}
.qe-r-head {
    padding: 7px 16px 4px;
    font-size: 0.55rem;
    font-weight: 700;
    color: #1E3550;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: "Consolas", monospace;
}
.qe-sep {
    height: 1px;
    background: #0B1A28;
    margin: 10px 16px 0;
}

/* ── RESPONSE TEXT (native markdown) ──────────────────────────────── */
.stMarkdown p {
    font-size: 0.84rem !important;
    color: #8AAEC8 !important;
    line-height: 1.75 !important;
    margin-bottom: 6px !important;
}
.stMarkdown li {
    font-size: 0.82rem !important;
    color: #8AAEC8 !important;
    line-height: 1.7 !important;
}
.stMarkdown strong { color: #B0C8DC !important; font-weight: 600 !important; }
.stMarkdown em { color: #7098B0 !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: #607888 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin: 10px 0 5px !important;
    border-bottom: 1px solid #0B1A28 !important;
    padding-bottom: 3px !important;
}
.stMarkdown code {
    font-size: 0.75rem !important;
    color: #4A90C8 !important;
    background: #071018 !important;
    padding: 1px 6px !important;
    border-radius: 2px !important;
    font-family: "Consolas", monospace !important;
}
.stMarkdown ol, .stMarkdown ul {
    padding-left: 1.2rem !important;
    margin: 4px 0 !important;
}

/* ── PROCESSING STATUS ─────────────────────────────────────────────── */
.proc-bar {
    font-size: 0.7rem;
    color: #3A5870;
    font-family: "Consolas", monospace;
    padding: 6px 12px;
    background: #071018;
    border-left: 2px solid #0078D4;
    margin: 6px 16px;
    line-height: 1.4;
}
.proc-tag {
    color: #0078D4;
    font-weight: 700;
    min-width: 96px;
    display: inline-block;
}

/* ── EMPTY STATE ───────────────────────────────────────────────────── */
.wl-empty {
    padding: 32px 16px;
    border-bottom: 1px solid #0B1A28;
    text-align: center;
}
.wl-empty-txt {
    font-size: 0.6rem;
    color: #1E3550;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: "Consolas", monospace;
}

/* ── REFERENCE PANEL ───────────────────────────────────────────────── */
.ref-strip {
    background: #040C18;
    border-bottom: 1px solid #142035;
    padding: 7px 12px;
    font-size: 0.58rem;
    font-weight: 700;
    color: #0078D4;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-family: "Consolas", monospace;
}

/* Reference table */
.ref-tbl { width: 100%; border-collapse: collapse; }
.ref-tbl th {
    font-size: 0.55rem;
    font-weight: 700;
    color: #1E3550;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 6px 10px 5px;
    border-bottom: 1px solid #142035;
    text-align: left;
    font-family: "Consolas", monospace;
    background: #040C18;
    white-space: nowrap;
}
.ref-tbl td {
    font-size: 0.67rem;
    color: #3A5870;
    padding: 5px 10px;
    border-bottom: 1px solid #091520;
    vertical-align: top;
    line-height: 1.35;
}
.ref-tbl td.c-n  { color: #1E3550; font-family: "Consolas", monospace; width: 22px; }
.ref-tbl td.c-d  { color: #6088A0; }
.ref-tbl td.c-p  { color: #3A5870; font-family: "Consolas", monospace; text-align:center; }
.ref-tbl td.c-s  { color: #2A4458; font-family: "Consolas", monospace; text-align:right; }
.ref-tbl tr:hover td { background: #0A1828 !important; color: #8AAEC8 !important; }

/* No references state */
.ref-empty {
    padding: 24px 12px;
    text-align: center;
}
.ref-empty-txt {
    font-size: 0.58rem;
    color: #142030;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: "Consolas", monospace;
    line-height: 2;
}

/* Excerpt */
.excerpt-blk {
    font-size: 0.65rem;
    color: #3A5870;
    font-family: "Consolas", monospace;
    background: #040C18;
    border-left: 2px solid #142035;
    padding: 8px 10px;
    line-height: 1.55;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── EXPANDERS ─────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #071018 !important;
    border: 1px solid #0B1A28 !important;
    border-radius: 2px !important;
    margin-top: 3px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.6rem !important;
    color: #2A4060 !important;
    font-family: "Consolas", monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* ── CHAT INPUT ────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: #071018 !important;
    border: 1px solid #142035 !important;
    border-radius: 2px !important;
    box-shadow: none !important;
    margin: 4px 8px 6px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #0060B0 !important;
    box-shadow: 0 0 0 1px rgba(0,96,176,0.3) !important;
}
[data-testid="stChatInput"] textarea {
    color: #A0BCCC !important;
    background: transparent !important;
    font-size: 0.84rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #1E3550 !important; }

/* ── HIDE CHAT AVATARS ─────────────────────────────────────────────── */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] { display: none !important; }
[data-testid="stChatMessage"] > div:first-child {
    width: 0 !important; min-width: 0 !important; padding: 0 !important;
}
[data-testid="stChatMessage"],
[data-testid="stChatMessageContent"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── SCROLLBAR ─────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #142035; border-radius: 2px; }

/* Global text override */
h1, h2, h3, h4, p, span, div, label { color: #8AAEC8; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

api_key  = os.getenv("GROQ_API_KEY", "")
index_ok = os.path.exists(FAISS_INDEX_PATH)

if api_key and index_ok:
    vectorstore = load_vectorstore()
    reranker    = load_reranker()
    llm         = get_llm()
    bm25        = build_bm25(vectorstore)
    st.session_state.system_ready = True

    all_docs    = list(vectorstore.docstore._dict.values())
    doc_sources = sorted(set(os.path.basename(d.metadata.get("source","")) for d in all_docs))
    n_chunks    = len(all_docs)
    idx_kb      = sum(f.stat().st_size for f in os.scandir(FAISS_INDEX_PATH)) // 1024
else:
    doc_sources = []
    n_chunks = idx_kb = 0

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — SYSTEM INFORMATION PANEL
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Product header ────────────────────────────────────────────────────────
    build_date = datetime.now().strftime("%Y-%m-%d")
    st.markdown(f"""
    <div class="sb-header">
        <div class="sb-brand">ELEKTA</div>
        <div class="sb-product">Linac Maintenance<br>Documentation Query System</div>
        <div class="sb-build">BUILD 3.0 &nbsp;·&nbsp; {build_date}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── System status ─────────────────────────────────────────────────────────
    ok = st.session_state.system_ready
    st.markdown(f"""
    <div class="sb-section">
        <div class="sb-section-title">System Status</div>
        <div class="sb-status-row">
            <div class="sb-dot {'dot-ok' if api_key else 'dot-err'}"></div>
            <div class="sb-dot-label">LLM Engine</div>
            <div class="sb-dot-val">{"ONLINE" if api_key else "NO KEY"}</div>
        </div>
        <div class="sb-status-row">
            <div class="sb-dot {'dot-ok' if index_ok else 'dot-err'}"></div>
            <div class="sb-dot-label">Vector Index</div>
            <div class="sb-dot-val">{"READY" if index_ok else "NOT FOUND"}</div>
        </div>
        <div class="sb-status-row">
            <div class="sb-dot {'dot-ok' if ok else 'dot-warn'}"></div>
            <div class="sb-dot-label">Hybrid Search</div>
            <div class="sb-dot-val">{"BM25 + FAISS" if ok else "STANDBY"}</div>
        </div>
        <div class="sb-status-row">
            <div class="sb-dot {'dot-ok' if ok else 'dot-warn'}"></div>
            <div class="sb-dot-label">Reranker</div>
            <div class="sb-dot-val">{"CROSSENCODER" if ok else "STANDBY"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Index parameters ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sb-section">
        <div class="sb-section-title">Index Parameters</div>
        <div class="sb-kv">
            <span class="sb-k">Embedding model</span>
            <span class="sb-v">{EMBEDDING_MODEL.split("/")[-1]}</span>
        </div>
        <div class="sb-kv">
            <span class="sb-k">Total chunks</span>
            <span class="sb-v">{n_chunks:,}</span>
        </div>
        <div class="sb-kv">
            <span class="sb-k">Chunk / overlap</span>
            <span class="sb-v">{CHUNK_SIZE} / {CHUNK_OVERLAP}</span>
        </div>
        <div class="sb-kv">
            <span class="sb-k">Index size</span>
            <span class="sb-v">{idx_kb} KB</span>
        </div>
        <div class="sb-kv">
            <span class="sb-k">Retrieval k</span>
            <span class="sb-v">{TOP_K_RETRIEVAL} → rerank {TOP_K_RERANKED}</span>
        </div>
        <div class="sb-kv">
            <span class="sb-k">BM25 / FAISS</span>
            <span class="sb-v">{BM25_WEIGHT} / {FAISS_WEIGHT}</span>
        </div>
        <div class="sb-kv">
            <span class="sb-k">LLM</span>
            <span class="sb-v">{DEFAULT_MODEL}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Document registry ─────────────────────────────────────────────────────
    if doc_sources:
        rows_html = ""
        for i, name in enumerate(doc_sources, 1):
            short = (name.replace("Linac - Corrective Maintenance - ", "")
                        .replace("Linac - ", "").replace(".pdf", ""))
            rows_html += (f'<div class="sb-doc">'
                          f'<span class="sb-doc-idx">{i:02d}</span>'
                          f'<span class="sb-doc-name">{short}</span>'
                          f'</div>')
        st.markdown(f"""
        <div class="sb-section">
            <div class="sb-section-title">Document Registry — {len(doc_sources)} file(s)</div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

    # ── Session control ───────────────────────────────────────────────────────
    st.markdown('<div style="padding:10px 18px;">', unsafe_allow_html=True)
    if st.button("NEW SESSION"):
        st.session_state.messages         = []
        st.session_state.last_sources     = []
        st.session_state.query_timestamps = []
        st.session_state.source_counts    = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — ERROR GUARDS
# ═══════════════════════════════════════════════════════════════════════════════

if not api_key:
    st.markdown("""
    <div style="padding:28px 20px;font-family:Consolas,monospace;">
        <div style="font-size:.6rem;color:#B02030;text-transform:uppercase;letter-spacing:2px;">
            ERR-001 &nbsp;·&nbsp; GROQ_API_KEY not found in environment
        </div>
        <div style="font-size:.7rem;color:#3A5870;margin-top:8px;">
            Add GROQ_API_KEY to .env or Streamlit Secrets and restart.
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

if not index_ok:
    st.markdown(f"""
    <div style="padding:28px 20px;font-family:Consolas,monospace;">
        <div style="font-size:.6rem;color:#C08000;text-transform:uppercase;letter-spacing:2px;">
            ERR-002 &nbsp;·&nbsp; FAISS index not found at '{FAISS_INDEX_PATH}/'
        </div>
        <div style="font-size:.7rem;color:#3A5870;margin-top:8px;">
            Run: python ingestion.py
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — QUERY LOG + REFERENCE PANEL
# ═══════════════════════════════════════════════════════════════════════════════

log_col, ref_col = st.columns([63, 37], gap="small")

# ── REFERENCE PANEL ───────────────────────────────────────────────────────────
with ref_col:
    n_src = len(st.session_state.last_sources)
    st.markdown(f"""
    <div class="ref-strip">
        RETRIEVED REFERENCES &nbsp;·&nbsp; {n_src} RECORD(S)
    </div>
    """, unsafe_allow_html=True)

    ref_placeholder = st.empty()

    def render_ref_table(src_list: list):
        """Render the reference table — called from ref_col and updated from log_col."""
        if not src_list:
            ref_placeholder.markdown("""
            <div class="ref-empty">
                <div class="ref-empty-txt">
                    NO REFERENCES LOADED<br>
                    <span style="color:#0E1E2E;">
                    Submit a query to retrieve documentation records</span>
                </div>
            </div>""", unsafe_allow_html=True)
            return

        rows = ""
        for i, (doc, score) in enumerate(src_list, 1):
            name  = os.path.basename(doc.metadata.get("source", "unknown"))
            page  = doc.metadata.get("page", "?")
            short = (name.replace("Linac - Corrective Maintenance - ", "")
                        .replace("Linac - ", "").replace(".pdf", ""))
            rows += (f"<tr>"
                     f'<td class="c-n">{i:02d}</td>'
                     f'<td class="c-d">{short}</td>'
                     f'<td class="c-p">{page}</td>'
                     f'<td class="c-s">{score:.2f}</td>'
                     f"</tr>")

        ref_placeholder.markdown(f"""
        <table class="ref-tbl">
          <thead>
            <tr>
              <th>#</th><th>Document</th><th>Pg</th><th>Score</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

    render_ref_table(st.session_state.last_sources)

    # Expandable excerpts
    if st.session_state.last_sources:
        for i, (doc, _score) in enumerate(st.session_state.last_sources, 1):
            name  = os.path.basename(doc.metadata.get("source", ""))
            page  = doc.metadata.get("page", "?")
            short = (name.replace("Linac - Corrective Maintenance - ", "")
                        .replace("Linac - ", "").replace(".pdf", ""))
            with st.expander(f"EXCERPT {i:02d} — {short} · PG {page}"):
                st.markdown(
                    f'<div class="excerpt-blk">{doc.page_content[:480].strip()}</div>',
                    unsafe_allow_html=True,
                )
                if os.path.exists(IMAGES_FOLDER):
                    stem    = Path(doc.metadata.get("source", "")).stem
                    pattern = f"{stem}_p{int(page) + 1:04d}_"
                    imgs    = sorted(Path(IMAGES_FOLDER).glob(f"{pattern}*"))
                    for img_path in imgs[:2]:
                        st.image(str(img_path), use_container_width=True)

# ── QUERY LOG ─────────────────────────────────────────────────────────────────
with log_col:

    # Header strip
    n_entries    = len(st.session_state.query_timestamps)
    session_date = datetime.now().strftime("%Y-%m-%d")
    st.markdown(f"""
    <div class="wl-strip">
        <span class="wl-title">QUERY LOG</span>
        <span class="wl-meta">
            {session_date} &nbsp;·&nbsp;
            {n_entries} ENTR{"IES" if n_entries != 1 else "Y"}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Empty state
    if not st.session_state.messages:
        st.markdown("""
        <div class="wl-empty">
            <div class="wl-empty-txt">— NO QUERIES IN CURRENT SESSION —</div>
        </div>
        """, unsafe_allow_html=True)

    # Render message history as numbered log entries
    msgs = st.session_state.messages
    idx  = 0
    while idx < len(msgs):
        if isinstance(msgs[idx], HumanMessage):
            q_num = idx // 2
            ts    = (st.session_state.query_timestamps[q_num]
                     if q_num < len(st.session_state.query_timestamps) else "")
            n_r   = (st.session_state.source_counts[q_num]
                     if q_num < len(st.session_state.source_counts) else 0)

            # Query block
            st.markdown(f"""
            <div class="qe-q-block">
                <div class="qe-q-head">QUERY &nbsp;#{q_num + 1:02d} &nbsp;·&nbsp; {ts}</div>
                <div class="qe-q-text">{msgs[idx].content}</div>
            </div>
            """, unsafe_allow_html=True)

            # Response block
            if idx + 1 < len(msgs) and isinstance(msgs[idx + 1], AIMessage):
                st.markdown(f"""
                <div class="qe-r-head">
                    RESPONSE &nbsp;·&nbsp; {n_r} reference(s) retrieved
                </div>
                """, unsafe_allow_html=True)
                # Native st.markdown — renders bold, lists, headers correctly
                with st.container():
                    st.markdown(
                        "<div style='padding:0 16px 4px;'></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(msgs[idx + 1].content)
                st.markdown('<div class="qe-sep"></div>', unsafe_allow_html=True)
                idx += 2
            else:
                idx += 1
        else:
            idx += 1

    # ── Chat input ─────────────────────────────────────────────────────────
    user_input = st.chat_input("Enter maintenance query…")

    if user_input:
        ts_now = datetime.now().strftime("%H:%M:%S")
        q_num  = len(st.session_state.query_timestamps)

        # Show query immediately
        st.markdown(f"""
        <div class="qe-q-block">
            <div class="qe-q-head">QUERY &nbsp;#{q_num + 1:02d} &nbsp;·&nbsp; {ts_now}</div>
            <div class="qe-q-text">{user_input}</div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.messages.append(HumanMessage(content=user_input))
        st.session_state.query_timestamps.append(ts_now)

        proc_box = st.empty()

        # Step 1 — Query analysis
        proc_box.markdown(f"""
        <div class="proc-bar">
            <span class="proc-tag">ANALYZING</span>
            Classifying query · generating retrieval variants…
        </div>""", unsafe_allow_html=True)
        queries = expand_queries(user_input)

        # Step 2 — Retrieval
        proc_box.markdown(f"""
        <div class="proc-bar">
            <span class="proc-tag">RETRIEVING</span>
            Hybrid BM25 + FAISS search · {n_chunks:,} chunks · {len(queries)} variant(s)…
        </div>""", unsafe_allow_html=True)
        seen_ids, candidate_docs = set(), []
        for q in queries:
            for doc in hybrid_retrieve(q):
                doc_id = (doc.metadata.get("source", ""),
                          doc.metadata.get("page", ""),
                          doc.page_content[:100])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    candidate_docs.append(doc)

        # Step 3 — Reranking
        proc_box.markdown(f"""
        <div class="proc-bar">
            <span class="proc-tag">RERANKING</span>
            CrossEncoder scoring {len(candidate_docs)} candidates · selecting top {TOP_K_RERANKED}…
        </div>""", unsafe_allow_html=True)
        reranked = rerank_docs(user_input, candidate_docs, top_n=TOP_K_RERANKED)
        top_docs = [doc for doc, _ in reranked]
        n_retrieved = len(reranked)

        # Update reference panel before streaming
        st.session_state.last_sources = reranked
        st.session_state.source_counts.append(n_retrieved)
        render_ref_table(reranked)

        # Build context message
        if top_docs:
            ctx_parts = [
                f"[Document: {os.path.basename(d.metadata.get('source','unknown'))} | "
                f"Page: {d.metadata.get('page','?')}]\n{d.page_content}"
                for d in top_docs
            ]
            context_msg = SystemMessage(
                content="Use ONLY the following excerpts to answer. "
                        "Do not use any knowledge outside this context.\n\n"
                        + "\n\n---\n\n".join(ctx_parts)
            )
        else:
            context_msg = SystemMessage(
                content="No relevant excerpts found. Reply: "
                        "'The provided documents do not contain the information "
                        "needed to answer this question.'"
            )

        messages_to_send = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + st.session_state.messages
            + [context_msg]
        )

        # Step 4 — Generation
        proc_box.markdown(f"""
        <div class="proc-bar">
            <span class="proc-tag">GENERATING</span>
            Composing response from {n_retrieved} retrieved reference(s)…
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="qe-r-head">
            RESPONSE &nbsp;·&nbsp; {n_retrieved} reference(s) retrieved
        </div>
        <div style="padding:0 16px 4px;"></div>
        """, unsafe_allow_html=True)

        stream_box  = st.empty()
        full_answer = ""
        for chunk in llm.stream(messages_to_send):
            full_answer += chunk.content
            stream_box.markdown(full_answer + " ▌")
        stream_box.markdown(full_answer)

        proc_box.empty()
        st.markdown('<div class="qe-sep"></div>', unsafe_allow_html=True)

        st.session_state.messages.append(AIMessage(content=full_answer))
