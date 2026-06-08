"""
app.py — Radiotherapy RAG Assistant
Premium Healthcare UI — Elekta Linac Maintenance Platform

Backend: UNCHANGED (hybrid BM25+FAISS, CrossEncoder reranker, multi-query, question decomposition)
UI:      Fully redesigned — clinical dark/light theme, 3-panel layout, medical-grade aesthetics
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
    FAISS_INDEX_PATH,
    IMAGES_FOLDER,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_RERANKED,
    BM25_WEIGHT,
    FAISS_WEIGHT,
    RERANKER_MODEL,
    MULTI_QUERY_VARIANTS,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
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
        key = (doc.metadata.get("source",""), doc.metadata.get("page",""), doc.page_content[:80])
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
    resp  = llm.invoke([msg])
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
# PAGE CONFIG — must be first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RadAssist — Elekta Linac",
    page_icon="https://www.elekta.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════════

defaults = {
    "theme":          "dark",
    "messages":       [],
    "last_sources":   [],
    "query_count":    0,
    "session_start":  datetime.now().strftime("%H:%M"),
    "system_ready":   False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

T = st.session_state.theme   # shorthand

# ═══════════════════════════════════════════════════════════════════════════════
# THEME CSS
# ═══════════════════════════════════════════════════════════════════════════════

DARK = {
    "bg":           "#060D1A",
    "bg2":          "#0A1628",
    "card":         "#0D1F36",
    "card_hover":   "#112540",
    "accent":       "#0EA5E9",
    "accent_hover": "#38BDF8",
    "accent_dim":   "rgba(14,165,233,0.12)",
    "text":         "#E2E8F0",
    "text2":        "#94A3B8",
    "text3":        "#475569",
    "border":       "rgba(14,165,233,0.14)",
    "border2":      "rgba(255,255,255,0.06)",
    "user_bg":      "linear-gradient(135deg,#0369A1,#0EA5E9)",
    "asst_bg":      "#0D1F36",
    "asst_border":  "1px solid rgba(14,165,233,0.18)",
    "success":      "#10B981",
    "warning":      "#F59E0B",
    "danger":       "#EF4444",
    "sidebar_bg":   "#040B17",
    "input_bg":     "#0D1F36",
    "scrollbar":    "#1E3A5F",
}

LIGHT = {
    "bg":           "#F0F4F8",
    "bg2":          "#FFFFFF",
    "card":         "#FFFFFF",
    "card_hover":   "#F8FAFC",
    "accent":       "#1B4F8A",
    "accent_hover": "#1E40AF",
    "accent_dim":   "rgba(27,79,138,0.08)",
    "text":         "#0F172A",
    "text2":        "#475569",
    "text3":        "#94A3B8",
    "border":       "#E2E8F0",
    "border2":      "#F1F5F9",
    "user_bg":      "linear-gradient(135deg,#1B4F8A,#2563EB)",
    "asst_bg":      "#FFFFFF",
    "asst_border":  "1px solid #E2E8F0",
    "success":      "#059669",
    "warning":      "#D97706",
    "danger":       "#DC2626",
    "sidebar_bg":   "#06101E",
    "input_bg":     "#FFFFFF",
    "scrollbar":    "#CBD5E1",
}

C = DARK if T == "dark" else LIGHT


def inject_css(c: dict):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── GLOBAL RESET ─────────────────────────────────────────── */
    html, body, [class*="css"] {{
        font-family: "Inter", "Segoe UI", -apple-system, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }}
    #MainMenu, footer, header, .stDeployButton {{ display: none !important; }}
    .block-container {{ padding: 1.5rem 1.5rem 0 1.5rem !important; max-width: 100% !important; }}
    .stApp {{ background: {c["bg"]} !important; }}

    /* ── SIDEBAR ──────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {c["sidebar_bg"]} !important;
        border-right: 1px solid rgba(14,165,233,0.12) !important;
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        padding: 0 !important;
    }}
    button[data-testid="stSidebarNavToggle"],
    button[data-testid="stSidebarCollapsedControl"] {{
        background: {c["sidebar_bg"]} !important;
        color: {c["accent"]} !important;
        border-color: {c["border"]} !important;
    }}

    /* ── TYPOGRAPHY ───────────────────────────────────────────── */
    h1, h2, h3, h4, p, span, div, label {{
        color: {c["text"]} !important;
    }}

    /* ── CHAT MESSAGES ────────────────────────────────────────── */
    [data-testid="stChatMessage"] {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin-bottom: 0.5rem !important;
    }}
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {{ display: none !important; }}
    [data-testid="stChatMessage"] > div:first-child {{
        width: 0 !important; min-width: 0 !important; padding: 0 !important;
    }}
    [data-testid="stChatMessageContent"] {{
        background: transparent !important;
    }}

    /* Assistant bubble — styled via container using :has() so markdown still renders */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
        background: {c["asst_bg"]} !important;
        border: {c["asst_border"]} !important;
        border-radius: 4px 16px 16px 16px !important;
        padding: 14px 18px 16px !important;
        margin-bottom: 1.2rem !important;
        max-width: 93% !important;
        animation: fadeInUp 0.25s ease forwards;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08) !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) li,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) strong,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) em {{
        color: {c["text"]} !important;
        font-size: 0.88rem !important;
        line-height: 1.65 !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) code {{
        background: {c["bg"]} !important;
        color: {c["accent"]} !important;
        padding: 1px 6px !important;
        border-radius: 4px !important;
        font-size: 0.82rem !important;
        font-family: "Courier New", monospace !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) h1,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) h2,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) h3 {{
        color: {c["text"]} !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        margin-top: 10px !important;
        margin-bottom: 4px !important;
        border-bottom: 1px solid {c["border"]} !important;
        padding-bottom: 4px !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) ol,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) ul {{
        padding-left: 1.2rem !important;
        margin: 6px 0 !important;
    }}

    /* ── BUTTONS ──────────────────────────────────────────────── */
    .stButton > button {{
        background: {c["accent_dim"]} !important;
        color: {c["accent"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.3px !important;
    }}
    .stButton > button:hover {{
        background: {c["accent"]} !important;
        color: #fff !important;
        border-color: {c["accent"]} !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(14,165,233,0.25) !important;
    }}
    .btn-danger > button {{
        background: rgba(239,68,68,0.1) !important;
        color: {c["danger"]} !important;
        border-color: rgba(239,68,68,0.25) !important;
    }}
    .btn-danger > button:hover {{
        background: {c["danger"]} !important;
        color: #fff !important;
    }}

    /* ── CHAT INPUT ───────────────────────────────────────────── */
    [data-testid="stChatInput"] {{
        background: {c["input_bg"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 20px rgba(0,0,0,0.15) !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: {c["accent"]} !important;
        box-shadow: 0 0 0 3px {c["accent_dim"]} !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: {c["text"]} !important;
        background: transparent !important;
        font-size: 0.9rem !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {c["text3"]} !important;
    }}

    /* ── EXPANDERS ────────────────────────────────────────────── */
    [data-testid="stExpander"] {{
        background: {c["card"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 10px !important;
        margin-bottom: 0.5rem !important;
    }}
    [data-testid="stExpander"] summary {{
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: {c["text2"]} !important;
    }}

    /* ── SPINNER ──────────────────────────────────────────────── */
    [data-testid="stSpinner"] {{
        color: {c["accent"]} !important;
    }}
    [data-testid="stSpinner"] > div {{
        border-top-color: {c["accent"]} !important;
    }}

    /* ── SCROLLBAR ────────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: {c["scrollbar"]};
        border-radius: 10px;
    }}

    /* ── CUSTOM COMPONENTS ────────────────────────────────────── */

    /* Stat card */
    .stat-card {{
        background: {c["card"]};
        border: 1px solid {c["border"]};
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }}
    .stat-card:hover {{
        border-color: {c["accent"]};
        background: {c["card_hover"]};
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(14,165,233,0.1);
    }}
    .stat-label {{
        font-size: 0.68rem;
        font-weight: 700;
        color: {c["text3"]};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }}
    .stat-value {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {c["accent"]};
        line-height: 1;
    }}
    .stat-sub {{
        font-size: 0.72rem;
        color: {c["text2"]};
        margin-top: 3px;
    }}

    /* Status dot */
    .status-dot {{
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }}
    .status-online {{ background: {c["success"]}; box-shadow: 0 0 6px {c["success"]}; }}
    .status-loading {{ background: {c["warning"]}; animation: blink 1.2s infinite; }}
    .status-error   {{ background: {c["danger"]}; }}
    @keyframes blink {{
        0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }}
    }}

    /* Nav item */
    .nav-item {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 16px;
        border-radius: 8px;
        margin: 2px 0;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
        color: #94A3B8;
        transition: all 0.15s ease;
        border: 1px solid transparent;
    }}
    .nav-item.active {{
        background: rgba(14,165,233,0.15);
        color: {c["accent"]};
        border-color: rgba(14,165,233,0.25);
    }}
    .nav-icon {{
        font-size: 1rem;
        width: 20px;
        text-align: center;
    }}

    /* Logo area */
    .logo-area {{
        padding: 24px 20px 16px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 8px;
    }}
    .logo-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: #E2E8F0 !important;
        letter-spacing: -0.3px;
        margin: 0;
    }}
    .logo-sub {{
        font-size: 0.7rem;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 2px;
    }}

    /* Chat message bubbles */
    .msg-wrapper {{
        display: flex;
        flex-direction: column;
        margin-bottom: 1.2rem;
        animation: fadeInUp 0.25s ease forwards;
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .msg-label {{
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
        padding: 0 4px;
    }}
    .msg-label.user  {{ color: {c["accent"]}; text-align: right; }}
    .msg-label.asst  {{ color: {c["text2"]}; }}
    .msg-bubble {{
        padding: 14px 18px;
        border-radius: 16px;
        line-height: 1.65;
        font-size: 0.88rem;
        max-width: 88%;
        word-wrap: break-word;
    }}
    .msg-bubble.user {{
        background: {c["user_bg"]};
        color: #FFFFFF;
        border-radius: 16px 4px 16px 16px;
        align-self: flex-end;
        box-shadow: 0 4px 20px rgba(14,165,233,0.25);
    }}
    .msg-bubble.asst {{
        background: {c["asst_bg"]};
        color: {c["text"]};
        border: {c["asst_border"]};
        border-radius: 4px 16px 16px 16px;
        align-self: flex-start;
    }}

    /* Source card */
    .source-card {{
        background: {c["card"]};
        border: 1px solid {c["border"]};
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }}
    .source-card::before {{
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: {c["accent"]};
        border-radius: 3px 0 0 3px;
    }}
    .source-card:hover {{
        border-color: {c["accent"]};
        transform: translateX(2px);
        box-shadow: 0 4px 16px rgba(14,165,233,0.12);
    }}
    .source-rank {{
        font-size: 0.65rem;
        font-weight: 700;
        color: {c["accent"]};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }}
    .source-name {{
        font-size: 0.78rem;
        font-weight: 600;
        color: {c["text"]};
        margin-bottom: 6px;
        line-height: 1.3;
    }}
    .source-meta {{
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
    }}
    .source-tag {{
        font-size: 0.68rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 20px;
        background: {c["accent_dim"]};
        color: {c["accent"]};
        border: 1px solid {c["border"]};
    }}
    .conf-bar-bg {{
        background: {c["border2"]};
        border-radius: 4px;
        height: 4px;
        margin-top: 8px;
        overflow: hidden;
    }}
    .conf-bar-fill {{
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, {c["accent"]}, {c["accent_hover"]});
        transition: width 0.6s ease;
    }}
    .conf-label {{
        font-size: 0.68rem;
        color: {c["text3"]};
        margin-top: 4px;
        text-align: right;
    }}

    /* Section headers */
    .section-header {{
        font-size: 0.68rem;
        font-weight: 700;
        color: {c["text3"]};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 8px 0 6px;
        border-bottom: 1px solid {c["border2"]};
        margin-bottom: 12px;
    }}

    /* App header bar */
    .app-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0 14px;
        border-bottom: 1px solid {c["border"]};
        margin-bottom: 16px;
    }}
    .app-topbar-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {c["text"]};
        letter-spacing: -0.3px;
    }}
    .app-topbar-sub {{
        font-size: 0.75rem;
        color: {c["text2"]};
        margin-top: 2px;
    }}
    .session-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: {c["accent_dim"]};
        border: 1px solid {c["border"]};
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        color: {c["text2"]};
    }}

    /* Empty state */
    .empty-state {{
        text-align: center;
        padding: 40px 20px;
        color: {c["text3"]};
    }}
    .empty-state-icon {{
        font-size: 2.5rem;
        margin-bottom: 12px;
        opacity: 0.4;
    }}
    .empty-state-title {{
        font-size: 0.9rem;
        font-weight: 600;
        color: {c["text2"]};
        margin-bottom: 6px;
    }}
    .empty-state-sub {{
        font-size: 0.78rem;
        color: {c["text3"]};
        line-height: 1.5;
    }}

    /* Loading dots animation */
    .loading-dots {{
        display: inline-flex;
        gap: 4px;
        align-items: center;
    }}
    .ld {{
        width: 6px; height: 6px;
        background: {c["accent"]};
        border-radius: 50%;
        animation: dotPulse 1.4s infinite ease-in-out;
    }}
    .ld:nth-child(2) {{ animation-delay: .2s; }}
    .ld:nth-child(3) {{ animation-delay: .4s; }}
    @keyframes dotPulse {{
        0%,80%,100% {{ transform: scale(0.6); opacity: 0.4; }}
        40% {{ transform: scale(1); opacity: 1; }}
    }}

    /* Divider */
    .nav-divider {{
        height: 1px;
        background: rgba(255,255,255,0.05);
        margin: 10px 0;
    }}

    /* Panel label */
    .panel-label {{
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {c["text3"]};
        padding: 0 4px;
        margin-bottom: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)


inject_css(C)

# ═══════════════════════════════════════════════════════════════════════════════
# GUARDS
# ═══════════════════════════════════════════════════════════════════════════════

api_key = os.getenv("GROQ_API_KEY", "")
index_ok = os.path.exists(FAISS_INDEX_PATH)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════

if api_key and index_ok:
    vectorstore = load_vectorstore()
    reranker    = load_reranker()
    llm         = get_llm()
    bm25        = build_bm25(vectorstore)
    st.session_state.system_ready = True

    # Index stats
    all_docs   = list(vectorstore.docstore._dict.values())
    doc_sources = set(
        os.path.basename(d.metadata.get("source", ""))
        for d in all_docs
    )
    n_docs   = len(doc_sources)
    n_chunks = len(all_docs)
    idx_files = list(os.scandir(FAISS_INDEX_PATH))
    idx_kb    = sum(f.stat().st_size for f in idx_files) // 1024
else:
    n_docs = n_chunks = idx_kb = 0

# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def stat_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {f'<div class="stat-sub">{sub}</div>' if sub else ""}
    </div>"""


def source_card(doc, score: float, rank: int) -> str:
    name  = os.path.basename(doc.metadata.get("source", "unknown"))
    page  = doc.metadata.get("page", "?")
    # Normalize score to 0-100 for confidence bar
    # ms-marco scores typically range from -10 to +10
    conf  = min(100, max(0, int((score + 5) / 10 * 100)))
    short = name.replace("Linac - Corrective Maintenance - ", "").replace(".pdf", "")
    return f"""
    <div class="source-card">
        <div class="source-rank">Source {rank}</div>
        <div class="source-name">{short}</div>
        <div class="source-meta">
            <span class="source-tag">Page {page}</span>
            <span class="source-tag">Score {score:.2f}</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{conf}%"></div>
        </div>
        <div class="conf-label">Relevance {conf}%</div>
    </div>"""


def empty_sources_html() -> str:
    return """
    <div class="empty-state">
        <div class="empty-state-icon">&#x1F50D;</div>
        <div class="empty-state-title">No sources yet</div>
        <div class="empty-state-sub">
            Retrieved document sources will appear here after your first query.
        </div>
    </div>"""


def loading_dots_html() -> str:
    return """
    <div style="padding:12px 0;">
        <div class="loading-dots">
            <div class="ld"></div>
            <div class="ld"></div>
            <div class="ld"></div>
        </div>
    </div>"""

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT NAVIGATION PANEL (SIDEBAR)
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Logo ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="logo-area">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:32px;height:32px;background:linear-gradient(135deg,#0369A1,#0EA5E9);
                        border-radius:8px;display:flex;align-items:center;justify-content:center;
                        font-weight:900;font-size:1rem;color:#fff;flex-shrink:0;">R</div>
            <div>
                <div class="logo-title">RadAssist</div>
                <div class="logo-sub">Elekta · Linac Platform</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── System Status ─────────────────────────────────────────────────────────
    status_cls   = "status-online" if st.session_state.system_ready else "status-error"
    status_label = "System Ready" if st.session_state.system_ready else "System Error"
    st.markdown(f"""
    <div style="padding:8px 20px 12px;">
        <span class="status-dot {status_cls}"></span>
        <span style="font-size:.75rem;font-weight:600;color:#94A3B8;">{status_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown('<div style="padding:0 12px;">', unsafe_allow_html=True)
    st.markdown("""
    <div class="nav-item active">
        <span class="nav-icon">&#x1F4AC;</span> Active Session
    </div>
    <div class="nav-item">
        <span class="nav-icon">&#x1F4C4;</span> Documents
    </div>
    <div class="nav-item">
        <span class="nav-icon">&#x2699;&#xFE0F;</span> Settings
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

    # ── Dashboard Cards ───────────────────────────────────────────────────────
    st.markdown('<div style="padding:0 12px 4px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Knowledge Base</div>', unsafe_allow_html=True)

    st.markdown(stat_card("Indexed Documents", str(n_docs), "PDF manuals"), unsafe_allow_html=True)
    st.markdown(stat_card("Vector Chunks", f"{n_chunks:,}", f"{idx_kb} KB index"), unsafe_allow_html=True)
    st.markdown(stat_card("Active Session", st.session_state.session_start,
                           f"{st.session_state.query_count} queries"), unsafe_allow_html=True)
    st.markdown(stat_card("Sources Loaded",
                           str(len(st.session_state.last_sources)),
                           "Last query"), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

    # ── Theme Toggle ──────────────────────────────────────────────────────────
    st.markdown('<div style="padding:8px 12px 4px;">', unsafe_allow_html=True)
    theme_label = "Switch to Light Mode" if T == "dark" else "Switch to Dark Mode"
    if st.button(f"{'☀' if T == 'dark' else '🌙'}  {theme_label}", use_container_width=True):
        st.session_state.theme = "light" if T == "dark" else "dark"
        st.rerun()

    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

    # ── Clear Session ─────────────────────────────────────────────────────────
    st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
    if st.button("Clear Session", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.last_sources = []
        st.session_state.query_count  = 0
        st.session_state.session_start = datetime.now().strftime("%H:%M")
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:16px 20px 20px;margin-top:auto;">
        <div style="font-size:.65rem;color:#1E3A5F;text-align:center;line-height:1.6;">
            LangChain &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; Groq<br>
            BM25 &nbsp;·&nbsp; CrossEncoder &nbsp;·&nbsp; Streamlit
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — GUARDS
# ═══════════════════════════════════════════════════════════════════════════════

if not api_key:
    st.markdown(f"""
    <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
                border-radius:12px;padding:20px 24px;max-width:600px;margin:40px auto;">
        <div style="font-size:1rem;font-weight:700;color:#EF4444;margin-bottom:6px;">
            API Key Required
        </div>
        <div style="font-size:.85rem;color:{C['text2']};">
            Add your <strong>GROQ_API_KEY</strong> to <code>.env</code> or Streamlit secrets.
            Get a free key at <a href="https://console.groq.com" style="color:{C['accent']};">
            console.groq.com</a>.
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

if not index_ok:
    st.markdown(f"""
    <div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
                border-radius:12px;padding:20px 24px;max-width:600px;margin:40px auto;">
        <div style="font-size:1rem;font-weight:700;color:{C['warning']};margin-bottom:6px;">
            Vector Index Not Found
        </div>
        <div style="font-size:.85rem;color:{C['text2']};">
            Place your PDFs in <code>documents/</code> and run:
            <code style="background:{C['card']};padding:2px 8px;border-radius:4px;
                         color:{C['accent']};">python ingestion.py</code>
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — TWO COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

chat_col, sources_col = st.columns([65, 35], gap="large")

# ── RIGHT SOURCES PANEL (rendered first to create placeholder) ────────────────
with sources_col:
    st.markdown(f"""
    <div class="app-topbar" style="border-color:{C['border']};">
        <div>
            <div class="app-topbar-title" style="font-size:.9rem;">Reference Panel</div>
            <div class="app-topbar-sub">Retrieved documentation sources</div>
        </div>
        <span class="session-pill">
            <span class="status-dot status-online"></span>
            {len(st.session_state.last_sources)} source(s)
        </span>
    </div>""", unsafe_allow_html=True)

    sources_placeholder = st.empty()

    # Render from session_state
    if st.session_state.last_sources:
        src_html = ""
        for i, (doc, score) in enumerate(st.session_state.last_sources, 1):
            src_html += source_card(doc, score, i)
        sources_placeholder.markdown(src_html, unsafe_allow_html=True)

        # Expandable excerpts
        for i, (doc, _) in enumerate(st.session_state.last_sources, 1):
            name  = os.path.basename(doc.metadata.get("source", "unknown"))
            page  = doc.metadata.get("page", "?")
            short = name.replace("Linac - Corrective Maintenance - ","").replace(".pdf","")
            with st.expander(f"Excerpt {i} — {short} · p.{page}"):
                st.markdown(f"""
                <div style="font-size:.78rem;color:{C['text2']};line-height:1.65;
                            background:{C['bg2']};padding:12px;border-radius:8px;
                            border-left:2px solid {C['accent']};">
                    {doc.page_content[:500].strip()}…
                </div>""", unsafe_allow_html=True)
                # Show local images if extracted
                if os.path.exists(IMAGES_FOLDER):
                    stem    = Path(doc.metadata.get("source","")).stem
                    pattern = f"{stem}_p{int(page)+1:04d}_"
                    imgs    = sorted(Path(IMAGES_FOLDER).glob(f"{pattern}*"))
                    for img_path in imgs[:2]:
                        st.image(str(img_path), use_container_width=True)
    else:
        sources_placeholder.markdown(empty_sources_html(), unsafe_allow_html=True)

# ── LEFT CHAT COLUMN ──────────────────────────────────────────────────────────
with chat_col:

    # Header
    n_msg = len([m for m in st.session_state.messages if isinstance(m, HumanMessage)])
    st.markdown(f"""
    <div class="app-topbar">
        <div>
            <div class="app-topbar-title">Linac Elekta — Maintenance Assistant</div>
            <div class="app-topbar-sub">
                Hybrid RAG · CrossEncoder · Question Decomposition
            </div>
        </div>
        <span class="session-pill">
            <span class="status-dot status-online"></span>
            Session {st.session_state.session_start} &nbsp;·&nbsp; {n_msg} queries
        </span>
    </div>""", unsafe_allow_html=True)

    # Empty state
    if not st.session_state.messages:
        st.markdown(f"""
        <div class="empty-state" style="padding:60px 20px;">
            <div class="empty-state-icon">&#x26A1;</div>
            <div class="empty-state-title">Ready to assist</div>
            <div class="empty-state-sub">
                Ask any question about Elekta Linac corrective maintenance.<br>
                Answers are grounded exclusively in the indexed documentation.
            </div>
            <div style="margin-top:24px;display:flex;flex-wrap:wrap;gap:8px;justify-content:center;">
                <span class="source-tag">Beam Physics</span>
                <span class="source-tag">Cooling Systems</span>
                <span class="source-tag">HT / RF Systems</span>
                <span class="source-tag">Movement Systems</span>
                <span class="source-tag">Dosimetry</span>
                <span class="source-tag">Covers</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # Conversation history
    for msg in st.session_state.messages:
        is_user = isinstance(msg, HumanMessage)
        role    = "user" if is_user else "assistant"
        with st.chat_message(role):
            if is_user:
                # User messages: custom HTML bubble (no markdown needed for user input)
                st.markdown(f"""
                <div class="msg-wrapper">
                    <div class="msg-label user">You</div>
                    <div class="msg-bubble user">{msg.content}</div>
                </div>""", unsafe_allow_html=True)
            else:
                # Assistant messages: native st.markdown so numbered lists / bold / headers render
                st.markdown('<div class="msg-label asst">RadAssist</div>', unsafe_allow_html=True)
                st.markdown(msg.content)

    # Chat input
    user_input = st.chat_input("Ask about Elekta Linac maintenance procedures…")

    # ── QUERY PROCESSING ──────────────────────────────────────────────────────
    if user_input:
        st.session_state.query_count += 1

        # Render user message
        with st.chat_message("user"):
            st.markdown(f"""
            <div class="msg-wrapper">
                <div class="msg-label user">You</div>
                <div class="msg-bubble user">{user_input}</div>
            </div>""", unsafe_allow_html=True)
        st.session_state.messages.append(HumanMessage(content=user_input))

        # Step indicators
        step_box = st.empty()

        # Step 1 — Query expansion
        step_box.markdown(f"""
        <div style="font-size:.78rem;color:{C['text3']};padding:8px 0;">
            <div class="loading-dots" style="display:inline-flex;margin-right:8px;">
                <div class="ld"></div><div class="ld"></div><div class="ld"></div>
            </div>
            Analyzing question…
        </div>""", unsafe_allow_html=True)
        queries = expand_queries(user_input)

        # Step 2 — Retrieval
        step_box.markdown(f"""
        <div style="font-size:.78rem;color:{C['text3']};padding:8px 0;">
            <div class="loading-dots" style="display:inline-flex;margin-right:8px;">
                <div class="ld"></div><div class="ld"></div><div class="ld"></div>
            </div>
            Searching {n_chunks:,} document chunks…
        </div>""", unsafe_allow_html=True)
        seen_ids, candidate_docs = set(), []
        for q in queries:
            for doc in hybrid_retrieve(q):
                doc_id = (doc.metadata.get("source",""), doc.metadata.get("page",""), doc.page_content[:100])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    candidate_docs.append(doc)

        # Step 3 — Reranking
        step_box.markdown(f"""
        <div style="font-size:.78rem;color:{C['text3']};padding:8px 0;">
            <div class="loading-dots" style="display:inline-flex;margin-right:8px;">
                <div class="ld"></div><div class="ld"></div><div class="ld"></div>
            </div>
            Reranking {len(candidate_docs)} candidates…
        </div>""", unsafe_allow_html=True)
        reranked = rerank_docs(user_input, candidate_docs, top_n=TOP_K_RERANKED)
        top_docs = [doc for doc, _ in reranked]

        # Update right panel immediately
        st.session_state.last_sources = reranked
        if reranked:
            src_html = ""
            for i, (doc, score) in enumerate(reranked, 1):
                src_html += source_card(doc, score, i)
            sources_placeholder.markdown(src_html, unsafe_allow_html=True)
        else:
            sources_placeholder.markdown(empty_sources_html(), unsafe_allow_html=True)

        # Build context
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

        # Stream response
        step_box.empty()
        with st.chat_message("assistant"):
            # Label
            st.markdown('<div class="msg-label asst">RadAssist</div>', unsafe_allow_html=True)
            # Native st.markdown stream — markdown (bold, lists, headers) renders correctly
            stream_box  = st.empty()
            full_answer = ""
            for chunk in llm.stream(messages_to_send):
                full_answer += chunk.content
                stream_box.markdown(full_answer + " ▌")
            stream_box.markdown(full_answer)

        st.session_state.messages.append(AIMessage(content=full_answer))
