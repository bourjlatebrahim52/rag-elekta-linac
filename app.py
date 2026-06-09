"""
app.py — Elekta Linac Maintenance Assistant
Simple clean interface — Backend unchanged
"""

import os
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
    TOP_K_RETRIEVAL, TOP_K_RERANKED,
    RERANKER_MODEL, MULTI_QUERY_VARIANTS,
    DEFAULT_MODEL, DEFAULT_TEMPERATURE,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

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
        "You are an expert at analyzing technical maintenance questions for Elekta Linac. "
        "The maintenance manuals contain text in BOTH French and English "
        "(technical terms, item names, and part numbers are usually in English). "
        "Break this complex question into 3-4 simple, independent sub-questions. "
        "For each sub-question, write it in BOTH French AND English on separate lines "
        "so retrieval works regardless of the document language. "
        "Each sub-question must be answerable on its own from a maintenance manual. "
        "Return ONLY the sub-questions, one per line, no numbering, no explanation.\n\n"
        f"Complex question: {question}"
    ))
    resp   = llm.invoke([msg])
    sub_qs = [q.strip() for q in resp.content.strip().split("\n") if q.strip()]
    return [question] + sub_qs[:6]

def generate_query_variants(question: str) -> list[str]:
    msg = HumanMessage(content=(
        "The Elekta Linac maintenance manuals contain text in BOTH French and English. "
        "Technical terms, item names, fault codes, and part numbers are usually in English "
        "even in French sections (e.g. 'Bending OT', 'PCB DIE-RHB', 'Gun I mon'). "
        "Generate 3 alternative phrasings of the question below:\n"
        "  - 1 variant in French using Elekta technical vocabulary\n"
        "  - 1 variant in English using Elekta technical vocabulary\n"
        "  - 1 variant using exact Elekta item names or fault codes if applicable\n"
        "Return ONLY the 3 questions, one per line, no numbering.\n\n"
        f"Question: {question}"
    ))
    resp     = llm.invoke([msg])
    variants = [q.strip() for q in resp.content.strip().split("\n") if q.strip()]
    return [question] + variants[:3]

def translate_to_english(question: str) -> str:
    """Translate a French question to English for bilingual document retrieval."""
    msg = HumanMessage(content=(
        "Translate this Elekta Linac maintenance question to English. "
        "Keep all technical terms, item names, fault codes, and part numbers exactly as-is. "
        "Return ONLY the English translation, nothing else.\n\n"
        f"Question: {question}"
    ))
    return llm.invoke([msg]).content.strip()

def looks_french(text: str) -> bool:
    """Simple heuristic to detect French questions."""
    french_words = ["le ","la ","les ","du ","de ","des ","est ","que ","pour ",
                    "comment","quoi","quelle","quelles","quel","quels","faire ",
                    "système","défaut","surchauffe","remplacement","étapes",
                    "après","avant","lors","lors","quand","pourquoi"]
    return any(w in text.lower() for w in french_words)

def expand_queries(question: str) -> list[str]:
    # Standard expansion (variants or decomposition)
    if is_complex_question(question):
        base = decompose_question(question)
    else:
        base = generate_query_variants(question)

    # If question is French, also search with English translation
    # (Elekta manuals mix French prose with English technical terms)
    if looks_french(question):
        english_q = translate_to_english(question)
        all_queries = list(dict.fromkeys([question, english_q] + base))
        return all_queries[:6]

    return base

def rerank_docs(query: str, docs: list, top_n: int = TOP_K_RERANKED):
    if not docs:
        return []
    pairs  = [(query, d.page_content) for d in docs]
    scores = reranker.score(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [(doc, score) for score, doc in ranked[:top_n]]

SYSTEM_PROMPT = (
    "You are a technical assistant for Elekta Linac corrective maintenance documentation.\n\n"
    "STRICT RULES:\n"
    "1. Answer ONLY from the excerpts provided. Never use training knowledge.\n"
    "2. For complex multi-part questions: answer each part separately. "
    "   If a part is NOT found in the context, write exactly: "
    "   'Cette information n'est pas disponible dans les documents fournis.' "
    "   Do NOT write generic advice, safety tips, or any content not directly quoted from the documents.\n"
    "3. If the context contains NO relevant information at all, respond ONLY with: "
    "   'Les documents fournis ne contiennent pas les informations nécessaires pour répondre à cette question.'\n"
    "4. ABSOLUTELY FORBIDDEN — never write these even as advice:\n"
    "   - Generic safety instructions ('Assurez-vous de...', 'Suivez les procédures...')\n"
    "   - Tool or spare parts checklists not from the document\n"
    "   - Recommendations based on general knowledge\n"
    "   - Words: 'deduce', 'infer', 'typically', 'usually', 'could be', 'might be', 'generally'\n"
    "   Violating this rule means applying rule 3.\n"
    "5. Structure: Direct answer → Numbered steps from the document → Reference (doc name + page).\n"
    "6. Be concise. No preamble. No filler.\n"
    "7. Detect the language of the question and reply in the same language."
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Elekta Linac — Assistant Maintenance",
    page_icon="⚕",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu, footer, header { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    button[data-testid="stSidebarCollapsedControl"] { display: none !important; }

    /* Clean white page */
    .stApp { background: #ffffff !important; }
    .block-container { max-width: 820px !important; padding: 2rem 1rem 1rem !important; }

    /* Header */
    .app-header {
        text-align: center;
        padding-bottom: 1.5rem;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    .app-sub {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 4px;
    }

    /* Chat messages */
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] { display: none !important; }
    [data-testid="stChatMessage"] > div:first-child {
        width: 0 !important; min-width: 0 !important; padding: 0 !important;
    }
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.25rem 0 !important;
    }

    /* User bubble */
    .user-bubble {
        background: #1d4ed8;
        color: #ffffff;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        display: inline-block;
        max-width: 80%;
        float: right;
        font-size: 0.9rem;
        line-height: 1.5;
        clear: both;
    }
    .user-wrap { overflow: hidden; margin-bottom: 0.75rem; }

    /* Assistant response */
    .stMarkdown p  { font-size: 0.9rem !important; color: #1f2937 !important; line-height: 1.7 !important; }
    .stMarkdown li { font-size: 0.88rem !important; color: #374151 !important; line-height: 1.65 !important; }
    .stMarkdown strong { color: #111827 !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-size: 0.95rem !important; color: #1d4ed8 !important;
        font-weight: 700 !important; margin: 10px 0 4px !important;
    }
    .stMarkdown code {
        background: #f3f4f6 !important; color: #1d4ed8 !important;
        padding: 1px 6px !important; border-radius: 4px !important;
        font-size: 0.82rem !important;
    }

    /* Source expander */
    [data-testid="stExpander"] {
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        background: #f9fafb !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.78rem !important;
        color: #6b7280 !important;
        font-weight: 600 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border: 1.5px solid #d1d5db !important;
        border-radius: 12px !important;
        background: #ffffff !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #1d4ed8 !important;
        box-shadow: 0 0 0 3px rgba(29,78,216,0.1) !important;
    }

    /* Clear button */
    .stButton > button {
        background: transparent !important;
        color: #9ca3af !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
        font-size: 0.75rem !important;
        padding: 3px 10px !important;
    }
    .stButton > button:hover {
        color: #ef4444 !important;
        border-color: #ef4444 !important;
        background: #fef2f2 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GUARDS
# ─────────────────────────────────────────────────────────────────────────────

api_key  = os.getenv("GROQ_API_KEY", "")
index_ok = os.path.exists(FAISS_INDEX_PATH)

if not api_key:
    st.error("GROQ_API_KEY manquant. Ajoutez-le dans `.env` ou les secrets Streamlit.")
    st.stop()
if not index_ok:
    st.error("Index FAISS introuvable. Lancez : `python ingestion.py`")
    st.stop()

vectorstore = load_vectorstore()
reranker    = load_reranker()
llm         = get_llm()
bm25        = build_bm25(vectorstore)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

# Header
col1, col2 = st.columns([8, 1])
with col1:
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Elekta Linac — Assistant de Maintenance</div>
        <div class="app-sub">Recherche dans les manuels de maintenance corrective</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    if st.button("Effacer"):
        st.session_state.messages = []
        st.rerun()

# Chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(
                f'<div class="user-wrap"><div class="user-bubble">{msg.content}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Input
user_input = st.chat_input("Posez votre question sur la maintenance Elekta Linac…")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(
            f'<div class="user-wrap"><div class="user-bubble">{user_input}</div></div>',
            unsafe_allow_html=True,
        )
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.spinner("Recherche en cours…"):
        queries = expand_queries(user_input)
        seen_ids, candidate_docs = set(), []
        for q in queries:
            for doc in hybrid_retrieve(q):
                doc_id = (doc.metadata.get("source",""), doc.metadata.get("page",""), doc.page_content[:100])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    candidate_docs.append(doc)
        reranked = rerank_docs(user_input, candidate_docs, top_n=TOP_K_RERANKED)
        top_docs = [doc for doc, _ in reranked]

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
    with st.chat_message("assistant"):
        box = st.empty()
        full_answer = ""
        for chunk in llm.stream(messages_to_send):
            full_answer += chunk.content
            box.markdown(full_answer + " ▌")
        box.markdown(full_answer)

        # Sources
        answered = "do not contain" not in full_answer.lower() and "ne contiennent pas" not in full_answer.lower()
        if top_docs and answered:
            with st.expander(f"Sources utilisées ({len(top_docs)})"):
                for i, (doc, score) in enumerate(reranked, 1):
                    src  = os.path.basename(doc.metadata.get("source", ""))
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**{i}.** `{src}` — page {page} &nbsp; *(score : {score:.2f})*")
                    with st.expander(f"Extrait {i}"):
                        st.caption(doc.page_content[:500].strip())
                        if os.path.exists(IMAGES_FOLDER):
                            stem    = Path(doc.metadata.get("source","")).stem
                            pattern = f"{stem}_p{int(page)+1:04d}_"
                            imgs    = sorted(Path(IMAGES_FOLDER).glob(f"{pattern}*"))
                            for img_path in imgs[:2]:
                                st.image(str(img_path), use_container_width=True)

    st.session_state.messages.append(AIMessage(content=full_answer))
