"""
app.py  –  RAG chatbot — Elekta Linac Maintenance
Improvements:
  Level 1 — Better embedding (BAAI/bge-small-en-v1.5)
  Level 2 — Hybrid search (BM25 + FAISS) + CrossEncoder reranker
  Level 3 — Multi-query retrieval + PDF image display
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

try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    from langchain.retrievers.ensemble import EnsembleRetriever

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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Linac Elekta — Technical Assistant",
    page_icon="https://www.elekta.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: "Inter","Segoe UI",sans-serif; }

    .app-header { border-bottom:1px solid #d9dee6; padding-bottom:.75rem; margin-bottom:1.25rem; }
    .app-title  { font-size:1.5rem; font-weight:700; color:#0d1b2a; margin:0; letter-spacing:-.3px; }
    .app-sub    { font-size:.875rem; color:#5a6474; margin-top:.2rem; }

    .stChatMessage { border-radius:6px; border:1px solid #e8eaed; padding:4px 8px; }

    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"],
    .stChatMessage > div:first-child img,
    .stChatMessage > div:first-child svg { display:none !important; }
    .stChatMessage > div:first-child { width:0!important; min-width:0!important; padding:0!important; }

    .source-row { display:flex; align-items:baseline; gap:12px; padding:6px 10px;
                  border-bottom:1px solid #e8eaed; font-size:.82rem; color:#2c3e50; }
    .source-row:last-child { border-bottom:none; }
    .source-index { font-weight:600; color:#4a5568; min-width:20px; }
    .source-doc   { font-weight:500; color:#1a202c; }
    .source-page  { color:#718096; }
    .source-excerpt { font-size:.80rem; color:#4a5568; background:#f7f8fa;
                      border-left:2px solid #cbd5e0; padding:6px 10px;
                      margin-top:4px; border-radius:3px; white-space:pre-wrap; }

    .badge { display:inline-block; background:#eef2ff; color:#3730a3;
             border-radius:4px; padding:1px 7px; font-size:.72rem;
             font-weight:600; margin-left:6px; vertical-align:middle; }

    section[data-testid="stSidebar"]               { display:none!important; }
    button[data-testid="stSidebarNavToggle"]        { display:none!important; }
    button[data-testid="stSidebarCollapsedControl"] { display:none!important; }
    #MainMenu { display:none!important; }
    footer    { display:none!important; }
</style>
""", unsafe_allow_html=True)

# ── Guards ────────────────────────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY", "")
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to your .env file or Streamlit secrets.")
    st.stop()

if not os.path.exists(FAISS_INDEX_PATH):
    st.error("FAISS index not found. Run: python ingestion.py")
    st.stop()

# ── Load resources (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading vector store...")
def load_vectorstore() -> FAISS:
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True
    )


@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker() -> HuggingFaceCrossEncoder:
    return HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=api_key,
        model_name=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        streaming=True,
    )


@st.cache_resource(show_spinner="Building hybrid search index...")
def build_ensemble(_vectorstore: FAISS) -> EnsembleRetriever:
    """Level 2 — BM25 + FAISS ensemble retriever."""
    # Extract stored docs from FAISS docstore for BM25
    all_docs = list(_vectorstore.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(all_docs, k=TOP_K_RETRIEVAL)
    faiss_ret = _vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
    return EnsembleRetriever(
        retrievers=[bm25, faiss_ret],
        weights=[BM25_WEIGHT, FAISS_WEIGHT],
    )


vectorstore = load_vectorstore()
reranker    = load_reranker()
llm         = get_llm()
ensemble    = build_ensemble(vectorstore)

# ── Multi-query helper (Level 3) ──────────────────────────────────────────────
def generate_query_variants(question: str) -> list[str]:
    """Ask the LLM to rephrase the question for better retrieval coverage."""
    msg = HumanMessage(content=(
        f"You are a technical search assistant. "
        f"Generate {MULTI_QUERY_VARIANTS} alternative phrasings of the following question "
        f"to improve document retrieval from Elekta Linac maintenance manuals. "
        f"Use different technical vocabulary. "
        f"Return ONLY the questions, one per line, no numbering.\n\n"
        f"Question: {question}"
    ))
    resp     = llm.invoke([msg])
    variants = [q.strip() for q in resp.content.strip().split("\n") if q.strip()]
    return [question] + variants[:MULTI_QUERY_VARIANTS]


# ── Reranking helper (Level 2) ────────────────────────────────────────────────
def rerank_docs(query: str, docs: list, top_n: int = TOP_K_RERANKED):
    """Score docs with CrossEncoder and return top_n, filtering out irrelevant ones."""
    if not docs:
        return []
    pairs  = [(query, d.page_content) for d in docs]
    scores = reranker.score(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    # Keep only docs with a positive relevance score (ms-marco: > 0 = relevant)
    return [(doc, score) for score, doc in ranked[:top_n] if score > -2]


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a technical assistant for Elekta Linac corrective maintenance documentation.\n\n"
    "STRICT RULES:\n"
    "1. Answer ONLY from the excerpts provided. Never use training knowledge.\n"
    "2. If the context lacks the answer, respond ONLY with: "
    "   'The provided documents do not contain the information needed to answer this question.'\n"
    "3. FORBIDDEN: 'deduce', 'infer', 'typically', 'usually', 'could be', 'might be', "
    "   'based on general knowledge'. Using these means applying rule 2.\n"
    "4. When the context contains the answer, structure your response:\n"
    "   - Direct answer (1-2 sentences).\n"
    "   - Numbered steps or values from the document.\n"
    "   - References: document name + page only. Omit sources not directly supporting a claim.\n"
    "5. Be concise. No preamble. No filler.\n"
    "6. Detect the language of the question and reply in the same language."
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_btn = st.columns([8, 1])
with col_title:
    st.markdown("""
        <div class="app-header">
            <p class="app-title">Linac Elekta — Technical Maintenance Assistant</p>
            <p class="app-sub">Query your Elekta Linac corrective maintenance manuals.
            Answers are grounded exclusively in the indexed documentation.</p>
        </div>""", unsafe_allow_html=True)
with col_btn:
    st.markdown("<div style='padding-top:.6rem;'>", unsafe_allow_html=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Enter your question about the Elekta Linac manuals...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # ── Level 3: Multi-query retrieval ────────────────────────────────────────
    with st.spinner("Generating query variants..."):
        queries = generate_query_variants(user_input)

    # ── Level 2a: Hybrid retrieval for each variant ───────────────────────────
    with st.spinner("Searching documentation..."):
        seen_ids = set()
        candidate_docs = []
        for q in queries:
            for doc in ensemble.invoke(q):
                doc_id = (
                    doc.metadata.get("source", ""),
                    doc.metadata.get("page", ""),
                    doc.page_content[:100],
                )
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    candidate_docs.append(doc)

    # ── Level 2b: CrossEncoder reranking ─────────────────────────────────────
    with st.spinner("Reranking results..."):
        reranked = rerank_docs(user_input, candidate_docs, top_n=TOP_K_RERANKED)

    top_docs = [doc for doc, _ in reranked]

    # ── Build context message ─────────────────────────────────────────────────
    if top_docs:
        ctx_parts = []
        for d in top_docs:
            src  = os.path.basename(d.metadata.get("source", "unknown"))
            page = d.metadata.get("page", "?")
            ctx_parts.append(f"[Document: {src} | Page: {page}]\n{d.page_content}")
        context_msg = SystemMessage(
            content="Use ONLY the following excerpts to answer. "
                    "Do not use any knowledge outside this context.\n\n"
                    + "\n\n---\n\n".join(ctx_parts)
        )
    else:
        context_msg = SystemMessage(
            content=(
                "No relevant excerpts were found. "
                "You must reply: 'The provided documents do not contain "
                "the information needed to answer this question.'"
            )
        )

    messages_to_send = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + st.session_state.messages
        + [context_msg]
    )

    # ── Stream response ───────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        box         = st.empty()
        full_answer = ""
        for chunk in llm.stream(messages_to_send):
            full_answer += chunk.content
            box.markdown(full_answer + "▌")
        box.markdown(full_answer)

        # ── Sources + images (only when LLM gave a real answer) ───────────────
        answered = "do not contain" not in full_answer.lower()

        if top_docs and answered:
            with st.expander(f"Sources used ({len(top_docs)})"):
                # Source table
                rows = ""
                for i, (doc, score) in enumerate(reranked, 1):
                    src  = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    rows += (
                        f'<div class="source-row">'
                        f'<span class="source-index">{i}.</span>'
                        f'<span class="source-doc">{src}</span>'
                        f'<span class="source-page">Page {page}</span>'
                        f'<span class="badge">score {score:.2f}</span>'
                        f"</div>"
                    )
                st.markdown(rows, unsafe_allow_html=True)

                # Excerpt + image per source
                for i, (doc, _) in enumerate(reranked, 1):
                    src  = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    with st.expander(f"View excerpt — {src}, page {page}"):
                        st.markdown(
                            f'<div class="source-excerpt">'
                            f"{doc.page_content[:600].strip()}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        # Level 3 — show PDF figure if extracted locally
                        if os.path.exists(IMAGES_FOLDER):
                            stem = Path(
                                doc.metadata.get("source", "")
                            ).stem
                            pattern = f"{stem}_p{int(page)+1:04d}_"
                            imgs = sorted(
                                Path(IMAGES_FOLDER).glob(f"{pattern}*")
                            )
                            for img_path in imgs[:3]:   # max 3 figures per page
                                st.image(
                                    str(img_path),
                                    caption=f"Figure from {src}, page {page}",
                                    use_container_width=True,
                                )

    st.session_state.messages.append(AIMessage(content=full_answer))
