"""
app.py  –  Streamlit RAG chatbot powered by ChatGroq + FAISS
Usage:  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import (
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    TOP_K,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
)

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG – Linac Elekta",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .stChatMessage { border-radius: 10px; }
        .source-card {
            background: #f0f2f6;
            border-left: 3px solid #4c8bf5;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 4px 0;
            font-size: 0.85em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Elekta_logo.svg/320px-Elekta_logo.svg.png",
        width=160,
    )
    st.title("⚙️ Settings")

    api_key = st.text_input(
        "🔑 Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_…",
        help="Free key at https://console.groq.com",
    )

    model_name = st.selectbox(
        "🤖 Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        help="llama-3.3-70b-versatile gives the best accuracy for technical manuals.",
    )

    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
        help="Lower → more factual / deterministic answers.",
    )

    top_k = st.slider(
        "📚 Chunks retrieved (k)",
        min_value=1,
        max_value=10,
        value=TOP_K,
        help="How many document chunks are passed as context per query.",
    )

    st.divider()

    show_sources = st.toggle("📄 Show source excerpts", value=True)
    language     = st.radio("🌐 Reply language", ["Auto-detect", "English", "Français"], index=0)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("ℹ️ Index info", use_container_width=True):
            st.session_state.show_info = not st.session_state.get("show_info", False)

    if st.session_state.get("show_info", False):
        if os.path.exists(FAISS_INDEX_PATH):
            idx_files = list(os.scandir(FAISS_INDEX_PATH))
            total_kb  = sum(f.stat().st_size for f in idx_files) // 1024
            st.info(f"FAISS index: **{len(idx_files)} file(s)**, ~{total_kb} KB")
        else:
            st.warning("No index found — run `python ingestion.py` first.")

    st.caption("LangChain · FAISS · Groq · Streamlit")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ RAG Chatbot — Linac Elekta Maintenance")
st.caption(
    "Ask questions about your Elekta Linac technical manuals. "
    "Answers are grounded exclusively in the indexed documents."
)

# ── Guard: API key ────────────────────────────────────────────────────────────
if not api_key:
    st.warning(
        "👈 Enter your **Groq API Key** in the sidebar to start.  \n"
        "Get a free key at [console.groq.com](https://console.groq.com)."
    )
    st.stop()

# ── Guard: FAISS index ────────────────────────────────────────────────────────
if not os.path.exists(FAISS_INDEX_PATH):
    st.error(
        "❌ **FAISS index not found.**  \n"
        "Place your PDFs in the `documents/` folder and run:\n"
        "```\npython ingestion.py\n```"
    )
    st.stop()

# ── Load resources (cached per session) ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading vector store…")
def load_vectorstore() -> FAISS:
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        emb,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=False)
def get_llm(key: str, model: str, temp: float) -> ChatGroq:
    return ChatGroq(api_key=key, model_name=model, temperature=temp, streaming=True)


vectorstore = load_vectorstore()
llm         = get_llm(api_key, model_name, temperature)
retriever   = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": top_k},
)

# ── System prompt ─────────────────────────────────────────────────────────────
_lang_instruction = {
    "Auto-detect": "Answer in the same language the user used in their question.",
    "English":     "Always answer in English.",
    "Français":    "Réponds toujours en français.",
}

SYSTEM_PROMPT = (
    "You are a technical expert assistant specialized in Elekta Linac maintenance "
    "(corrective maintenance, beam physics, cooling systems, dosimetry, HT/RF systems, "
    "movement systems, covers, etc.).\n"
    "Rules:\n"
    "1. Answer ONLY using the context provided — never add information not present in it.\n"
    "2. If the answer is not in the context, reply exactly: "
    "'This information is not found in the provided documents.'\n"
    "3. Be precise and cite the exact steps or values from the document when relevant.\n"
    "4. When quoting, mention the document name and page number.\n"
    f"5. {_lang_instruction[language]}"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about the Elekta Linac manuals…")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Retrieve relevant chunks
    with st.spinner("🔍 Searching documents…"):
        docs = retriever.invoke(user_input)

    # Build context message
    if docs:
        ctx_parts = []
        for d in docs:
            src  = os.path.basename(d.metadata.get("source", "unknown"))
            page = d.metadata.get("page", "?")
            ctx_parts.append(f"[Document: {src} | Page: {page}]\n{d.page_content}")
        context_msg = SystemMessage(
            content="Use the following excerpts to answer:\n\n"
            + "\n\n---\n\n".join(ctx_parts)
        )
    else:
        context_msg = SystemMessage(
            content="No relevant document excerpts found for this question."
        )

    # Build full message list for the LLM
    messages_to_send = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + st.session_state.messages
        + [context_msg]
    )

    # Stream response
    with st.chat_message("assistant"):
        response_box = st.empty()
        full_answer  = ""
        for chunk in llm.stream(messages_to_send):
            full_answer += chunk.content
            response_box.markdown(full_answer + "▌")
        response_box.markdown(full_answer)  # remove cursor

        # Source documents
        if docs and show_sources:
            with st.expander(f"📄 {len(docs)} source excerpt(s) used"):
                for i, doc in enumerate(docs, 1):
                    src  = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>#{i} — {src} | Page {page}</strong><br>'
                        f'<em>{doc.page_content[:400].strip()}…</em>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(AIMessage(content=full_answer))
