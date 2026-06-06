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
    page_title="Linac Elekta — Technical Assistant",
    page_icon="https://www.elekta.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Global font */
        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", sans-serif;
        }

        /* Main header */
        .app-header {
            border-bottom: 1px solid #d9dee6;
            padding-bottom: 0.75rem;
            margin-bottom: 1.25rem;
        }
        .app-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0d1b2a;
            margin: 0;
            letter-spacing: -0.3px;
        }
        .app-subtitle {
            font-size: 0.875rem;
            color: #5a6474;
            margin-top: 0.2rem;
        }

        /* Chat messages */
        .stChatMessage {
            border-radius: 6px;
            border: 1px solid #e8eaed;
            padding: 4px 8px;
        }

        /* Source card — collapsed view */
        .source-row {
            display: flex;
            align-items: baseline;
            gap: 12px;
            padding: 6px 10px;
            border-bottom: 1px solid #e8eaed;
            font-size: 0.82rem;
            color: #2c3e50;
        }
        .source-row:last-child { border-bottom: none; }
        .source-index {
            font-weight: 600;
            color: #4a5568;
            min-width: 20px;
        }
        .source-doc  { font-weight: 500; color: #1a202c; }
        .source-page { color: #718096; }

        /* Source excerpt (nested expander) */
        .source-excerpt {
            font-size: 0.80rem;
            color: #4a5568;
            background: #f7f8fa;
            border-left: 2px solid #cbd5e0;
            padding: 6px 10px;
            margin-top: 4px;
            border-radius: 3px;
            white-space: pre-wrap;
        }

        /* Guard messages */
        .guard-box {
            background: #fff8e1;
            border: 1px solid #f0c040;
            border-radius: 5px;
            padding: 10px 14px;
            font-size: 0.875rem;
            color: #5d4037;
        }

        /* Sidebar refinements */
        section[data-testid="stSidebar"] {
            background: #f7f8fa;
            border-right: 1px solid #dde2ea;
        }
        section[data-testid="stSidebar"] label {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            color: #374151 !important;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Elekta_logo.svg/320px-Elekta_logo.svg.png",
        width=150,
    )
    st.markdown("### Settings")

    api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_...",
        help="Obtain a free key at https://console.groq.com",
    )

    model_name = st.selectbox(
        "Model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        help="llama-3.3-70b-versatile is recommended for technical documentation.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
        help="Lower values produce more factual, deterministic answers.",
    )

    top_k = st.slider(
        "Chunks retrieved (k)",
        min_value=1,
        max_value=10,
        value=TOP_K,
        help="Number of document chunks retrieved per query.",
    )

    st.divider()

    show_sources = st.toggle("Show sources", value=True)
    language = st.radio(
        "Reply language",
        ["Auto-detect", "English", "Français"],
        index=0,
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Index info", use_container_width=True):
            st.session_state.show_info = not st.session_state.get("show_info", False)

    if st.session_state.get("show_info", False):
        if os.path.exists(FAISS_INDEX_PATH):
            idx_files = list(os.scandir(FAISS_INDEX_PATH))
            total_kb  = sum(f.stat().st_size for f in idx_files) // 1024
            st.info(f"FAISS index: {len(idx_files)} file(s), ~{total_kb} KB")
        else:
            st.warning("No index found. Run python ingestion.py first.")

    st.caption("LangChain · FAISS · Groq · Streamlit")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <p class="app-title">Linac Elekta — Technical Maintenance Assistant</p>
        <p class="app-subtitle">
            Query your Elekta Linac corrective maintenance manuals.
            Answers are grounded exclusively in the indexed documentation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Guard: API key ────────────────────────────────────────────────────────────
if not api_key:
    st.markdown(
        '<div class="guard-box">'
        'Enter your <strong>Groq API Key</strong> in the sidebar to continue. '
        'A free key is available at '
        '<a href="https://console.groq.com" target="_blank">console.groq.com</a>.'
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Guard: FAISS index ────────────────────────────────────────────────────────
if not os.path.exists(FAISS_INDEX_PATH):
    st.error(
        "FAISS index not found. "
        "Place your PDFs in the documents/ folder and run: python ingestion.py"
    )
    st.stop()

# ── Load resources (cached per session) ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading vector store...")
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

# ── System prompt ─────────────────────────────────────────────────────────────
_lang_instruction = {
    "Auto-detect": "Detect the language of the user's question and reply in the same language.",
    "English":     "Always reply in English.",
    "Français":    "Réponds toujours en français.",
}

SYSTEM_PROMPT = (
    "You are a technical expert assistant specialized in Elekta Linac corrective maintenance "
    "(beam physics, cooling systems, dosimetry, HT/RF systems, movement systems, covers, etc.).\n\n"
    "Response rules — follow them strictly:\n\n"
    "1. BASE YOUR ANSWER SOLELY ON THE PROVIDED CONTEXT. "
    "   Do not add any information that is not explicitly present in the retrieved excerpts.\n"
    "2. If the context does not contain sufficient information to answer the question, "
    "   respond with exactly: "
    "   'The provided documents do not contain enough information to answer this question.'\n"
    "3. STRUCTURE every response as follows:\n"
    "   - Direct answer: one or two sentences that directly answer the question.\n"
    "   - Supporting details: numbered steps, values, or conditions taken verbatim or closely "
    "     paraphrased from the document.\n"
    "   - References: cite only the document name and page number for each claim you make. "
    "     Do not list a source unless it directly supports a specific claim in your answer. "
    "     If a retrieved chunk is only loosely related, omit it entirely.\n"
    "4. Never speculate, extrapolate, or provide general engineering knowledge outside the context.\n"
    "5. Be concise. Avoid unnecessary preamble or filler sentences.\n"
    f"6. {_lang_instruction[language]}"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render conversation history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Enter your question about the Elekta Linac manuals...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # ── Retrieve with relevance scores ────────────────────────────────────────
    with st.spinner("Searching documentation..."):
        docs_with_scores = vectorstore.similarity_search_with_score(
            user_input, k=top_k
        )

    # Filter: keep only chunks whose L2 distance < 1.2  (cosine sim > ~0.4)
    # Then cap at top 3 for source display to avoid noise
    SCORE_THRESHOLD = 1.2
    MAX_DISPLAY     = 3

    filtered = [
        (doc, score)
        for doc, score in docs_with_scores
        if score < SCORE_THRESHOLD
    ]
    # Sort ascending (lower L2 = more relevant) and keep top MAX_DISPLAY
    filtered.sort(key=lambda x: x[1])
    top_docs    = [d for d, _ in filtered[:MAX_DISPLAY]]
    all_context = [d for d, _ in docs_with_scores]   # full set for LLM context

    # ── Build context message for the LLM (use full retrieved set) ───────────
    if all_context:
        ctx_parts = []
        for d in all_context:
            src  = os.path.basename(d.metadata.get("source", "unknown"))
            page = d.metadata.get("page", "?")
            ctx_parts.append(
                f"[Document: {src} | Page: {page}]\n{d.page_content}"
            )
        context_msg = SystemMessage(
            content=(
                "Use ONLY the following excerpts to answer. "
                "Do not use any knowledge outside this context.\n\n"
                + "\n\n---\n\n".join(ctx_parts)
            )
        )
    else:
        context_msg = SystemMessage(
            content="No relevant excerpts were found for this question."
        )

    # ── Build full message list ───────────────────────────────────────────────
    messages_to_send = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + st.session_state.messages
        + [context_msg]
    )

    # ── Stream response ───────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        response_box = st.empty()
        full_answer  = ""
        for chunk in llm.stream(messages_to_send):
            full_answer += chunk.content
            response_box.markdown(full_answer + "▌")
        response_box.markdown(full_answer)

        # ── Source display ────────────────────────────────────────────────────
        if show_sources and top_docs:
            with st.expander(f"Sources used ({len(top_docs)})"):
                rows_html = ""
                for i, doc in enumerate(top_docs, 1):
                    src  = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    rows_html += (
                        f'<div class="source-row">'
                        f'<span class="source-index">{i}.</span>'
                        f'<span class="source-doc">{src}</span>'
                        f'<span class="source-page">Page {page}</span>'
                        f"</div>"
                    )
                st.markdown(rows_html, unsafe_allow_html=True)

                # Optional: show raw excerpt per source in nested expanders
                for i, doc in enumerate(top_docs, 1):
                    src  = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    with st.expander(f"View excerpt — {src}, page {page}"):
                        st.markdown(
                            f'<div class="source-excerpt">'
                            f"{doc.page_content[:600].strip()}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        elif show_sources and not top_docs:
            st.caption("No sufficiently relevant sources were identified for this query.")

    st.session_state.messages.append(AIMessage(content=full_answer))
