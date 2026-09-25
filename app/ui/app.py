"""Streamlit Frontend for Document Q&A Assistant using RAG."""
import os
import time
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        background-color: #EEF2FF;
        color: #4338CA;
    }
    .source-card {
        border-left: 4px solid #4F46E5;
        padding: 10px 14px;
        background-color: #F8FAFC;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if FastAPI backend is reachable."""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return res.status_code == 200, res.json() if res.status_code == 200 else {}
    except Exception:
        return False, {}


def fetch_stats():
    """Fetch vector store stats from backend."""
    try:
        res = requests.get(f"{API_BASE_URL}/stats", timeout=2)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"total_chunks": 0, "total_documents": 0, "sources": [], "groq_api_configured": False}


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = fetch_stats()


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/knowledge-sharing.png", width=64)
    st.title("RAG Control Panel")

    # API Status Check
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success("🟢 API Connected")
    else:
        st.error(f"🔴 API Disconnected ({API_BASE_URL})")
        st.caption("Ensure FastAPI backend is running via `uvicorn app.api.main:app`")

    st.divider()

    # Document Upload Section
    st.subheader("📄 Ingest Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt", "md"],
        help="Upload files to parse, chunk, embed, and store in FAISS."
    )

    if uploaded_file is not None:
        if st.button("🚀 Process & Index Document", use_container_width=True, type="primary"):
            with st.spinner("Processing document (Parsing → Chunking → Embedding → FAISS)..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f" Indexed {data['chunks_created']} chunks!")
                        st.session_state.stats = fetch_stats()
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

    st.divider()

    # RAG Tuning Parameters
    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider("Top-k Chunks", min_value=1, max_value=10, value=4, help="Number of context chunks passed to the LLM.")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="Controls LLM response randomness.")

    st.divider()

    # Vector Store Diagnostics
    st.subheader("📊 Index Statistics")
    stats = fetch_stats()
    st.write(f"**Total Chunks:** `{stats.get('total_chunks', 0)}`")
    st.write(f"**Documents:** `{stats.get('total_documents', 0)}`")
    if stats.get("sources"):
        with st.expander("Indexed Sources"):
            for s in stats["sources"]:
                st.markdown(f"- `{s}`")

    # Clear Index Action
    if st.button("🗑️ Clear Vector Index", use_container_width=True):
        try:
            res = requests.post(f"{API_BASE_URL}/clear")
            if res.status_code == 200:
                st.info("Vector index cleared.")
                st.session_state.messages = []
                st.session_state.stats = fetch_stats()
                st.rerun()
        except Exception as e:
            st.error(f"Error clearing index: {e}")


# --- MAIN CONTENT AREA ---
st.markdown('<div class="main-header">Document Q&A Assistant with RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Grounded Question-Answering using Groq Llama 3.3 70B, FAISS Vector Search, and Local SentenceTransformers.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["💬 Document Chat", "📈 RAG Evaluation", "🏗️ Architecture"])

# --- TAB 1: DOCUMENT CHAT ---
with tab1:
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"🔍 Retrieved Sources ({len(msg['sources'])} chunks)"):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <b>Chunk #{src['chunk_id']}</b> | <b>Source:</b> {src['source']} (Page {src['page']}) | 
                                <b>Cosine Sim:</b> <code>{src['similarity_score']}</code>
                                <br/><br/>
                                <i>"{src['content']}"</i>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # Chat Input Box
    user_query = st.chat_input("Ask a question about your uploaded documents...")

    if user_query:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and querying Groq LLM..."):
                try:
                    payload = {
                        "question": user_query,
                        "top_k": top_k,
                        "temperature": temperature
                    }
                    resp = requests.post(f"{API_BASE_URL}/ask", json=payload)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        answer_text = data["answer"]
                        sources = data.get("sources", [])
                        latency = data.get("latency_seconds", 0.0)
                        model = data.get("model_used", "")

                        st.markdown(answer_text)
                        st.caption(f"⚡ Latency: {latency}s | Model: `{model}`")

                        if sources:
                            with st.expander(f"🔍 Retrieved Sources ({len(sources)} chunks)"):
                                for idx, src in enumerate(sources, 1):
                                    st.markdown(
                                        f"""
                                        <div class="source-card">
                                            <b>Chunk #{src['chunk_id']}</b> | <b>Source:</b> {src['source']} (Page {src['page']}) | 
                                            <b>Cosine Sim:</b> <code>{src['similarity_score']}</code>
                                            <br/><br/>
                                            <i>"{src['content']}"</i>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                        # Save to session
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "sources": sources
                        })
                    else:
                        err_msg = resp.json().get("detail", resp.text)
                        st.error(f"⚠️ Error: {err_msg}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⚠️ Error: {err_msg}"
                        })
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")


# --- TAB 2: RAG EVALUATION ---
with tab2:
    st.subheader("🔬 Evaluate Retrieval Relevance & Answer Accuracy")
    st.markdown(
        "Benchmark the RAG pipeline by measuring **Query-Context Cosine Similarity**, "
        "**Context Precision/Recall**, and **Answer Semantic Match**."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        eval_query = st.text_input(
            "Test Question",
            value="What is Retrieval-Augmented Generation?",
            help="Query to evaluate against retrieved chunks."
        )
        eval_ref = st.text_area(
            "Reference (Ground Truth) Answer",
            value="Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by retrieving relevant documents from an external knowledge base before generating a response.",
            help="Expected ground truth answer used to measure semantic similarity."
        )
        eval_keywords = st.text_input(
            "Expected Keywords (comma-separated)",
            value="retrieval, generation, knowledge base, language models"
        )

        if st.button("🧪 Run Single Evaluation", type="primary"):
            with st.spinner("Evaluating RAG pipeline..."):
                keywords_list = [k.strip() for k in eval_keywords.split(",") if k.strip()]
                eval_payload = {
                    "question": eval_query,
                    "reference_answer": eval_ref,
                    "expected_keywords": keywords_list,
                    "top_k": top_k
                }
                try:
                    res = requests.post(f"{API_BASE_URL}/evaluate", json=eval_payload)
                    if res.status_code == 200:
                        eval_data = res.json()
                        st.session_state["last_eval"] = eval_data
                    else:
                        st.error(f"Evaluation failed: {res.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if "last_eval" in st.session_state:
            last = st.session_state["last_eval"]
            rm = last.get("retrieval_metrics", {})
            am = last.get("answer_metrics", {})

            st.markdown("### 📊 Evaluation Results")
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Avg Retrieval Relevance", f"{rm.get('mean_cosine_similarity', 0.0):.4f}")
            with mcol2:
                st.metric("Top-1 Relevance", f"{rm.get('max_cosine_similarity', 0.0):.4f}")
            with mcol3:
                st.metric("Answer Semantic Match", f"{am.get('semantic_similarity', 0.0):.4f}")

            st.write(f"**Token F1 Score:** `{am.get('token_f1', 0.0)}` | **Precision:** `{am.get('token_precision', 0.0)}` | **Recall:** `{am.get('token_recall', 0.0)}`")
            st.write(f"**Keyword Hit Rate:** `{rm.get('keyword_hit_rate', 0.0) * 100}%`")

            with st.expander("Generated Answer vs Reference"):
                st.write("**Generated:**", last.get("generated_answer", ""))
                st.write("**Reference:**", last.get("reference_answer", ""))

    st.divider()
    st.subheader("⚡ Automated Benchmark Suite")
    if st.button("🚀 Run Full Benchmark Suite"):
        with st.spinner("Running batch evaluation on test dataset..."):
            try:
                res = requests.post(f"{API_BASE_URL}/evaluate/benchmark?top_k={top_k}")
                if res.status_code == 200:
                    bench = res.json()
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        st.metric("Avg Retrieval Relevance", f"{bench['average_retrieval_relevance']:.4f}")
                    with b2:
                        st.metric("Avg Answer Semantic Sim", f"{bench['average_answer_semantic_similarity']:.4f}")
                    with b3:
                        st.metric("Avg Token F1", f"{bench['average_token_f1']:.4f}")
                    st.json(bench["details"])
                else:
                    st.error(f"Benchmark failed: {res.text}")
            except Exception as e:
                st.error(f"Error: {e}")


# --- TAB 3: ARCHITECTURE ---
with tab3:
    st.subheader("🏗️ System Architecture & Data Flow")
    st.markdown("""
    ```
    +------------------+      +-------------------+      +-------------------------------+
    | Document Files   | ---> | Document Loader   | ---> | RecursiveCharacterTextSplitter|
    | (PDF, DOCX, TXT) |      | (pypdf, docx, txt)|      | (Chunk size: 600, Overlap: 100)|
    +------------------+      +-------------------+      +---------------+---------------+
                                                                         |
                                                                         v
    +--------------------+      +----------------------------------------+
    | FAISS Vector Index | <--- | SentenceTransformer (all-MiniLM-L6-v2) |
    | (IndexFlatIP)      |      | (Dense 384-d L2-Normalized Vectors)    |
    +--------------------+      +----------------------------------------+
                                                 |
                                                 v
    +--------------------+      +----------------------------------------+
    | User Query         | ---> | FAISS Similarity Search (Top-k Chunks) |
    +--------------------+      +-------------------+--------------------+
                                                    |
                                                    v
    +--------------------+      +----------------------------------------+
    | Grounded Response  | <--- | Groq API (llama-3.3-70b-versatile)     |
    +--------------------+      +----------------------------------------+
    ```
    """)
    st.info(
        "**Key Production Features:**\n"
        "- Zero embedding API costs with local `all-MiniLM-L6-v2`.\n"
        "- Ultra-fast inference with Groq `llama-3.3-70b-versatile`.\n"
        "- In-memory FAISS indexing with disk persistence.\n"
        "- Built-in quantitative retrieval & answer accuracy evaluation metrics."
    )
