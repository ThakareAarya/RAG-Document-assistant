# 📚 Production Document Q&A Assistant using RAG

A production-quality **Retrieval-Augmented Generation (RAG)** application designed to ingest enterprise documents (PDF, DOCX, TXT), index them into a high-performance local vector database (**FAISS**), retrieve contextually relevant passages via dense semantic search, and synthesize accurate, grounded answers using **Groq's LPU inference engine (`llama-3.3-70b-versatile`)**.

---

## 🌟 Key Features

- **⚡ Blazing Fast Generation**: Powered by Groq's LPU inference running Meta's `llama-3.3-70b-versatile` (>250 tokens/sec).
- **🔒 100% Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` locally on CPU/GPU. Zero third-party embedding API cost or latency.
- **⚡ In-Memory & Persisted Vector Store**: Custom FAISS (`IndexFlatIP`) index with disk persistence (`.bin` and `metadata.json`).
- **📑 Multi-Format Parsing**: Production-grade document loading for `.pdf` (pypdf), `.docx` (python-docx), and `.txt`/`.md`.
- **✂️ Recursive Semantic Chunking**: Chunk size and overlap tuned to preserve sentence and paragraph integrity.
- **🛡️ Strict Grounding & Anti-Hallucination**: Prompt engineering designed to answer only from context and decline gracefully when data is missing.
- **📊 Built-In Evaluation Suite**: Quantitative evaluation of **Retrieval Relevance** (Cosine Similarity) and **Answer Accuracy** (Semantic Similarity, Token F1, Hit Rate).
- **🚀 Dual Interface**: Full **FastAPI** REST backend + clean, interactive **Streamlit** frontend.

---

## 🏗️ Architecture & Data Flow

```
[Document Files] ──> [Loader (PDF/DOCX/TXT)] ──> [Recursive Text Splitter]
                                                               │
                                                               ▼
[FAISS Index] <── [SentenceTransformer (all-MiniLM-L6-v2)] <───┘
      │
      │ (Top-k Dense Retrieval)
      ▼
[Grounded Prompt Formulation] ──> [Groq LLM (llama-3.3-70b-versatile)] ──> [Answer + Cited Sources]
                                                                                    │
                                                                                    ▼
                                                                        [Evaluation Module]
```

Detailed architecture specifications can be found in [`architecture.md`](./architecture.md).

---

## 📂 Project Structure

```
rag-document-assistant/
├── .env.example                # Template for environment variables (GROQ_API_KEY)
├── .env                        # Local secrets (ignored by Git)
├── .gitignore                  # Git exclusions
├── requirements.txt            # Pinned dependencies
├── README.md                   # Documentation and setup guide
├── architecture.md             # System architecture & component breakdown
├── test_rag.py                 # End-to-end integration and verification script
│
├── data/
│   ├── samples/                # Sample test documents
│   └── uploads/                # User uploaded documents
│
├── vectorstore/                # FAISS binary index and JSON metadata
│   ├── index.bin
│   └── metadata.json
│
└── app/
    ├── __init__.py
    ├── config.py               # Centralized configuration
    ├── ingestion/
    │   ├── __init__.py
    │   └── loader.py           # Multi-format document parser & chunker
    ├── retrieval/
    │   ├── __init__.py
    │   ├── embedder.py         # Local SentenceTransformer embeddings singleton
    │   └── vectorstore.py      # FAISS vector store with persistence
    ├── llm/
    │   ├── __init__.py
    │   ├── prompts.py          # Grounded prompts & citation formatters
    │   └── groq_client.py      # Groq API client with fallback demo mode
    ├── eval/
    │   ├── __init__.py
    │   └── evaluate.py         # Retrieval relevance & Answer accuracy metrics
    ├── api/
    │   ├── __init__.py
    │   └── main.py             # FastAPI REST endpoints
    └── ui/
        └── app.py              # Streamlit interactive UI
```

---

## 🚀 Quickstart & Installation

### 1. Clone or Open the Repository
```bash
cd rag-document-assistant
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and enter your Groq API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
*(Note: If you don't have an API key yet, the application automatically runs in **Demo Mode**, testing the full local ingestion, embedding, FAISS search, and evaluation pipeline).*

---

## 🧪 Run End-to-End Sanity Check

Run the self-contained verification script:
```bash
python test_rag.py
```
This script will:
1. Parse the sample document in `data/samples/sample_rag_overview.txt`.
2. Generate semantic chunks and embed them locally via `all-MiniLM-L6-v2`.
3. Index the chunks in FAISS and run top-k similarity searches.
4. Execute query generation via Groq.
5. Run the evaluation benchmark and output quantitative scores.

---

## 🖥️ Running the Application

### Option A: Start the FastAPI Backend
```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- API Health Endpoint: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Option B: Start the Streamlit Frontend
In a separate terminal:
```bash
streamlit run app/ui/app.py
```
- Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Multipart upload for PDF, DOCX, TXT. Chunks and embeds into FAISS. |
| `POST` | `/ask` | Queries the index, retrieves top-k chunks, and returns the grounded answer + sources. |
| `GET` | `/stats` | Returns current index metrics (chunks count, sources, embedding model). |
| `POST` | `/evaluate` | Evaluates retrieval relevance and answer accuracy for a given query. |
| `POST` | `/evaluate/benchmark` | Runs batch evaluation over reference test cases. |
| `POST` | `/clear` | Resets the FAISS index and clears disk artifacts. |

---

## 📊 Evaluation Framework

The evaluation system in `app/eval/evaluate.py` quantifies RAG quality across two key dimensions:

1. **Retrieval Relevance**:
   - **Mean / Max Cosine Similarity**: Measures how closely the retrieved vector matches the query embedding.
   - **Keyword Hit Rate**: Verifies whether critical domain terms or expected entities appear in retrieved passages.
2. **Answer Accuracy**:
   - **Semantic Similarity**: Cosine distance between generated response and expected reference answer.
   - **Token F1 & Precision/Recall**: Measures exact lexical and semantic overlap with ground truth.

---

## 💼 Interview & Resume Talking Points

- **Why SentenceTransformers + FAISS over Cloud Vector DBs?**
  Eliminates embedding API rate limits, per-token costs, and latency overhead. `all-MiniLM-L6-v2` produces compact 384-d vectors that execute inner product searches in sub-millisecond times on standard CPU instances.
- **How Hallucination is Prevented:**
  System prompts explicitly restrict the model to the retrieved context window and enforce standard refusal phrases when evidence is missing.
- **Chunking Strategy Tradeoffs:**
  Using `RecursiveCharacterTextSplitter` with 600 characters and 100 character overlap preserves semantic continuity across chunk boundaries without diluting embedding density.
