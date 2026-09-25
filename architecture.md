# Document Q&A Assistant with RAG — Architecture

## 1. Executive Overview

This project implements a production-grade **Retrieval-Augmented Generation (RAG)** pipeline designed to ingest enterprise documents (PDF, DOCX, TXT), index them into a high-performance local vector database (FAISS), retrieve contextually relevant chunks via dense semantic search, and synthesize accurate, grounded answers using Groq's high-speed inference engine (`llama-3.3-70b-versatile`).

---

## 2. High-Level Flow

```
+---------------------------------------------------------------------------------------------+
|                                     INGESTION PIPELINE                                      |
|                                                                                             |
|   +------------------+      +-------------------+      +-------------------------------+    |
|   | Document Files   | ---> | Document Loader   | ---> | RecursiveCharacterTextSplitter|    |
|   | (PDF, DOCX, TXT) |      | (pypdf, docx, txt)|      | (Chunk size: 600, Overlap: 100)|   |
|   +------------------+      +-------------------+      +---------------+---------------+    |
|                                                                        |                    |
|                                                                        v                    |
|   +--------------------+      +----------------------------------------+                    |
|   | FAISS Vector Index | <--- | SentenceTransformer (all-MiniLM-L6-v2) |                    |
|   | (IndexFlatIP)      |      | (Dense 384-d L2-Normalized Vectors)    |                    |
|   +--------------------+      +----------------------------------------+                    |
+---------------------------------------------------------------------------------------------+

                                               |
                                               v

+---------------------------------------------------------------------------------------------+
|                                RETRIEVAL & GENERATION PIPELINE                              |
|                                                                                             |
|   +--------------------+      +----------------------------------------+                    |
|   | User Query         | ---> | Embed Query                            |                    |
|   | (via API / UI)     |      | (all-MiniLM-L6-v2)                     |                    |
|   +--------------------+      +-------------------+--------------------+                    |
|                                                   |                                         |
|                                                   v                                         |
|                               +----------------------------------------+                    |
|                               | FAISS Similarity Search (Top-k Chunks) |                    |
|                               | (Cosine Similarity Scoring)            |                    |
|                               +-------------------+--------------------+                    |
|                                                   |                                         |
|                                                   v                                         |
|   +--------------------+      +----------------------------------------+                    |
|   | Grounded Prompt    | <--- | Context Aggregator & Relevance Scorer  |                    |
|   | Engineering        |      | (Filter & Format Metadata/Citations)   |                    |
|   +---------+----------+      +----------------------------------------+                    |
|             |                                                                               |
|             v                                                                               |
|   +---------------------------------------+                                                 |
|   | Groq LLM (llama-3.3-70b-versatile)   |                                                 |
|   | Fast API Inference                    |                                                 |
|   +-----------------+---------------------+                                                 |
|                     |                                                                       |
|                     v                                                                       |
|   +---------------------------------------+      +--------------------------------------+   |
|   | Final Answer + Cited Source Chunks    | ---> | Evaluation Hooks                     |   |
|   | (Streamlit UI / FastAPI Response)     |      | (Relevance Cosine, Semantic Match)   |   |
|   +---------------------------------------+      +--------------------------------------+   |
+---------------------------------------------------------------------------------------------+
```

---

## 3. Directory & Folder Structure

```
rag-document-assistant/
├── .env.example                # Sample environment template (GROQ_API_KEY)
├── .env                        # Local secret configurations (ignored by git)
├── .gitignore                  # Git exclusions (data, vectorstore, env, cache)
├── requirements.txt            # Pinned production dependencies
├── README.md                   # Installation, setup, API reference & resume guide
├── architecture.md             # This architecture documentation
├── test_rag.py                 # Standalone end-to-end sanity check and benchmark script
│
├── data/                       # Local document storage
│   └── samples/                # Sample PDF, TXT, DOCX files for testing
│
├── vectorstore/                # FAISS vector storage artifacts
│   ├── index.bin               # FAISS binary index
│   └── metadata.json           # Chunk metadata (text, source, page, chunk_id)
│
└── app/
    ├── __init__.py
    ├── config.py               # Centralized configuration (models, chunking, paths)
    │
    ├── ingestion/              # Ingestion Layer
    │   ├── __init__.py
    │   └── loader.py           # Multi-format document parser & text chunker
    │
    ├── retrieval/              # Retrieval Layer
    │   ├── __init__.py
    │   ├── embedder.py         # Local SentenceTransformers embedding engine
    │   └── vectorstore.py      # FAISS vector store with persistence & similarity search
    │
    ├── llm/                    # Generation Layer
    │   ├── __init__.py
    │   ├── prompts.py          # Grounded prompts preventing hallucinations
    │   └── groq_client.py      # Groq API client (llama-3.3-70b-versatile)
    │
    ├── eval/                   # Evaluation Layer
    │   ├── __init__.py
    │   └── evaluate.py         # Retrieval relevance & Answer accuracy metrics
    │
    ├── api/                    # API Layer (FastAPI)
    │   ├── __init__.py
    │   └── main.py             # REST API endpoints (/upload, /ask, /stats, /reset)
    │
    └── ui/                     # Presentation Layer (Streamlit)
        └── app.py              # Interactive chat UI with source inspection & metrics
```

---

## 4. Component Details & Design Decisions

### 4.1 Ingestion (`app/ingestion/loader.py`)
- **Parsers Supported**:
  - `PDF`: `pypdf` extracts text page-by-page, retaining `page_number` in metadata.
  - `DOCX`: `python-docx` extracts paragraphs, preserving section structure.
  - `TXT / MD`: Raw text loading with UTF-8 decoding.
- **Chunking Strategy**:
  - Uses `RecursiveCharacterTextSplitter` with `chunk_size=600` characters and `chunk_overlap=100` characters.
  - Separators prioritized by: `["\n\n", "\n", " ", ""]` to maintain natural paragraph and sentence boundaries.
  - Every chunk receives unique metadata: `{"source": filename, "page": page_num, "chunk_id": int, "char_count": int}`.

### 4.2 Local Embeddings (`app/retrieval/embedder.py`)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Reasoning**:
  - Zero external API costs & ultra-low latency (<10ms per chunk on CPU).
  - High retrieval accuracy on generic semantic similarity benchmarks.
  - Unit L2-normalization applied to allow Inner Product (`IndexFlatIP`) in FAISS to mathematically equal Cosine Similarity.

### 4.3 Vector Store (`app/retrieval/vectorstore.py`)
- **Index**: `faiss.IndexFlatIP` (Exact Cosine Similarity via normalized dot product).
- **Persistence**:
  - Vector index written to `vectorstore/index.bin`.
  - Chunk texts and metadata written to `vectorstore/metadata.json`.
- **Search**:
  - Returns top-$k$ chunks with similarity scores $s \in [-1.0, 1.0]$ where values closer to $1.0$ indicate high semantic relevance.

### 4.4 LLM Generation (`app/llm/groq_client.py`)
- **Model**: `llama-3.3-70b-versatile` hosted on Groq's LPU hardware for sub-second response times.
- **Prompt Guardrails**:
  - Strict grounding: The model is instructed to answer **only** based on the provided context.
  - Refusal fallback: If the retrieved chunks lack sufficient evidence, it answers: *"I cannot find sufficient information in the provided documents to answer this question."*
  - Citation tags: Instructs the model to append `[Source: filename, Page: X]` where relevant.

---

## 5. Evaluation Hooks (`app/eval/evaluate.py`)

The evaluation framework evaluates both stages of the RAG pipeline:

```
                  +----------------------------------------------+
                  |              Evaluation Metrics              |
                  +----------------------------------------------+
                               /                \
                              /                  \
                             v                    v
              +----------------------+   +-----------------------+
              | Retrieval Relevance  |   |    Answer Accuracy    |
              +----------------------+   +-----------------------+
              | - Cosine Similarity  |   | - Semantic Similarity |
              | - Top-k Relevance    |   | - Token F1 Score      |
              | - Context Hit Rate   |   | - Exact Match Ratio   |
              +----------------------+   +-----------------------+
```

1. **Retrieval Relevance**:
   - **Query-Context Cosine Similarity**: Measures the semantic distance between the user query embedding and the retrieved chunk embeddings.
   - **Retrieval Hit Score**: Checks if the retrieved chunks contain key domain terms or reference answers.
2. **Answer Accuracy**:
   - **Semantic Answer Similarity**: Computes embedding cosine distance between generated response and expected reference answer.
   - **Token F1 & Precision**: Computes token overlap to measure factuality.
