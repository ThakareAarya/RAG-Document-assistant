"""FastAPI REST API for Document Q&A Assistant using RAG."""
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.ingestion.loader import DocumentLoader, chunk_documents
from app.retrieval.vectorstore import get_vector_store
from app.retrieval.embedder import get_embedder
from app.llm.groq_client import get_groq_client
from app.eval.evaluate import RAGEvaluator, run_evaluation_suite


# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A Assistant RAG API",
    description="Production-grade RAG backend with local FAISS vector search and Groq Llama 3.3 70B inference.",
    version="1.0.0"
)

# Enable CORS for Streamlit / external frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eagerly initialize singletons
vector_store = get_vector_store()
llm_client = get_groq_client()
evaluator = RAGEvaluator(vector_store=vector_store, llm_client=llm_client)


# --- Pydantic Schemas ---

class AskRequest(BaseModel):
    question: str = Field(..., description="User question to answer using indexed documents.")
    top_k: int = Field(default=settings.TOP_K, ge=1, le=20, description="Number of context chunks to retrieve.")
    temperature: Optional[float] = Field(default=settings.TEMPERATURE, ge=0.0, le=1.0, description="LLM sampling temperature.")


class SourceChunk(BaseModel):
    source: str
    page: int
    chunk_id: int
    similarity_score: float
    content: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    latency_seconds: float
    model_used: str
    mock_mode: bool
    context_chunks_count: int


class UploadResponse(BaseModel):
    filename: str
    file_type: str
    chunks_created: int
    total_indexed_chunks: int
    message: str


class StatsResponse(BaseModel):
    total_chunks: int
    total_documents: int
    sources: List[str]
    embedding_model: str
    llm_model: str
    groq_api_configured: bool


class EvalRequest(BaseModel):
    question: str
    reference_answer: Optional[str] = None
    expected_keywords: Optional[List[str]] = None
    top_k: Optional[int] = settings.TOP_K


# --- API Endpoints ---

@app.get("/", tags=["General"])
async def root():
    """Root endpoint for status check."""
    return {
        "status": "online",
        "service": "Document Q&A Assistant RAG API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store_ready": vector_store is not None,
        "indexed_chunks": vector_store.index.ntotal,
        "groq_configured": llm_client.is_configured
    }


@app.post("/upload", response_model=UploadResponse, tags=["RAG Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (PDF, DOCX, TXT) into the FAISS vector index.
    
    Workflow:
    1. Save uploaded file to disk.
    2. Extract raw text with metadata.
    3. Split into semantic chunks via RecursiveCharacterTextSplitter.
    4. Generate dense embeddings and add to FAISS index.
    """
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(allowed_extensions)}"
        )

    # Save to uploads directory
    destination_path = settings.UPLOADS_DIR / file.filename
    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Parse and chunk document
    try:
        raw_docs = DocumentLoader.load_file(destination_path)
        if not raw_docs:
            raise HTTPException(status_code=400, detail="Document appears to be empty or unreadable.")

        chunked_docs = chunk_documents(raw_docs)
        chunks_added = vector_store.add_documents(chunked_docs)

        return UploadResponse(
            filename=file.filename,
            file_type=file_ext,
            chunks_created=chunks_added,
            total_indexed_chunks=vector_store.index.ntotal,
            message=f"Successfully indexed {chunks_added} chunks from '{file.filename}'."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/ask", response_model=AskResponse, tags=["RAG Query"])
async def ask_question(request: AskRequest):
    """Query the indexed documents using RAG.
    
    Workflow:
    1. Retrieve top-k relevant document chunks from FAISS.
    2. Format retrieved context with source attribution.
    3. Pass query + context to Groq's Llama 3.3 70B model.
    4. Return synthesized answer and verified source citations.
    """
    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="Vector store is empty. Please upload at least one document first via /upload."
        )

    start_time = time.time()
    query = request.question.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Similarity search
    retrieved_results = vector_store.similarity_search_with_score(query, k=request.top_k)

    # 2. Call Groq LLM
    llm_output = llm_client.generate_answer(query, retrieved_results)
    latency = round(time.time() - start_time, 3)

    # 3. Format source chunks
    source_chunks = [
        SourceChunk(
            source=doc.metadata.get("source", "Unknown"),
            page=doc.metadata.get("page", 1),
            chunk_id=doc.metadata.get("chunk_id", idx),
            similarity_score=round(score, 4),
            content=doc.page_content
        )
        for idx, (doc, score) in enumerate(retrieved_results)
    ]

    return AskResponse(
        question=query,
        answer=llm_output["answer"],
        sources=source_chunks,
        latency_seconds=latency,
        model_used=llm_output.get("model", settings.GROQ_MODEL),
        mock_mode=llm_output.get("mock_mode", False),
        context_chunks_count=len(retrieved_results)
    )


@app.get("/stats", response_model=StatsResponse, tags=["Diagnostics"])
async def get_stats():
    """Retrieve vector store and system metrics."""
    llm_client._refresh_client()
    stats = vector_store.get_stats()
    return StatsResponse(
        total_chunks=stats["total_chunks"],
        total_documents=stats["total_documents"],
        sources=stats["sources"],
        embedding_model=stats["embedding_model"],
        llm_model=settings.GROQ_MODEL,
        groq_api_configured=llm_client.is_configured
    )


@app.post("/evaluate", tags=["Evaluation"])
async def evaluate_single_query(request: EvalRequest):
    """Evaluate retrieval relevance and answer accuracy for a single query."""
    if vector_store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Vector store is empty. Upload documents before evaluation.")

    result = evaluator.evaluate_query(
        query=request.question,
        reference_answer=request.reference_answer,
        expected_keywords=request.expected_keywords,
        top_k=request.top_k
    )
    return result


@app.post("/evaluate/benchmark", tags=["Evaluation"])
async def evaluate_benchmark_suite(top_k: int = Query(default=4, ge=1, le=10)):
    """Run full benchmark evaluation on sample test cases."""
    if vector_store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Vector store is empty. Upload documents before running benchmark.")

    summary = run_evaluation_suite(top_k=top_k)
    return summary


@app.post("/clear", tags=["Maintenance"])
async def clear_index():
    """Clear all indexed vectors and metadata."""
    vector_store.clear()
    return {"status": "cleared", "total_chunks": vector_store.index.ntotal}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True
    )
