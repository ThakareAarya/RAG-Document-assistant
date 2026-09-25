"""End-to-end sanity check and evaluation script for the RAG Document Assistant."""
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.config import settings
from app.ingestion.loader import DocumentLoader, chunk_documents
from app.retrieval.embedder import get_embedder
from app.retrieval.vectorstore import get_vector_store
from app.llm.groq_client import get_groq_client
from app.eval.evaluate import RAGEvaluator, run_evaluation_suite


def main():
    print("=" * 70)
    print(">>> STARTING RAG DOCUMENT ASSISTANT END-TO-END SANITY CHECK")
    print("=" * 70)

    # 1. Check Configuration
    print("\n[Step 1] Verifying System Configuration...")
    print(f"  • Base Dir: {settings.BASE_DIR}")
    print(f"  • Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"  • Groq Model: {settings.GROQ_MODEL}")
    print(f"  • Chunk Size: {settings.CHUNK_SIZE}, Overlap: {settings.CHUNK_OVERLAP}")
    print(f"  • Groq API Key Set: {'YES' if settings.GROQ_API_KEY and settings.GROQ_API_KEY != 'your_groq_api_key_here' else 'NO (Demo/Fallback Mode Active)'}")

    # 2. Ingest Sample Document
    sample_file = settings.SAMPLES_DIR / "sample_rag_overview.txt"
    print(f"\n[Step 2] Loading and Parsing Document: {sample_file.name}...")
    if not sample_file.exists():
        print(f"❌ Error: Sample file {sample_file} not found.")
        return

    raw_docs = DocumentLoader.load_file(sample_file)
    print(f"  • Extracted {len(raw_docs)} raw document sections.")

    chunked_docs = chunk_documents(raw_docs)
    print(f"  • Generated {len(chunked_docs)} semantic chunks.")
    for idx, c in enumerate(chunked_docs[:2]):
        print(f"    - Chunk {idx+1} ({c.metadata.get('char_count')} chars): {c.page_content[:80]}...")

    # 3. Vector Embedding & FAISS Indexing
    print("\n[Step 3] Initializing FAISS Vector Store and Indexing Chunks...")
    vector_store = get_vector_store()
    vector_store.clear()  # Fresh index for sanity test
    num_added = vector_store.add_documents(chunked_docs)
    print(f"  • Successfully indexed {num_added} chunks into FAISS.")
    print(f"  • Index Stats: {vector_store.get_stats()}")

    # 4. Retrieval & Similarity Search
    test_queries = [
        "What are the limitations of Large Language Models that RAG solves?",
        "What embedding model and vector dimension is used?",
        "Why is FAISS IndexFlatIP used for search?"
    ]

    print("\n[Step 4] Testing Retrieval with Cosine Similarity Search...")
    for q in test_queries:
        print(f"\n  [Query]: '{q}'")
        results = vector_store.similarity_search_with_score(q, k=2)
        for idx, (doc, score) in enumerate(results, 1):
            print(f"     Rank {idx} (Score: {score:.4f}): {doc.page_content[:90]}...")

    # 5. LLM Response Generation
    print("\n[Step 5] Testing Grounded Answer Generation...")
    llm_client = get_groq_client()
    sample_query = "What embedding model and vector dimension does the pipeline use?"
    top_chunks = vector_store.similarity_search_with_score(sample_query, k=3)
    response = llm_client.generate_answer(sample_query, top_chunks)

    print(f"  • Query: {sample_query}")
    print(f"  • Latency: {response['latency_seconds']}s")
    print(f"  • Model: {response['model']}")
    print(f"  • Answer:\n    {response['answer'].replace(chr(10), chr(10) + '    ')}")

    # 6. Evaluation Suite Execution
    print("\n[Step 6] Running Quantitative Evaluation Suite...")
    eval_test_cases = [
        {
            "query": "What embedding model is used in the pipeline?",
            "reference_answer": "The pipeline uses sentence-transformers/all-MiniLM-L6-v2 which creates 384-dimensional dense vectors.",
            "expected_keywords": ["sentence-transformers", "all-MiniLM-L6-v2", "384"]
        },
        {
            "query": "What are the three main limitations of LLMs mentioned?",
            "reference_answer": "The limitations are knowledge cutoff, hallucinations, and lack of domain specificity.",
            "expected_keywords": ["knowledge cutoff", "hallucinations", "domain specificity"]
        }
    ]

    eval_summary = run_evaluation_suite(test_cases=eval_test_cases, top_k=3)
    print("\n" + "=" * 70)
    print("[EVALUATION BENCHMARK RESULTS]")
    print("=" * 70)
    print(f"  • Test Cases Evaluated: {eval_summary['num_test_cases']}")
    print(f"  • Average Retrieval Relevance (Cosine Sim): {eval_summary['average_retrieval_relevance']:.4f}")
    print(f"  • Average Answer Semantic Similarity:       {eval_summary['average_answer_semantic_similarity']:.4f}")
    print(f"  • Average Token F1 Score:                   {eval_summary['average_token_f1']:.4f}")

    print("\n" + "=" * 70)
    print("[SUCCESS] SANITY CHECK COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
