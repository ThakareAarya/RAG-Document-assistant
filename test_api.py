"""Integration test for FastAPI endpoints (/health, /upload, /ask, /stats, /evaluate)."""
import os
import sys
from pathlib import Path
from starlette.testclient import TestClient

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.api.main import app

client = TestClient(app)

def test_api_endpoints():
    print("=" * 70)
    print(">>> TESTING FASTAPI RAG ENDPOINTS")
    print("=" * 70)

    # 1. Health Check
    print("\n[1] Testing GET /health...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print(f"    Response: {resp.json()}")

    # 2. Upload Document
    sample_file = BASE_DIR / "data" / "samples" / "sample_rag_overview.txt"
    print(f"\n[2] Testing POST /upload with {sample_file.name}...")
    with open(sample_file, "rb") as f:
        resp = client.post(
            "/upload",
            files={"file": (sample_file.name, f, "text/plain")}
        )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    data = resp.json()
    print(f"    Response: {data}")
    print(f"    Chunks Indexed: {data['chunks_created']}")

    # 3. Stats Check
    print("\n[3] Testing GET /stats...")
    resp = client.get("/stats")
    assert resp.status_code == 200, f"Stats failed: {resp.text}"
    print(f"    Response: {resp.json()}")

    # 4. Ask Question
    print("\n[4] Testing POST /ask...")
    payload = {
        "question": "What is Retrieval-Augmented Generation?",
        "top_k": 3
    }
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200, f"Ask failed: {resp.text}"
    ask_data = resp.json()
    print(f"    Question: {ask_data['question']}")
    print(f"    Model Used: {ask_data['model_used']}")
    print(f"    Context Chunks Count: {ask_data['context_chunks_count']}")
    print(f"    Sources Retrieved: {len(ask_data['sources'])}")
    for s in ask_data["sources"]:
        print(f"      - Source: {s['source']} (Page {s['page']}) | Score: {s['similarity_score']}")

    # 5. Evaluate Query
    print("\n[5] Testing POST /evaluate...")
    eval_payload = {
        "question": "What is RAG?",
        "reference_answer": "RAG is an architectural pattern that optimizes LLMs by referencing external knowledge bases.",
        "expected_keywords": ["retrieval", "knowledge", "architecture"]
    }
    resp = client.post("/evaluate", json=eval_payload)
    assert resp.status_code == 200, f"Evaluate failed: {resp.text}"
    eval_data = resp.json()
    print(f"    Retrieval Relevance Score: {eval_data['retrieval_metrics']['mean_cosine_similarity']}")
    print(f"    Answer Semantic Sim: {eval_data['answer_metrics']['semantic_similarity']}")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL FASTAPI ENDPOINTS VERIFIED AND WORKING PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_api_endpoints()
