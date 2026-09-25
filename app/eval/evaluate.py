"""Evaluation module for RAG retrieval relevance and answer accuracy scoring."""
import re
import string
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from app.ingestion.loader import Document
from app.retrieval.embedder import get_embedder, Embedder
from app.retrieval.vectorstore import FAISSVectorStore, get_vector_store
from app.llm.groq_client import GroqRAGClient, get_groq_client


def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return " ".join(tokens)


def compute_token_f1(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Compute token-level Precision, Recall, and F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        exact_match = 1.0 if pred_tokens == gt_tokens else 0.0
        return {"precision": exact_match, "recall": exact_match, "f1": exact_match}

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Count matching tokens with frequency
    match_count = sum(min(pred_tokens.count(tok), gt_tokens.count(tok)) for tok in common)
    precision = match_count / len(pred_tokens)
    recall = match_count / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


class RAGEvaluator:
    """Evaluates both Retrieval Relevance and Answer Generation Accuracy."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        llm_client: Optional[GroqRAGClient] = None
    ):
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store or get_vector_store()
        self.llm_client = llm_client or get_groq_client()

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Document, float]],
        expected_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Score the relevance of retrieved chunks for a given query.
        
        Metrics:
            - mean_cosine_similarity: Average score of retrieved chunks
            - max_cosine_similarity: Best score among retrieved chunks (Top-1)
            - chunk_scores: Individual chunk scores and metadata
            - keyword_hit_rate: Proportion of expected keywords found in retrieved text
        """
        if not retrieved_chunks:
            return {
                "mean_cosine_similarity": 0.0,
                "max_cosine_similarity": 0.0,
                "chunk_scores": [],
                "keyword_hit_rate": 0.0,
                "num_chunks_retrieved": 0
            }

        scores = [score for _, score in retrieved_chunks]
        mean_score = float(np.mean(scores))
        max_score = float(np.max(scores))

        # Check for keyword hits in context
        all_context = " ".join([doc.page_content.lower() for doc, _ in retrieved_chunks])
        keyword_hit_rate = 1.0
        if expected_keywords:
            hits = sum(1 for kw in expected_keywords if kw.lower() in all_context)
            keyword_hit_rate = round(hits / len(expected_keywords), 4)

        chunk_details = [
            {
                "chunk_id": doc.metadata.get("chunk_id", i),
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 1),
                "score": round(score, 4),
                "snippet": doc.page_content[:120] + "..."
            }
            for i, (doc, score) in enumerate(retrieved_chunks)
        ]

        return {
            "mean_cosine_similarity": round(mean_score, 4),
            "max_cosine_similarity": round(max_score, 4),
            "keyword_hit_rate": keyword_hit_rate,
            "num_chunks_retrieved": len(retrieved_chunks),
            "chunk_scores": chunk_details
        }

    def evaluate_answer(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> Dict[str, Any]:
        """Score answer accuracy against a ground-truth reference answer.
        
        Metrics:
            - semantic_similarity: Cosine similarity between embedding vectors of generated & reference
            - token_f1: Token overlap F1 score
            - token_precision: Token overlap Precision
            - token_recall: Token overlap Recall
            - exact_match: Boolean flag indicating exact normalized string equality
        """
        if not generated_answer or not reference_answer:
            return {
                "semantic_similarity": 0.0,
                "token_f1": 0.0,
                "token_precision": 0.0,
                "token_recall": 0.0,
                "exact_match": False
            }

        # 1. Semantic Similarity via SentenceTransformer
        emb_gen = self.embedder.embed_query(generated_answer)
        emb_ref = self.embedder.embed_query(reference_answer)
        # Vectors are L2 normalized, so inner product = cosine similarity
        semantic_sim = float(np.dot(emb_gen, emb_ref))
        semantic_sim = max(-1.0, min(1.0, semantic_sim))

        # 2. Token Overlap Metrics
        f1_metrics = compute_token_f1(generated_answer, reference_answer)

        # 3. Exact Match
        exact_match = normalize_text(generated_answer) == normalize_text(reference_answer)

        return {
            "semantic_similarity": round(semantic_sim, 4),
            "token_f1": f1_metrics["f1"],
            "token_precision": f1_metrics["precision"],
            "token_recall": f1_metrics["recall"],
            "exact_match": exact_match
        }

    def evaluate_query(
        self,
        query: str,
        reference_answer: Optional[str] = None,
        expected_keywords: Optional[List[str]] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """Run full end-to-end evaluation for a single query."""
        # 1. Retrieval
        retrieved_chunks = self.vector_store.similarity_search_with_score(query, k=top_k)
        retrieval_eval = self.evaluate_retrieval(query, retrieved_chunks, expected_keywords)

        # 2. Generation
        llm_response = self.llm_client.generate_answer(query, retrieved_chunks)
        generated_answer = llm_response.get("answer", "")

        # 3. Answer Accuracy Evaluation (if reference provided)
        answer_eval = {}
        if reference_answer:
            answer_eval = self.evaluate_answer(generated_answer, reference_answer)

        return {
            "query": query,
            "generated_answer": generated_answer,
            "reference_answer": reference_answer,
            "latency_seconds": llm_response.get("latency_seconds", 0.0),
            "retrieval_metrics": retrieval_eval,
            "answer_metrics": answer_eval,
            "mock_mode": llm_response.get("mock_mode", False)
        }


def run_evaluation_suite(
    test_cases: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 4
) -> Dict[str, Any]:
    """Execute a benchmark evaluation on a batch of test Q&A items."""
    evaluator = RAGEvaluator()

    # Default synthetic benchmark test suite
    if not test_cases:
        test_cases = [
            {
                "query": "What is the primary role of an embedding model in RAG?",
                "reference_answer": "An embedding model converts text chunks and user queries into dense numerical vectors to enable semantic similarity search.",
                "expected_keywords": ["dense", "vector", "semantic", "search", "embedding"]
            },
            {
                "query": "Why is FAISS used in this architecture?",
                "reference_answer": "FAISS provides fast, efficient similarity search and indexing for dense vector embeddings.",
                "expected_keywords": ["faiss", "similarity", "search", "index", "vector"]
            },
            {
                "query": "Which Groq LLM model is used for generating responses?",
                "reference_answer": "The application uses Groq's llama-3.3-70b-versatile model for fast and accurate response generation.",
                "expected_keywords": ["llama", "70b", "groq", "versatile"]
            }
        ]

    results = []
    total_mean_relevance = []
    total_semantic_sim = []
    total_f1 = []

    for item in test_cases:
        eval_result = evaluator.evaluate_query(
            query=item["query"],
            reference_answer=item.get("reference_answer"),
            expected_keywords=item.get("expected_keywords"),
            top_k=top_k
        )
        results.append(eval_result)

        total_mean_relevance.append(eval_result["retrieval_metrics"]["mean_cosine_similarity"])
        if "semantic_similarity" in eval_result["answer_metrics"]:
            total_semantic_sim.append(eval_result["answer_metrics"]["semantic_similarity"])
            total_f1.append(eval_result["answer_metrics"]["token_f1"])

    summary = {
        "num_test_cases": len(test_cases),
        "average_retrieval_relevance": round(float(np.mean(total_mean_relevance)), 4) if total_mean_relevance else 0.0,
        "average_answer_semantic_similarity": round(float(np.mean(total_semantic_sim)), 4) if total_semantic_sim else 0.0,
        "average_token_f1": round(float(np.mean(total_f1)), 4) if total_f1 else 0.0,
        "details": results
    }

    return summary


if __name__ == "__main__":
    print("[Evaluation] Running evaluation test suite...")
    summary = run_evaluation_suite()
    print(f"Test cases evaluated: {summary['num_test_cases']}")
    print(f"Average Retrieval Relevance: {summary['average_retrieval_relevance']}")
    print(f"Average Answer Semantic Sim: {summary['average_answer_semantic_similarity']}")
    print(f"Average Token F1: {summary['average_token_f1']}")
