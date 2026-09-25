"""FAISS Vector Store wrapper with persistence and cosine similarity search."""
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss

from app.config import settings
from app.ingestion.loader import Document
from app.retrieval.embedder import get_embedder, Embedder


class FAISSVectorStore:
    """FAISS In-Memory Vector Store with disk persistence."""

    def __init__(
        self,
        embedder: Embedder = None,
        index_path: Path = None,
        metadata_path: Path = None
    ):
        self.embedder = embedder or get_embedder()
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.metadata_path = metadata_path or settings.METADATA_PATH
        
        self.dimension = self.embedder.dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Document] = []
        
        # Load existing index if available on disk
        self.load()

    def add_documents(self, documents: List[Document]) -> int:
        """Embed and add documents to the FAISS index and metadata store.
        
        Args:
            documents: List of chunked Document objects.
            
        Returns:
            Total count of new documents added.
        """
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.embed_texts(texts)

        # Add vectors to FAISS
        self.index.add(embeddings)
        self.documents.extend(documents)

        # Persist to disk
        self.save()
        return len(documents)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = None
    ) -> List[Tuple[Document, float]]:
        """Search the FAISS index for top-k closest documents using Cosine Similarity.
        
        Args:
            query: User search query.
            k: Number of nearest neighbors to retrieve.
            
        Returns:
            List of (Document, cosine_similarity_score) tuples sorted descending by score.
        """
        k = k or settings.TOP_K
        if self.index.ntotal == 0 or not self.documents:
            return []

        # Actual k cannot exceed indexed vectors
        k = min(k, self.index.ntotal)

        # Embed query vector
        query_embedding = self.embedder.embed_query(query).reshape(1, -1)

        # Query FAISS index (IndexFlatIP returns Inner Product = Cosine Similarity)
        scores, indices = self.index.search(query_embedding, k)

        results: List[Tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx]
                results.append((doc, float(score)))

        return results

    def save(self):
        """Persist FAISS index and document metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Write metadata
        docs_data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self.documents
        ]
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """Load FAISS index and metadata from disk if they exist."""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    docs_data = json.load(f)
                self.documents = [
                    Document(page_content=d["page_content"], metadata=d["metadata"])
                    for d in docs_data
                ]
                print(f"[VectorStore] Loaded {self.index.ntotal} vectors from {self.index_path}")
                return True
            except Exception as e:
                print(f"[VectorStore] Error loading index: {e}. Initializing fresh index.")
                self.clear()
                return False
        return False

    def clear(self):
        """Reset the vector store and remove persisted disk files."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        print("[VectorStore] Index cleared.")

    def get_stats(self) -> Dict[str, Any]:
        """Return operational statistics about the vector store."""
        unique_sources = set(doc.metadata.get("source", "unknown") for doc in self.documents)
        return {
            "total_chunks": self.index.ntotal,
            "total_documents": len(unique_sources),
            "sources": list(unique_sources),
            "embedding_dimension": self.dimension,
            "embedding_model": self.embedder.model_name
        }


# Global store instance
_global_vector_store = None

def get_vector_store() -> FAISSVectorStore:
    """Retrieve or initialize the global FAISSVectorStore singleton."""
    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = FAISSVectorStore()
    return _global_vector_store
