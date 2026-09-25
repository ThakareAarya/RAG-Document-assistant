"""Retrieval package for local embeddings and FAISS vector storage."""
from app.retrieval.embedder import Embedder, get_embedder
from app.retrieval.vectorstore import FAISSVectorStore, get_vector_store

__all__ = ["Embedder", "get_embedder", "FAISSVectorStore", "get_vector_store"]
