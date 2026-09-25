"""Local dense embedding generation using SentenceTransformers."""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class Embedder:
    """Wrapper around SentenceTransformer with L2-normalization for cosine similarity."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern to prevent reloading model weights repeatedly."""
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = None):
        if self._initialized:
            return
        self.model_name = model_name or settings.EMBEDDING_MODEL
        print(f"[Embedder] Loading local embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        print(f"[Embedder] Model loaded successfully. Dimension: {self.dimension}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate normalized dense embeddings for a list of text strings.
        
        Args:
            texts: List of text chunks to embed.
            
        Returns:
            np.ndarray of shape (len(texts), dimension) with dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # normalize_embeddings=True ensures dot product equals cosine similarity
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate normalized dense embedding for a single query string."""
        return self.embed_texts([query])[0]


def get_embedder() -> Embedder:
    """Helper function to retrieve the singleton Embedder instance."""
    return Embedder()
