"""Ingestion package for document loading and chunking."""
from app.ingestion.loader import Document, DocumentLoader, chunk_documents

__all__ = ["Document", "DocumentLoader", "chunk_documents"]
