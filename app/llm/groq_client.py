"""Groq LLM Client for RAG response generation using llama-3.3-70b-versatile."""
import os
import time
from typing import List, Tuple, Dict, Any, Optional
from groq import Groq

from app.config import settings
from app.ingestion.loader import Document
from app.llm.prompts import RAG_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_context_block


class GroqRAGClient:
    """Wrapper around the Groq API for grounded retrieval-augmented generation."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.custom_api_key = api_key
        self.model = model or settings.GROQ_MODEL
        self.temperature = settings.TEMPERATURE
        self.max_tokens = settings.MAX_TOKENS
        self.client = None
        self.is_configured = False
        self._refresh_client()

    def _refresh_client(self):
        """Dynamically reload .env and refresh client if key was added or updated."""
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=settings.ENV_FILE, override=True)
        
        raw_key = self.custom_api_key or os.getenv("GROQ_API_KEY", "")
        clean_key = raw_key.strip().strip('"').strip("'")
        
        if clean_key and clean_key != "your_groq_api_key_here" and len(clean_key) > 10:
            if not self.is_configured or (self.client and getattr(self, "_active_key", None) != clean_key):
                try:
                    self.client = Groq(api_key=clean_key)
                    self._active_key = clean_key
                    self.is_configured = True
                    print("[GroqClient] Successfully connected to Groq API!")
                except Exception as e:
                    print(f"[GroqClient] Error initializing Groq client: {e}")
                    self.is_configured = False
        else:
            self.is_configured = False

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Document, float]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a grounded answer for the user query given the retrieved context chunks."""
        self._refresh_client()
        start_time = time.time()
        sys_prompt = system_prompt or RAG_SYSTEM_PROMPT

        # Format context
        if retrieved_chunks:
            context_str = format_context_block(retrieved_chunks)
        else:
            context_str = "No relevant context found in uploaded documents."

        user_content = USER_PROMPT_TEMPLATE.format(
            context_str=context_str,
            query=query
        )

        # Fallback / Mock mode when no API key is provided
        if not self.is_configured or not self.client:
            latency = time.time() - start_time
            mock_answer = self._generate_fallback_answer(query, retrieved_chunks)
            return {
                "answer": mock_answer,
                "model": f"{self.model} (Demo Mode - API Key not set)",
                "latency_seconds": round(latency, 3),
                "context_chunks_used": len(retrieved_chunks),
                "mock_mode": True
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1.0,
                stream=False
            )
            answer_text = response.choices[0].message.content
            latency = time.time() - start_time

            return {
                "answer": answer_text,
                "model": self.model,
                "latency_seconds": round(latency, 3),
                "context_chunks_used": len(retrieved_chunks),
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else {},
                "mock_mode": False
            }

        except Exception as e:
            # Auto-fallback to available models if model name is deprecated or tier-restricted
            if "model_not_found" in str(e) or "404" in str(e):
                try:
                    available = [
                        m.id for m in self.client.models.list().data 
                        if "guard" not in m.id and "whisper" not in m.id
                    ]
                    if available:
                        fallback_model = available[0]
                        print(f"[GroqClient] Model '{self.model}' not found on endpoint. Auto-falling back to available model: '{fallback_model}'")
                        self.model = fallback_model
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_content}
                            ],
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            top_p=1.0,
                            stream=False
                        )
                        answer_text = response.choices[0].message.content
                        latency = time.time() - start_time
                        return {
                            "answer": answer_text,
                            "model": self.model,
                            "latency_seconds": round(latency, 3),
                            "context_chunks_used": len(retrieved_chunks),
                            "token_usage": {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens
                            } if response.usage else {},
                            "mock_mode": False
                        }
                except Exception as fallback_err:
                    print(f"[GroqClient] Fallback error: {fallback_err}")

            latency = time.time() - start_time
            error_msg = f"Error during Groq API call: {str(e)}"
            print(f"[GroqClient] {error_msg}")
            
            # Fall back cleanly so app doesn't crash
            fallback = self._generate_fallback_answer(query, retrieved_chunks, error=str(e))
            return {
                "answer": fallback,
                "model": self.model,
                "latency_seconds": round(latency, 3),
                "context_chunks_used": len(retrieved_chunks),
                "error": error_msg,
                "mock_mode": True
            }

    def _generate_fallback_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Document, float]],
        error: Optional[str] = None
    ) -> str:
        """Create a transparent demo response when API key is missing or invalid."""
        if not retrieved_chunks:
            return "No matching context was found in the indexed documents to answer this question."

        top_doc, score = retrieved_chunks[0]
        source = top_doc.metadata.get("source", "document")
        page = top_doc.metadata.get("page", 1)

        prefix = ""
        if error:
            prefix = f"> ⚠️ **Groq API Notice**: {error}\n\n"
        elif not self.is_configured:
            prefix = (
                "> 💡 **Demo Mode Notice**: `GROQ_API_KEY` is not configured in `.env`. "
                "Retrieval is fully working with local FAISS & MiniLM embeddings! "
                "Below is the most relevant retrieved context snippet:\n\n"
            )

        return (
            f"{prefix}"
            f"**Query**: *{query}*\n\n"
            f"**Most Relevant Match** *(Cosine Similarity: {score:.4f} | Source: {source}, Page {page})*:\n"
            f"> \"{top_doc.page_content[:300]}...\"\n\n"
            f"*To get full generative answers using Groq Llama 3.3 70B, please provide your `GROQ_API_KEY` in the `.env` file.*"
        )


_global_groq_client = None

def get_groq_client() -> GroqRAGClient:
    """Singleton getter for GroqRAGClient."""
    global _global_groq_client
    if _global_groq_client is None:
        _global_groq_client = GroqRAGClient()
    return _global_groq_client
