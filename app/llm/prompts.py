"""System prompts and prompt formatting templates for RAG generation."""

RAG_SYSTEM_PROMPT = """You are an expert, precise, and helpful Document Q&A Assistant.
Your mission is to answer the user's question accurately, concisely, and strictly based on the provided context retrieved from documents.

Guidelines:
1. Grounding: Answer ONLY based on the facts directly mentioned in the Context. Do not introduce outside knowledge or speculation.
2. Missing Information: If the provided context does not contain enough information to answer the question faithfully, clearly state: "I cannot find sufficient information in the provided documents to answer this question."
3. Citations: When referencing facts, attribute them to their source using inline references like [Source: <filename>, Page: <page_number>].
4. Clarity & Formatting: Use bullet points, bold text, or numbered lists where appropriate to make the response easy to read.
"""

USER_PROMPT_TEMPLATE = """Context Information:
---------------------
{context_str}
---------------------

Question: {query}

Answer the question strictly using the context above. If the context does not contain the answer, state that the information is unavailable in the documents."""


def format_context_block(retrieved_chunks) -> str:
    """Format a list of (Document, score) or Document into a formatted context block.
    
    Args:
        retrieved_chunks: List of (Document, score) or Document objects.
        
    Returns:
        Structured context string with metadata citations.
    """
    context_parts = []
    for idx, item in enumerate(retrieved_chunks, 1):
        if isinstance(item, tuple):
            doc, score = item
            score_str = f" (Relevance Score: {score:.4f})"
        else:
            doc = item
            score_str = ""
            
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", idx)
        
        block = (
            f"[Document {idx} | Source: {source} | Page: {page} | Chunk: {chunk_id}{score_str}]\n"
            f"{doc.page_content}"
        )
        context_parts.append(block)
        
    return "\n\n".join(context_parts)
