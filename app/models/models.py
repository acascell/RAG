from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """Request model for the /ask endpoint.

    Attributes:
        session_id: Unique identifier for the conversation session, used to maintain memory.
        question: The user's question to be answered by the RAG pipeline.
        model: Optional Ollama model name override for generation.
    """
    session_id: str
    question: str
    model: Optional[str] = None


class IngestRequest(BaseModel):
    """Request model for the /ingest endpoint.

    Attributes:
        text: The raw document text to be chunked, embedded, and indexed.
        doc_id: A unique identifier for the document, stored as metadata.
    """
    text: str
    doc_id: str

