from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    session_id: str
    question: str
    model: Optional[str] = None

class IngestRequest(BaseModel):
    text: str
    doc_id: str