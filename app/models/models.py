from pydantic import BaseModel
from typing import Optional

class AskRequest(BaseModel):
    question: str
    model: Optional[str] = None

class IngestRequest(BaseModel):
    text: str
    doc_id: str