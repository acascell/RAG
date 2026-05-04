from fastapi import APIRouter
from app.ingestion.ingest_service import ingest
from app.retrieval.rag_service import rag
from app.models.models import AskRequest, IngestRequest

router = APIRouter()

@router.post("/ask")
async def ask(req: AskRequest):
    result = await rag.ask(req.question, model=req.model) if req.model else await rag.ask(req.question)
    return result

@router.post("/ingest")
async def ingest_doc(req: IngestRequest):
    await ingest(req.text, req.doc_id)
    return {"status": "ok"}