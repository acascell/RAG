import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.service.ingest_service import ingest
from app.service.rag_service import rag
from app.models.models import ChatRequest, IngestRequest

router = APIRouter()

@router.post("/ask")
async def ask(req: ChatRequest):
    async def event_stream():
        kwargs = {"session_id": req.session_id, "question": req.question}
        if req.model:
            kwargs["model"] = req.model
        async for chunk in rag.ask(**kwargs):
            data = json.loads(chunk)
            token = data.get("response", "")
            if token:
                yield token

    return StreamingResponse(event_stream(), media_type="text/plain")

@router.post("/ingest")
async def ingest_doc(req: IngestRequest):
    await ingest(req.text, req.doc_id)
    return {"status": "ok"}