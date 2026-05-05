import json

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from app.service.ingest_service import ingest
from app.service.rag_service import rag
from app.models.models import AskRequest, IngestRequest

router = APIRouter()

@router.post("/ask")
async def ask(req: AskRequest):
    result = await rag.ask(req.question, model=req.model) if req.model else await rag.ask(req.question)
    return result

@router.post("/ask-stream")
async def ask_stream(req: AskRequest):
    kwargs = {"question": req.question}
    if req.model:
        kwargs["model"] = req.model

    async def event_stream():
        async for chunk in rag.ask_stream(**kwargs):
            data = json.loads(chunk)
            token = data.get("response", "")
            if token:
                yield token

    return StreamingResponse(event_stream(), media_type="text/plain", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Transfer-Encoding": "chunked"})

@router.post("/ingest")
async def ingest_doc(req: IngestRequest):
    await ingest(req.text, req.doc_id)
    return {"status": "ok"}