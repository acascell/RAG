from fastapi import APIRouter

router = APIRouter()

@router.get("/ask")
async def ask():
    return {"message": "Hello World"}

