from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.db.create_index import create_index, create_memory_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_memory_index()
    await create_index()
    yield


app = FastAPI(title="RAG System", lifespan=lifespan)
app.include_router(router)