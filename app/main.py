from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.db.create_index import create_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler that initializes resources on startup.

    Creates the OpenSearch index (if it doesn't exist) before the application
    starts accepting requests.

    Args:
        app: The FastAPI application instance.
    """
    await create_index()
    yield


app = FastAPI(title="RAG System", lifespan=lifespan)
app.include_router(router)