from app.ingestion.chunking import chunk_text
from app.ingestion.embeddings import embed
from app.db.opensearch import client
from app.core.config import settings

async def ingest(text: str, doc_id: str):
    """perform ingestion in chunks and store each chunk as a documen t in the opensearch index"""

    chunks = chunk_text(text, settings.CHUNK_SIZE)
    for i, chunk in enumerate(chunks):
        vector = await embed(chunk)

        await client.index(
            index=settings.INDEX_NAME,
            id=f"{doc_id}_{i}",
            body={
                "text": chunk,
                "embedding": vector,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_id": i
                }
            }
        )