import asyncio
from app.core.ollama import ollama_client
from app.ingestion.chunking import chunk_text
from app.db.opensearch import client
from app.core.config import settings

BATCH_SIZE = 8


async def embed_batch(chunks: list[str]):
    tasks = [ollama_client.embed(c) for c in chunks]
    return await asyncio.gather(*tasks)


async def bulk_index(docs):
    body = []

    for doc in docs:
        body.append({"index": {"_index": settings.INDEX_NAME}})
        body.append(doc)

    await client.bulk(body=body)


async def ingest(text: str, doc_id: str):
    chunks = chunk_text(text)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        embeddings = await embed_batch(batch)

        docs = [
            {
                "text": chunk,
                "embedding": emb,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk": i + j
                }
            }
            for j, (chunk, emb) in enumerate(zip(batch, embeddings))
        ]

        await bulk_index(docs)