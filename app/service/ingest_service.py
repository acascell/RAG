import asyncio
from app.core.ollama import ollama_client
from app.ingestion.chunking import chunk_text
from app.db.opensearch import client
from app.core.config import settings

BATCH_SIZE = 8


async def embed_batch(chunks: list[str]):
    """Generate embeddings for a batch of text chunks in parallel.

    Args:
        chunks: List of text strings to embed.

    Returns:
        A list of embedding vectors (list of floats), one per chunk.
    """
    tasks = [ollama_client.embed(c) for c in chunks]
    return await asyncio.gather(*tasks)


async def bulk_index(docs):
    """Index multiple documents into OpenSearch in a single bulk request.

    Args:
        docs: List of document dicts, each containing 'text', 'embedding', and 'metadata' fields.
    """
    body = []

    for doc in docs:
        body.append({"index": {"_index": settings.INDEX_NAME}})
        body.append(doc)

    await client.bulk(body=body)


async def ingest(text: str, doc_id: str):
    """Ingest a document into the RAG pipeline.

    Splits the text into chunks, generates embeddings in batches, and indexes
    everything into OpenSearch for later retrieval.

    Args:
        text: The raw document text to ingest.
        doc_id: A unique identifier for the document, stored as metadata.
    """
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