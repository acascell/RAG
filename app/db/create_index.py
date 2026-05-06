import asyncio
from opensearchpy.exceptions import ConnectionError as OSConnectionError
from app.db.opensearch import client
from app.core.config import settings


async def create_index():
    """Create the OpenSearch index with KNN vector support if it doesn't already exist.

    Retries the connection up to 10 times (with 2-second delays) to handle
    cases where OpenSearch is still starting up. The index is configured with:
    - A 'text' field for BM25 full-text search.
    - A 'embedding' field (768-dim KNN vector) for semantic search.
    - A 'metadata' object field for document metadata.

    Raises:
        ConnectionError: If OpenSearch is unreachable after all retry attempts.
    """
    for attempt in range(10):
        try:
            exists = await client.indices.exists(index=settings.INDEX_NAME)
            break
        except (OSConnectionError, ConnectionError):
            if attempt == 9:
                raise
            await asyncio.sleep(2)
    else:
        return

    if exists:
        return

    await client.indices.create(
        index=settings.INDEX_NAME,
        body={
            "settings": {
                "index": {
                    "knn": True
                }
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text"
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 768
                    },
                    "metadata": {"type": "object"}
                }
            },
        },
    )