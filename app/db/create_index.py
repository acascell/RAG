import asyncio
from opensearchpy.exceptions import ConnectionError as OSConnectionError
from app.db.opensearch import client
from app.core.config import settings


async def create_index():
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