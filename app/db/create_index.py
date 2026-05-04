from app.db.opensearch import client
from app.core.config import settings


async def create_index():
    exists = await client.indices.exists(settings.INDEX_NAME)
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
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 768
                    },
                    "metadata": {"type": "object"}
                }
            },
        },
    )