from app.db.opensearch import client
from app.core.config import settings

async def search(query_vector: str, k=5):
    response = await client.search(
        index=settings.INDEX_NAME,
        body={
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            },
        },
    )

    return [hit["_source"]["text"] for hit in response["hits"]["hits"]]