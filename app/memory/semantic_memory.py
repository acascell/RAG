from app.core.ollama import ollama_client
from app.db.opensearch import client
from app.core.config import settings

async def store_memory(session_id: str, facts: list[str]):
    """Store memory facts embedding them using ollama client and then index them
    in the memory index from opensearch"""
    for fact in facts:
        embedding = await ollama_client.embed(fact, model=settings.OLLAMA_EMBEDDING_MODEL)
        doc = {
            "session_id": session_id,
            "text": fact,
            "embedding": embedding,
        }
        await client.index(index=settings.OPEN_SEARCH_MEMORY_INDEX, body=doc)


async def get_memory(session_id: str, query: str, k=3):
    query_emb = await ollama_client.embed(query)

    res = await client.search(
        index=settings.OPEN_SEARCH_MEMORY_INDEX,
        body={
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_emb,
                                    "k": k
                                }
                            }
                        }
                    ],
                    "filter": [
                        {
                            "term": {
                                "session_id": session_id
                            }
                        }
                    ]
                }
            }
        }
    )

    return [hit["_source"]["text"] for hit in res["hits"]["hits"]]