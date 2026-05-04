from app.db.opensearch import client
from app.core.config import settings

async def vector_search(query_vector: str, k=5):
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


async def bm25_search(query: str, k=5):
    res = await client.search(
        index=settings.INDEX_NAME,
        body={
            "size": k,
            "query": {
                "match": {
                    "text": query
                }
            }
        },
    )

    return [
        {
            "text": hit["_source"]["text"],
            "score": hit["_score"]
        }
        for hit in res["hits"]["hits"]
    ]


def fuse_results(vector_results, bm25_results):
    """
    Simple rank fusion (RRF-like but simplified)
    """

    scores = {}

    def add(results, weight):
        for i, r in enumerate(results):
            key = r["text"]

            rank_score = 1 / (i + 1)

            if key not in scores:
                scores[key] = 0

            scores[key] += weight * rank_score

    add(vector_results, 0.7)
    add(bm25_results, 0.3)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [item[0] for item in sorted_items[:5]]