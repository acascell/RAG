from app.db.opensearch import client
from app.core.config import settings


async def vector_search(vector, k=10):
    res = await client.search(
        index=settings.INDEX_NAME,
        body={
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vector,
                        "k": k
                    }
                }
            }
        }
    )

    return [
        {
            "text": h["_source"]["text"],
            "score": h["_score"]
        }
        for h in res["hits"]["hits"]
    ]


async def bm25_search(query, k=10):
    res = await client.search(
        index=settings.INDEX_NAME,
        body={
            "size": k,
            "query": {
                "match": {
                    "text": query
                }
            }
        }
    )

    return [
        {
            "text": h["_source"]["text"],
            "score": h["_score"]
        }
        for h in res["hits"]["hits"]
    ]


def normalize(scores):
    if not scores:
        return scores

    max_s = max(s["score"] for s in scores)
    min_s = min(s["score"] for s in scores)

    for s in scores:
        if max_s - min_s == 0:
            s["norm"] = 1
        else:
            s["norm"] = (s["score"] - min_s) / (max_s - min_s)

    return scores


def fuse_results(vector_hits, bm25_hits):
    vector_hits = normalize(vector_hits)
    bm25_hits = normalize(bm25_hits)

    combined = {}

    for v in vector_hits:
        combined[v["text"]] = 0.7 * v["norm"]

    for b in bm25_hits:
        combined[b["text"]] = combined.get(b["text"], 0) + 0.3 * b["norm"]

    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in sorted_docs[:8]]