from app.db.opensearch import client
from app.core.config import settings


async def vector_search(vector, k=10):
    """Perform KNN vector similarity search against the OpenSearch index.

    Args:
        vector: The query embedding vector (list of floats).
        k: The number of nearest neighbors to retrieve.

    Returns:
        A list of dicts with 'text' and 'score' keys, sorted by relevance.
    """
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
    """Perform BM25 keyword-based search against the OpenSearch index.

    Args:
        query: The text query to match against indexed documents.
        k: The maximum number of results to return.

    Returns:
        A list of dicts with 'text' and 'score' keys, sorted by relevance.
    """
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
    """Min-max normalize scores to the [0, 1] range.

    Adds a 'norm' key to each entry. If all scores are identical,
    normalizes to 1.

    Args:
        scores: A list of dicts, each containing a 'score' key.

    Returns:
        The same list with an added 'norm' key per entry.
    """
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
    """Fuse vector and BM25 search results using weighted score combination.

    Combines results from semantic (vector) and keyword (BM25) search with
    a 70/30 weighting in favor of vector similarity. Deduplicates by text
    content and returns the top 8 results.

    Args:
        vector_hits: Results from vector_search (list of dicts with 'text' and 'score').
        bm25_hits: Results from bm25_search (list of dicts with 'text' and 'score').

    Returns:
        A list of up to 8 text strings, ordered by fused relevance score.
    """
    vector_hits = normalize(vector_hits)
    bm25_hits = normalize(bm25_hits)

    combined = {}

    for v in vector_hits:
        combined[v["text"]] = 0.7 * v["norm"]

    for b in bm25_hits:
        combined[b["text"]] = combined.get(b["text"], 0) + 0.3 * b["norm"]

    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in sorted_docs[:8]]