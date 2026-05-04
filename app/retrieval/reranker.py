from app.core.ollama import ollama_client
import json

RERANK_PROMPT = """
You are a ranking system.

Given a question and a list of documents, rank them by relevance.

Return ONLY JSON array of indexes in best-first order.

Question:
{question}

Documents:
{docs}
"""

async def rerank(question: str, docs: list[str]):
    formatted_docs = "\n".join(
        [f"[{i}] {doc}" for i, doc in enumerate(docs)]
    )

    prompt = RERANK_PROMPT.format(
        question=question,
        docs=formatted_docs
    )

    response = await ollama_client.generate(
        prompt,
        model="mistral-small"
    )

    try:
        order = json.loads(response)
        return [docs[i] for i in order if i < len(docs)]
    except:
        # fallback if model fails formatting
        return docs