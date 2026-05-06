from app.core.ollama import ollama_client

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
    """Rerank documents by relevance to the question using LLM-based scoring.

    Prompts the LLM to order document indices by relevance. Falls back to
    the original ordering if the model output cannot be parsed.

    Args:
        question: The user query to rank documents against.
        docs: List of document text strings to be reranked.

    Returns:
        A reordered list of document strings, most relevant first.
    """
    docs_str = "\n".join([f"{i}. {d}" for i, d in enumerate(docs)])

    prompt = f"""
        Rank the following documents by relevance to the question.
        
        Return ONLY a list of numbers.
        
        Question:
        {question}
        
        Documents:
        {docs_str}
        """

    res = await ollama_client.generate(prompt, model="mistral")

    try:
        order = [int(x) for x in res.strip().split() if x.isdigit()]
        return [docs[i] for i in order if i < len(docs)]
    except:
        return docs