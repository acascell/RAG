from app.core.config import settings
from app.core.ollama import ollama_client


async def rewrite_query(question: str, history: list[dict]):
    """Rewrite the user's question into a standalone query using conversation history.

    Uses the LLM to reformulate ambiguous or context-dependent questions into
    self-contained queries that can be effectively used for retrieval.

    Args:
        question: The original user question.
        history: List of prior conversation turns, each a dict with 'role' and 'content'.

    Returns:
        A rewritten, standalone query string.
    """
    history_text = "\n".join([
        f"{msg['role']}: {msg['content']}" for msg in history])

    prompt = f"""
    You are a query re writer. You get the user question and then rewrite into a standalone query.
    Conversation:
    {history_text}
    
    Question:
    {question}

    Rewritten Query:
    """

    rewritten = await ollama_client.generate(prompt, model=settings.OLLAMA_GENERATION_MODEL)
    return rewritten.strip()