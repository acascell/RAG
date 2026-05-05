from app.core.config import settings
from app.core.ollama import ollama_client


async def rewrite_query(question: str, history: list[dict]):
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