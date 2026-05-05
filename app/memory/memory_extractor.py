from app.core.config import settings
from app.core.ollama import ollama_client

async def extract_facts(text: str):
    prompt = f"""
    Extract important long-term facts from the text.
    Ignore generic conversation.
    Return short bullet points.
    
    Text:
    {text}
    """

    response = await ollama_client.generate(prompt, model=settings.OLLAMA_GENERATION_MODEL)
    facts = [
        fact.strip() for fact in response.split('\n') if fact.strip()]
    return facts