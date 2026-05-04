from app.core.ollama import ollama_client

async def embed(text: str):
    return await ollama_client.embed(text)