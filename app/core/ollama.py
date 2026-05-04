import httpx
from app.core.config import settings

class OllamaClient:
    async def embed(self, text: str):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.OLLAMA_URL}/api/embeddings",
                json={
                    "model": settings.OLLAMA_EMBEDDING_MODEL,
                    "prompt": text
                }
            )
            return response.json()["embedding"]

    async def generate(self, prompt: str, model: str = settings.OLLAMA_GENERATION_MODEL):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]

ollama_client = OllamaClient()

