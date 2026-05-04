import httpx
from app.core.config import settings

class OllamaClient:
    @staticmethod
    async def embed(text: str):
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_URL}/api/embed",
                json={
                    "model": settings.OLLAMA_EMBEDDING_MODEL,
                    "input": text
                }
            )
            response.raise_for_status()
            data = response.json()
            if "embeddings" not in data:
                raise RuntimeError(f"Ollama embedding response missing 'embeddings' key: {data}")
            return data["embeddings"][0]

    @staticmethod
    async def generate(prompt: str, model: str = settings.OLLAMA_GENERATION_MODEL):
        async with httpx.AsyncClient(timeout=120.0) as client:
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

