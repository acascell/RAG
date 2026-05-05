import httpx
import asyncio
from app.core.config import settings

class OllamaClient:
    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=settings.OLLAMA_URL,
            timeout=httpx.Timeout(timeout=300.0, connect=10.0),
        )

    async def embed(self, text: str, model: str = settings.OLLAMA_EMBEDDING_MODEL):
        response = await self._client.post(
            "/api/embed",
            json={
                "model": model,
                "input": text
            }
        )
        response.raise_for_status()
        data = response.json()
        if "embeddings" not in data:
            raise RuntimeError(f"Ollama embedding response missing 'embeddings' key: {data}")
        return data["embeddings"][0]

    async def generate(self, prompt: str, model: str = settings.OLLAMA_GENERATION_MODEL, retries: int = 3):
        for attempt in range(retries):
            response = await self._client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            if response.status_code == 200:
                return response.json()["response"]
            if attempt < retries - 1:
                print(f"Ollama generate attempt {attempt + 1} failed ({response.status_code}): {response.text}. Retrying...")
                await asyncio.sleep(2)
        raise RuntimeError(
            f"Ollama generate failed after {retries} attempts ({response.status_code}): {response.text}"
        )

    async def stream_generate(self, prompt: str, model="qwen2.5"):
        async with self._client.stream(
                "POST",
                "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                },
                timeout=None,
        ) as r:
            async for line in r.aiter_lines():
                if line:
                    yield line

ollama_client = OllamaClient()

