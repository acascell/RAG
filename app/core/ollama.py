import httpx
import asyncio
from app.core.config import settings


class OllamaClient:
    """HTTP client for interacting with the Ollama API.

    Provides methods for generating embeddings, synchronous text generation,
    and streaming text generation using models served by Ollama.
    """

    async def embed(self, text: str):
        """Generate a vector embedding for the given text.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            RuntimeError: If the response does not contain an 'embeddings' key.
            httpx.HTTPStatusError: If the HTTP request fails.
        """
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

    async def generate(self, prompt: str, model: str = settings.OLLAMA_GENERATION_MODEL, retries: int = 3):
        """Generate a complete text response (non-streaming) from the LLM.

        Retries on failure with exponential backoff.

        Args:
            prompt: The input prompt to send to the model.
            model: The Ollama model name to use for generation.
            retries: Number of retry attempts before raising an error.

        Returns:
            The generated text response as a string.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(retries):
                response = await client.post(
                    f"{settings.OLLAMA_URL}/api/generate",
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
        """Stream generated text from the LLM token by token.

        Yields raw JSON lines from the Ollama streaming API. Each line contains
        a JSON object with a 'response' field holding the next token.

        Args:
            prompt: The input prompt to send to the model.
            model: The Ollama model name to use for generation.

        Yields:
            JSON-encoded strings, one per generated token.
        """
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                    "POST",
                    f"{settings.OLLAMA_URL}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True,
                    },
            ) as r:

                async for line in r.aiter_lines():
                    if line:
                        yield line

ollama_client = OllamaClient()
