from app.core.ollama import ollama_client


async def embed(text: str):
    """Generate a vector embedding for the given text using the configured Ollama model.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector (768 dimensions).
    """
    return await ollama_client.embed(text)