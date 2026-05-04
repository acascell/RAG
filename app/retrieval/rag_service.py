from app.core.config import settings
from app.core.ollama import ollama_client
from app.ingestion.embeddings import embed
from app.retrieval.search import search
from app.retrieval.prompt import build_prompt

class RagService:
    @staticmethod
    async def ask(question: str, model: str = settings.OLLAMA_GENERATION_MODEL):
        query_vector = await embed(question)
        contexts = await search(query_vector)
        prompt = build_prompt(question, contexts)
        answer = await ollama_client.generate(prompt, model=model)

        return {
            "answer": answer,
            "contexts": contexts
        }

rag = RagService()