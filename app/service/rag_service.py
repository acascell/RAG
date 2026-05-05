import asyncio

from app.core.config import settings
from app.core.ollama import ollama_client
from app.ingestion.embeddings import embed
from app.retrieval.hybrid_search import vector_search, bm25_search, fuse_results
from app.retrieval.prompt import build_prompt
from app.retrieval.reranker import rerank


class RagService:
    @staticmethod
    async def ask(question: str, model: str = settings.OLLAMA_GENERATION_MODEL):
        # 1 embed query
        query_vector = await embed(question)

        # 2 retrieve
        vector_hits = await vector_search(query_vector)
        bm25_hits = await bm25_search(question)

        # 3 fuse results
        merged = fuse_results(vector_hits, bm25_hits)

        # 4 rerank
        reranked = await rerank(question, merged)

        # 5 prompt
        prompt = build_prompt(question, reranked)

        # 6 generate
        answer = await ollama_client.generate(prompt, model=model)

        return {
            "answer": answer,
            "contexts": reranked
        }

    @staticmethod
    async def ask_stream(question: str, model: str = settings.OLLAMA_GENERATION_MODEL):
        query_vector = await embed(question)

        v, b = await asyncio.gather(vector_search(query_vector), bm25_search(question))

        merged = fuse_results(v, b)

        # Skip reranker for streaming to reduce time-to-first-token
        prompt = build_prompt(question, merged)

        async for chunk in ollama_client.stream_generate(prompt, model=model):
            yield chunk

rag = RagService()