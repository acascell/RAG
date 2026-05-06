import asyncio

from app.core.config import settings
from app.core.ollama import ollama_client
from app.ingestion.embeddings import embed
from app.retrieval.hybrid_search import vector_search, bm25_search, fuse_results
from app.retrieval.prompt import build_prompt
from app.retrieval.query_rewriter import rewrite_query
from app.retrieval.reranker import rerank
from app.service.memory import memory


class RagService:
    """Core RAG (Retrieval-Augmented Generation) service.

    Orchestrates the full pipeline: query rewriting, embedding, hybrid retrieval,
    result fusion, prompt construction, and streaming LLM generation.
    """

    async def ask(self, session_id: str, question: str, model: str = settings.OLLAMA_GENERATION_MODEL):
        """Run the full RAG pipeline and stream the generated answer.

        Steps:
        1. Retrieve conversation history from memory.
        2. Rewrite the question into a standalone query.
        3. Embed the rewritten query.
        4. Perform parallel vector + BM25 search.
        5. Fuse and rank results.
        6. Build the LLM prompt with context.
        7. Stream the generated response.

        Args:
            session_id: Unique session identifier for conversation memory.
            question: The user's question.
            model: Ollama model name to use for generation.

        Yields:
            JSON-encoded strings containing response tokens.
        """
        # retrieve memory
        history = memory.get(session_id)

        # rewrite query passing history
        rewritten = await rewrite_query(question, history)

        # 1 embed query
        query_vector = await embed(rewritten)

        # 2 retrieve (parallel)
        v, b = await asyncio.gather(vector_search(query_vector), bm25_search(rewritten))

        # 3 fuse results
        merged = fuse_results(v, b)

        # reranked = await rerank(rewritten, merged)

        # 4 prompt
        prompt = build_prompt(rewritten, merged, history)

        # update memory
        memory.add(session_id, "user", question)

        # 5 stream generate
        async for chunk in ollama_client.stream_generate(prompt, model=model):
            yield chunk

rag = RagService()