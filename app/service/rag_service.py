import asyncio

from app.core.config import settings
from app.core.ollama import ollama_client

from app.ingestion.embeddings import embed

from app.memory.semantic_memory import (
    get_memory,
    store_memory
)

from app.memory.memory import short_term_memory  # short-term memory
from app.memory.memory_extractor import extract_facts
from app.retrieval.hybrid_search import (
    vector_search,
    bm25_search,
    fuse_results
)

from app.retrieval.prompt import build_prompt
from app.retrieval.query_rewriter import rewrite_query



class RagService:

    async def ask(
        self,
        session_id: str,
        question: str,
        model: str = settings.OLLAMA_GENERATION_MODEL
    ):
        # short-term (chat history)
        history = short_term_memory.get(session_id)

        # Only rewrite if there's conversation history (follow-up questions)
        # This avoids an expensive LLM call for first questions
        if history:
            rewritten = await rewrite_query(question=question, history=history)
        else:
            rewritten = question

        # semantic memory retrieval (embedding call)
        sem_memory = await get_memory(session_id, rewritten)

        # embed rewritten query for vector search
        query_vector = await embed(rewritten)

        # BM25 doesn't need embedding, can run after vector is ready
        # but since opensearch is fast, just run sequentially to avoid ollama contention
        vector_hits = await vector_search(query_vector)
        bm25_hits = await bm25_search(rewritten)

        # fuse results
        merged = fuse_results(vector_hits, bm25_hits)

        # build up context
        context = sem_memory + merged

        prompt = build_prompt(
            question=question,   # keep original user wording
            contexts=context,
            history=history
        )

        # store user messages in short term
        short_term_memory.add(session_id, "user", question)

        # define output
        answer = ""

        async for chunk in ollama_client.stream_generate(prompt, model=model):
            answer += chunk
            yield chunk

        # store result
        short_term_memory.add(session_id, "assistant", answer)

        # NOTE: Long-term memory extraction disabled — Ollama crashes under
        # resource pressure when running an extra generate call after each response.
        # To re-enable, uncomment the block below or call a dedicated /memorize endpoint.
        # try:
        #     facts = await extract_facts(f"{question}\n{answer}")
        #     if facts:
        #         await store_memory(session_id, facts)
        # except Exception as e:
        #     print(f"[Memory extraction error] {type(e).__name__}: {e}")


rag = RagService()