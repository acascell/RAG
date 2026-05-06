import json
import pytest
from openai import OpenAI

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from app.service.rag_service import rag
from app.core.config import settings


async def run_query_with_context(question: str):
    """
    Execute RAG query and return:
    - final answer
    - retrieved contexts (actual, from pipeline)
    """

    chunks = []
    retrieved_contexts = []

    async for event in rag.ask_with_trace(
        session_id="test",
        question=question,
    ):
        # pipeline contract
        if event["type"] == "context":
            retrieved_contexts.append(event["content"])

        elif event["type"] == "chunk":
            chunks.append(event["content"])

    answer = "".join(chunks)

    return answer, retrieved_contexts


@pytest.mark.asyncio
async def test_rag_pipeline():
    # Load evaluation dataset
    with open("tests/fixtures/dataset.json") as f:
        raw_dataset = json.load(f)

    eval_data = []

    for item in raw_dataset:
        question = item.get("question") or item.get("user_input")
        reference = item.get("ground_truth") or item.get("reference")

        answer, retrieved_contexts = await run_query_with_context(question)

        eval_data.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
            }
        )

    eval_dataset = EvaluationDataset.from_list(eval_data)

    # Ollama client via OpenAI-compatible API
    client = OpenAI(
        base_url=f"{settings.OLLAMA_URL}/v1",
        api_key="ollama",
    )

    llm = llm_factory(
        model=settings.OLLAMA_EVAL_MODEL,
        client=client,
    )

    embeddings = embedding_factory(
        provider="openai",
        model=settings.OLLAMA_EMBEDDING_MODEL,
        client=client,
    )

    evaluation_results = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n--- Evaluation Scores ---")
    print(evaluation_results)