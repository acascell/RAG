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


async def run_query(question: str, session_id: str = "test") -> str:
    responses = []
    async for chunk in rag.ask(session_id=session_id, question=question):
        try:
            data = json.loads(chunk)
            resp = data.get("response", "")
            responses.append(resp)
        except Exception as e:
            print(f"[DEBUG] Failed to parse chunk: {chunk}, error: {e}")
    return "".join(responses)


@pytest.mark.asyncio
async def test_rag_pipeline():
    # Load evaluation dataset
    with open("tests/fixtures/dataset.json") as f:
        raw_dataset = json.load(f)

    eval_data = []

    for idx, item in enumerate(raw_dataset):
        question = item.get("question") or item.get("user_input")
        reference = item.get("ground_truth") or item.get("reference")
        contexts = item.get("contexts") or item.get("retrieved_contexts", [])

        # Use a unique session ID for each question to avoid state leakage
        session_id = f"test_question_{idx}"
        answer = await run_query(question, session_id=session_id)

        print("############# EVAL DEBUG")
        print(f"question: {question}")
        print(f"reference: {reference}")
        print(f"context: {contexts}")
        print(f"answer: {answer}")
        print("###### END DEBUG")

        eval_data.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
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
        model=settings.OLLAMA_GENERATION_MODEL,
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