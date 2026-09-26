import json
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_FILE = BASE_DIR / "data" / "rag_dataset.json"
RESULTS_FILE = BASE_DIR / "data" / "ragas_results.json"


def load_dataset():
    """Load generated RAG results into a Ragas evaluation dataset."""

    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    samples = []

    for item in data:
        samples.append(
            {
                "user_input": item["question"],
                "response": item["answer"],
                "retrieved_contexts": item["contexts"],
                "reference": item["ground_truth"],
            }
        )

    return EvaluationDataset.from_list(samples)


def evaluate_rag():
    """Evaluate the RAG pipeline using local Ollama models."""

    print("\nLoading RAG evaluation dataset...")

    dataset = load_dataset()

    # Local Ollama LLM used by Ragas
    ollama_llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    # Local Ollama embedding model
    ollama_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    # Wrap LangChain models for Ragas
    evaluator_llm = LangchainLLMWrapper(ollama_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        ollama_embeddings
    )

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ]

    print("\nStarting Ragas evaluation...")
    print(
        "Using Ollama llama3.2:3b for evaluation "
        "and nomic-embed-text for embeddings."
    )
    print("Running with batch_size=1.\n")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        batch_size=1,
    )

    print("\n===== RAGAS RESULTS =====")
    print(result)

    # Convert Ragas results into JSON-safe values
    scores = {}

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_without_reference",
        "context_recall",
    ]

    for metric in metric_names:
        try:
            value = result[metric]

            if value != value:
                scores[metric] = None
            else:
                scores[metric] = float(value)

        except (KeyError, TypeError, ValueError):
            scores[metric] = None

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(scores, file, indent=4)

    print("\nRagas results saved to:")
    print(RESULTS_FILE)

    return result


if __name__ == "__main__":
    evaluate_rag()