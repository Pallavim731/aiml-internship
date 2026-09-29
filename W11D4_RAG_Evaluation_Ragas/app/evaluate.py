import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "rag_dataset.json"


def load_dataset():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return Dataset.from_list(data)


def main():
    dataset = load_dataset()

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    print("Starting Ragas evaluation...")
    print("Evaluating 10 questions...\n")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    print("\n===== RAGAS EVALUATION RESULTS =====")
    print(result)

    output_file = BASE_DIR / "data" / "evaluation_results.csv"

    result.to_pandas().to_csv(
        output_file,
        index=False
    )

    print("\nResults saved to:")
    print(output_file)


if __name__ == "__main__":
    main()