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

from rag_pipeline import build_vector_store, retrieve_documents


BASE_DIR = Path(__file__).resolve().parent.parent
QA_FILE = BASE_DIR / "data" / "qa_pairs.json"


def generate_dataset(chunk_size, k):
    questions = json.loads(
        QA_FILE.read_text(encoding="utf-8")
    )

    vector_store = build_vector_store(
        chunk_size=chunk_size,
        chunk_overlap=50
    )

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    records = []

    for item in questions:
        question = item["question"]
        ground_truth = item["answer"]

        documents = retrieve_documents(
            vector_store,
            question,
            k=k
        )

        contexts = [
            doc.page_content
            for doc in documents
        ]

        context_text = "\n\n".join(contexts)

        prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context_text}

Question:
{question}

Give a short and direct answer.
"""

        response = llm.invoke(prompt)

        records.append({
            "question": question,
            "answer": response.content,
            "contexts": contexts,
            "ground_truth": ground_truth
        })

    return Dataset.from_list(records)


def evaluate_dataset(dataset):
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm,
        embeddings=embeddings
    )

    return result.to_pandas()


def main():
    print("======================================")
    print("W11D4 RAG OPTIMISATION")
    print("======================================")

    print("\nRunning baseline configuration...")
    print("Chunk size = 500")
    print("Retrieval k = 3")

    baseline_dataset = generate_dataset(
        chunk_size=500,
        k=3
    )

    baseline_results = evaluate_dataset(
        baseline_dataset
    )

    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]

    baseline_scores = {
        metric: float(baseline_results[metric].mean())
        for metric in metrics
    }

    print("\n===== BASELINE SCORES =====")

    for metric, score in baseline_scores.items():
        print(f"{metric}: {score:.4f}")

    lowest_metric = min(
        baseline_scores,
        key=baseline_scores.get
    )

    print(
        f"\nLowest metric: {lowest_metric}"
    )

    print("\nRunning optimised configuration...")
    print("Chunk size = 300")
    print("Retrieval k = 4")

    optimized_dataset = generate_dataset(
        chunk_size=300,
        k=4
    )

    optimized_results = evaluate_dataset(
        optimized_dataset
    )

    optimized_scores = {
        metric: float(optimized_results[metric].mean())
        for metric in metrics
    }

    print("\n===== OPTIMISED SCORES =====")

    for metric, score in optimized_scores.items():
        print(f"{metric}: {score:.4f}")

    print("\n===== COMPARISON =====")

    for metric in metrics:
        change = optimized_scores[metric] - baseline_scores[metric]

        print(
            f"{metric}: "
            f"{baseline_scores[metric]:.4f} -> "
            f"{optimized_scores[metric]:.4f} "
            f"(change: {change:+.4f})"
        )

    output = {
        "baseline": {
            "chunk_size": 500,
            "retrieval_k": 3,
            "scores": baseline_scores
        },
        "optimized": {
            "chunk_size": 300,
            "retrieval_k": 4,
            "scores": optimized_scores
        },
        "lowest_baseline_metric": lowest_metric
    }

    output_file = BASE_DIR / "data" / "optimization_results.json"

    output_file.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8"
    )

    print("\nResults saved to:")
    print(output_file)


if __name__ == "__main__":
    main()