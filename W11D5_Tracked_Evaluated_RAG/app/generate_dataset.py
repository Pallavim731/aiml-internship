import json
from pathlib import Path

from app.graph import run_rag


BASE_DIR = Path(__file__).resolve().parent.parent
QA_FILE = BASE_DIR / "data" / "qa_pairs.json"
OUTPUT_FILE = BASE_DIR / "data" / "rag_dataset.json"


def generate_dataset():
    """Run the RAG pipeline for all questions and save the results."""

    with open(QA_FILE, "r", encoding="utf-8") as file:
        qa_pairs = json.load(file)

    dataset = []

    for index, item in enumerate(qa_pairs, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\nProcessing question {index}/10:")
        print(question)

        result = run_rag(question)

        dataset.append(
            {
                "question": question,
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": ground_truth,
            }
        )

        print("Answer generated successfully.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)

    print("\nDataset generation completed.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Total questions: {len(dataset)}")


if __name__ == "__main__":
    generate_dataset()