import json
from pathlib import Path

from rag_pipeline import answer_question


questions = [
    "What is a digital twin?",
    "What does HVAC stand for?",
    "What is the purpose of a building digital twin?",
    "Why is temperature important in HVAC monitoring?",
    "Why is humidity important in HVAC systems?",
    "What is energy monitoring?",
    "How can AI be integrated with digital twins?",
    "What is Retrieval-Augmented Generation?",
    "What is the purpose of ChromaDB in a RAG system?",
    "What does the retrieval parameter k control?"
]


results = []

for question in questions:
    print(f"\nProcessing: {question}")

    result = answer_question(
        question,
        chunk_size=500,
        k=3
    )

    results.append(result)


output_file = Path("results/qa_dataset.json")

with output_file.open("w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n10 Q&A pairs generated successfully.")
print(f"Saved to: {output_file}")