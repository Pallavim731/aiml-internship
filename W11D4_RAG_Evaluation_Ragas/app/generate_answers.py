import json
from pathlib import Path

from langchain_ollama import ChatOllama

from rag_pipeline import build_vector_store, retrieve_documents


BASE_DIR = Path(__file__).resolve().parent.parent
QA_FILE = BASE_DIR / "data" / "qa_pairs.json"
OUTPUT_FILE = BASE_DIR / "data" / "rag_dataset.json"


def load_questions():
    with open(QA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_answer(llm, question, context):
    prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Give a short and direct answer.
"""

    response = llm.invoke(prompt)
    return response.content


def main():
    questions = load_questions()

    vector_store = build_vector_store(
        chunk_size=500,
        chunk_overlap=50
    )

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    dataset = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        reference_answer = item["answer"]

        documents = retrieve_documents(
            vector_store,
            question,
            k=3
        )

        contexts = [
            document.page_content
            for document in documents
        ]

        context_text = "\n\n".join(contexts)

        answer = generate_answer(
            llm,
            question,
            context_text
        )

        dataset.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": reference_answer
            }
        )

        print(f"Generated {index}/10")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            dataset,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nSaved dataset to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()