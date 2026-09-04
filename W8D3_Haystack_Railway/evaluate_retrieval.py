from app.pipeline import HaystackRAG


QUESTIONS = [
    {
        "question": "What is artificial intelligence?",
        "expected_document": "document1.pdf",
    },
    {
        "question": "What is machine learning?",
        "expected_document": "document1.pdf",
    },
    {
        "question": "What are the three major cloud service models?",
        "expected_document": "document2.pdf",
    },
    {
        "question": "What is cloud computing?",
        "expected_document": "document2.pdf",
    },
    {
        "question": "What does the CIA triad stand for?",
        "expected_document": "document3.pdf",
    },
    {
        "question": "What are common cybersecurity threats?",
        "expected_document": "document3.pdf",
    },
    {
        "question": "What is the Internet of Things?",
        "expected_document": "document4.pdf",
    },
    {
        "question": "What are examples of IoT applications?",
        "expected_document": "document4.pdf",
    },
    {
        "question": "What is a digital twin?",
        "expected_document": "document5.pdf",
    },
    {
        "question": "How can digital twins improve system efficiency?",
        "expected_document": "document5.pdf",
    },
]


def calculate_precision(documents, expected_document, k=5):
    top_documents = documents[:k]

    relevant = 0

    for document in top_documents:
        source = document.meta.get("source_file", "")

        if source == expected_document:
            relevant += 1

    return relevant / k


def show_results(title, documents, expected_document):
    print(f"\n{title}")

    for i, document in enumerate(documents, start=1):
        source = document.meta.get("source_file", "unknown")

        print(
            f"  {i}. {source} "
            f"| score={document.score}"
        )

    precision = calculate_precision(
        documents,
        expected_document,
        k=5
    )

    print(
        f"Precision@5: {precision:.2f}"
    )

    return precision


def main():

    print("========================================")
    print("HAYSTACK RETRIEVAL EVALUATION")
    print("========================================")

    rag = HaystackRAG()

    bm25_scores = []
    dense_scores = []

    results = []

    for number, item in enumerate(QUESTIONS, start=1):

        question = item["question"]
        expected = item["expected_document"]

        print("\n" + "=" * 70)
        print(f"QUESTION {number}: {question}")
        print(f"Expected document: {expected}")

        # -----------------------------
        # BM25
        # -----------------------------

        bm25_documents = rag.bm25_search(
            question,
            top_k=5
        )

        bm25_precision = show_results(
            "BM25 RESULTS",
            bm25_documents,
            expected
        )

        # -----------------------------
        # Dense
        # -----------------------------

        dense_documents = rag.dense_search(
            question,
            top_k=5
        )

        dense_precision = show_results(
            "DENSE RESULTS",
            dense_documents,
            expected
        )

        bm25_scores.append(bm25_precision)
        dense_scores.append(dense_precision)

        results.append(
            {
                "question": question,
                "expected": expected,
                "bm25_precision": bm25_precision,
                "dense_precision": dense_precision,
            }
        )

    # -----------------------------
    # AVERAGE PRECISION
    # -----------------------------

    average_bm25 = sum(bm25_scores) / len(bm25_scores)
    average_dense = sum(dense_scores) / len(dense_scores)

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"Average BM25 Precision@5 : {average_bm25:.2f}"
    )

    print(
        f"Average Dense Precision@5: {average_dense:.2f}"
    )

    if average_dense > average_bm25:

        print(
            "\nDense retrieval performed better "
            "on this evaluation."
        )

    elif average_bm25 > average_dense:

        print(
            "\nBM25 retrieval performed better "
            "on this evaluation."
        )

    else:

        print(
            "\nBM25 and Dense retrieval produced "
            "the same average precision."
        )


if __name__ == "__main__":
    main()