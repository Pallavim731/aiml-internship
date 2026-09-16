import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "outputs" / "retrieval_comparison.csv"


def calculate_precision():

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    total = len(rows)

    bm25_correct = sum(
        int(row["BM25 Top-1 Relevant"])
        for row in rows
    )

    dense_correct = sum(
        int(row["Dense Top-1 Relevant"])
        for row in rows
    )

    bm25_precision = (
        bm25_correct / total
    ) * 100

    dense_precision = (
        dense_correct / total
    ) * 100

    print("\nRetrieval Evaluation")
    print("=" * 50)

    print(
        f"Total Questions : {total}"
    )

    print(
        f"BM25 Relevant   : {bm25_correct}/{total}"
    )

    print(
        f"BM25 Precision@1: {bm25_precision:.2f}%"
    )

    print(
        f"Dense Relevant  : {dense_correct}/{total}"
    )

    print(
        f"Dense Precision@1: {dense_precision:.2f}%"
    )

    print("\nComparison")

    if bm25_precision > dense_precision:

        print(
            "BM25 performed better."
        )

    elif dense_precision > bm25_precision:

        print(
            "Dense retrieval performed better."
        )

    else:

        print(
            "Both methods achieved the same Precision@1."
        )


if __name__ == "__main__":

    calculate_precision()