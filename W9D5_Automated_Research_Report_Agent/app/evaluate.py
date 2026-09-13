from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset


def evaluate_report(question, answer, context):
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[context]],
        "ground_truth": [answer],
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
    )

    return result