"""
Simple ML utility module.

This module demonstrates clean, documented Python code
with input validation and reusable functions.
"""


def calculate_accuracy(correct: int, total: int) -> float:
    """
    Calculate model accuracy as a percentage.

    Args:
        correct: Number of correct predictions.
        total: Total number of predictions.

    Returns:
        Accuracy percentage as a float.

    Raises:
        ValueError: If total is zero/negative or correct is invalid.
    """
    if total <= 0:
        raise ValueError("Total predictions must be greater than zero.")

    if correct < 0 or correct > total:
        raise ValueError(
            "Correct predictions must be between 0 and total."
        )

    return (correct / total) * 100


def classify_accuracy(accuracy: float) -> str:
    """
    Classify an accuracy percentage.

    Args:
        accuracy: Accuracy value between 0 and 100.

    Returns:
        A simple performance category.
    """
    if not 0 <= accuracy <= 100:
        raise ValueError("Accuracy must be between 0 and 100.")

    if accuracy >= 90:
        return "Excellent"

    if accuracy >= 75:
        return "Good"

    if accuracy >= 60:
        return "Average"

    return "Needs Improvement"


if __name__ == "__main__":
    accuracy = calculate_accuracy(92, 100)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Performance: {classify_accuracy(accuracy)}")