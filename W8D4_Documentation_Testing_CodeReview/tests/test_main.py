import pytest

from app.main import calculate_accuracy, classify_accuracy


def test_calculate_accuracy():
    assert calculate_accuracy(90, 100) == 90.0


def test_calculate_accuracy_partial():
    assert calculate_accuracy(45, 50) == 90.0


def test_calculate_accuracy_invalid_total():
    with pytest.raises(ValueError):
        calculate_accuracy(5, 0)


def test_calculate_accuracy_invalid_correct():
    with pytest.raises(ValueError):
        calculate_accuracy(110, 100)


def test_classify_excellent():
    assert classify_accuracy(95) == "Excellent"


def test_classify_good():
    assert classify_accuracy(80) == "Good"


def test_classify_average():
    assert classify_accuracy(65) == "Average"


def test_classify_needs_improvement():
    assert classify_accuracy(50) == "Needs Improvement"


def test_classify_invalid_accuracy():
    with pytest.raises(ValueError):
        classify_accuracy(101)