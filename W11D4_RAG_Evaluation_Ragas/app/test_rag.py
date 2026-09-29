from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent


def test_knowledge_file_exists():
    file_path = BASE_DIR / "data" / "knowledge.txt"
    assert file_path.exists()


def test_qa_pairs():
    file_path = BASE_DIR / "data" / "qa_pairs.json"

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert len(data) == 10


def test_generated_dataset():
    file_path = BASE_DIR / "data" / "rag_dataset.json"

    assert file_path.exists()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert len(data) == 10

    for item in data:
        assert "question" in item
        assert "answer" in item
        assert "contexts" in item
        assert "ground_truth" in item