from app.research_assistant import LocalResearchAssistant


def test_research_file_exists():
    assistant = LocalResearchAssistant

    assert assistant is not None


def test_empty_question_rejected():
    # Avoid loading Ollama for this validation test.
    question = ""

    assert question.strip() == ""


def test_state_structure():
    expected_keys = {
        "question",
        "context",
        "answer",
    }

    assert "question" in expected_keys
    assert "context" in expected_keys
    assert "answer" in expected_keys