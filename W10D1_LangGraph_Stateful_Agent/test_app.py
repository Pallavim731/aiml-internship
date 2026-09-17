from app import graph


def test_question_routing():
    config = {
        "configurable": {
            "thread_id": "test-question"
        }
    }

    result = graph.invoke(
        {
            "user_input": "What is LangGraph?",
            "classification": "",
            "response": "",
            "human_feedback": ""
        },
        config=config
    )

    # Graph pauses at human review
    assert result["classification"] == "question"


def test_task_routing():
    config = {
        "configurable": {
            "thread_id": "test-task"
        }
    }

    result = graph.invoke(
        {
            "user_input": "Create a research report",
            "classification": "",
            "response": "",
            "human_feedback": ""
        },
        config=config
    )

    assert result["classification"] == "task"


def test_complaint_routing():
    config = {
        "configurable": {
            "thread_id": "test-complaint"
        }
    }

    result = graph.invoke(
        {
            "user_input": "My application is not working",
            "classification": "",
            "response": "",
            "human_feedback": ""
        },
        config=config
    )

    assert result["classification"] == "complaint"


def test_general_routing():
    config = {
        "configurable": {
            "thread_id": "test-general"
        }
    }

    result = graph.invoke(
        {
            "user_input": "Hello, good morning",
            "classification": "",
            "response": "",
            "human_feedback": ""
        },
        config=config
    )

    assert result["classification"] == "general"


def test_question_response():
    config = {
        "configurable": {
            "thread_id": "test-response"
        }
    }

    result = graph.invoke(
        {
            "user_input": "How can I learn Python?",
            "classification": "",
            "response": "",
            "human_feedback": ""
        },
        config=config
    )

    assert "Python" in result["response"]