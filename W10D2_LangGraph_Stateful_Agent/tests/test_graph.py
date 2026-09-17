from app.graph import classify, route, routing_decision


def test_technical_python():
    state = {"user_input": "How can I learn Python?"}

    result = classify(state)

    assert result["category"] == "technical"


def test_technical_java():
    state = {"user_input": "How do I write Java code?"}

    result = classify(state)

    assert result["category"] == "technical"


def test_urgent_request():
    state = {"user_input": "This is an emergency, I need help"}

    result = classify(state)

    assert result["category"] == "urgent"


def test_general_ai():
    state = {"user_input": "What is artificial intelligence?"}

    result = classify(state)

    assert result["category"] == "general"


def test_general_smart_building():
    state = {"user_input": "Tell me something about smart buildings"}

    result = classify(state)

    assert result["category"] == "general"


def test_routing():
    state = {"category": "technical"}

    result = route(state)

    assert result["category"] == "technical"

    decision = routing_decision(result)

    assert decision == "technical"