from app.state import CustomerSupportState
from app.workflow import detect_issue_type


def test_state_structure():
    """Check that the customer support state has the expected fields."""

    state: CustomerSupportState = {
        "customer_id": "CUST001",
        "conversation": [],
        "current_message": "Where is my order?",
        "response": "",
        "issue_type": "",
        "resolved": False,
    }

    assert state["customer_id"] == "CUST001"
    assert isinstance(state["conversation"], list)
    assert state["resolved"] is False


def test_order_issue_detection():
    """Check order issue detection."""

    result = detect_issue_type(
        "My order has not arrived yet."
    )

    assert result == "order"


def test_refund_issue_detection():
    """Check refund issue detection."""

    result = detect_issue_type(
        "I want a refund."
    )

    assert result == "refund"


def test_payment_issue_detection():
    """Check payment issue detection."""

    result = detect_issue_type(
        "I was charged twice."
    )

    assert result == "payment"


def test_account_issue_detection():
    """Check account issue detection."""

    result = detect_issue_type(
        "I cannot login to my account."
    )

    assert result == "account"


def test_general_issue_detection():
    """Check general issue detection."""

    result = detect_issue_type(
        "I need some help."
    )

    assert result == "general"


def test_conversation_state_is_preserved():
    """Check that conversation history can store multiple turns."""

    conversation = [
        "Customer: My order has not arrived.",
        "Agent: I can help you check your order.",
        "Customer: When can I expect it?",
        "Agent: Please provide your order number.",
    ]

    state: CustomerSupportState = {
        "customer_id": "CUST001",
        "conversation": conversation,
        "current_message": "When can I expect it?",
        "response": "Please provide your order number.",
        "issue_type": "order",
        "resolved": False,
    }

    assert len(state["conversation"]) == 4
    assert "My order has not arrived." in state["conversation"][0]
    assert "When can I expect it?" in state["conversation"][2]
    assert state["issue_type"] == "order"