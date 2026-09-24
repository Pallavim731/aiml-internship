"""
LangGraph workflow for the Stateful Customer Support Agent.
"""

from langgraph.graph import StateGraph, START, END

from .state import CustomerSupportState
from .crew import run_support_crew


def detect_issue_type(message):
    """Identify a simple issue category from the customer message."""

    message_lower = message.lower()

    if any(word in message_lower for word in ["refund", "money back", "return"]):
        return "refund"

    if any(
        word in message_lower
        for word in ["order", "delivery", "delivered", "shipping"]
    ):
        return "order"

    if any(
        word in message_lower
        for word in ["payment", "paid", "charge", "charged"]
    ):
        return "payment"

    if any(
        word in message_lower
        for word in ["password", "login", "account"]
    ):
        return "account"

    return "general"


def support_node(state: CustomerSupportState):
    """Process the current customer message."""

    current_message = state["current_message"]

    conversation = state.get("conversation", [])

    conversation_history = "\n".join(conversation)

    response = run_support_crew(
        customer_message=current_message,
        conversation_history=conversation_history,
    )

    updated_conversation = list(conversation)

    updated_conversation.append(
        f"Customer: {current_message}"
    )

    updated_conversation.append(
        f"Agent: {response}"
    )

    issue_type = detect_issue_type(current_message)

    return {
        "conversation": updated_conversation,
        "response": response,
        "issue_type": issue_type,
        "resolved": False,
    }


def build_workflow():
    """Build and compile the LangGraph workflow."""

    graph = StateGraph(CustomerSupportState)

    graph.add_node("support", support_node)

    graph.add_edge(START, "support")
    graph.add_edge("support", END)

    return graph.compile()