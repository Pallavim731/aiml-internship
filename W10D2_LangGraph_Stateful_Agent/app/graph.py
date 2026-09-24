from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from app.state import AgentState


# -------------------------
# NODE 1: CLASSIFY
# -------------------------
def classify(state: AgentState):
    text = state["user_input"].lower()

    if any(word in text for word in [
        "python", "java", "code", "programming", "computer"
    ]):
        category = "technical"

    elif any(word in text for word in [
        "urgent", "emergency", "danger", "help"
    ]):
        category = "urgent"

    else:
        category = "general"

    return {"category": category}


# -------------------------
# NODE 2: ROUTE
# -------------------------
def route(state: AgentState):
    return {"category": state["category"]}


# -------------------------
# CONDITIONAL ROUTING FUNCTION
# -------------------------
def routing_decision(state: AgentState):
    return state["category"]


# -------------------------
# NODE 3: RESPOND
# -------------------------
def respond(state: AgentState):
    category = state["category"]

    if category == "technical":
        response = (
            "This is a technical question. "
            "You can solve it using programming concepts "
            "and appropriate tools."
        )

    elif category == "urgent":
        response = (
            "This request is marked as urgent. "
            "Please seek immediate help from the appropriate person or service."
        )

    else:
        response = (
            "This is a general question. "
            "Here is a general response to help you."
        )

    return {"response": response}


# -------------------------
# NODE 4: HUMAN REVIEW
# -------------------------
def human_review(state: AgentState):
    approval = interrupt(
        "Human review required. Enter 'yes' to approve the response."
    )

    return {"human_approval": approval}


# -------------------------
# HUMAN DECISION
# -------------------------
def after_human_review(state: AgentState):
    if str(state.get("human_approval", "")).lower() == "yes":
        return "approved"

    return "rejected"


# -------------------------
# CREATE GRAPH
# -------------------------
builder = StateGraph(AgentState)


# Add nodes
builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("respond", respond)
builder.add_node("human_review", human_review)


# START → classify
builder.add_edge(START, "classify")


# classify → route
builder.add_edge("classify", "route")


# Conditional routing
builder.add_conditional_edges(
    "route",
    routing_decision,
    {
        "technical": "respond",
        "general": "respond",
        "urgent": "respond"
    }
)


# respond → human review
builder.add_edge("respond", "human_review")


# human review → END
builder.add_conditional_edges(
    "human_review",
    after_human_review,
    {
        "approved": END,
        "rejected": END
    }
)


# Memory/checkpoint
memory = MemorySaver()


# Compile graph
graph = builder.compile(checkpointer=memory)