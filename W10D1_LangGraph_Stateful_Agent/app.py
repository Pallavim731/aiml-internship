from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# -----------------------------
# 1. Define shared state
# -----------------------------
class AgentState(TypedDict):
    user_input: str
    classification: str
    response: str
    human_feedback: str


# -----------------------------
# 2. CLASSIFY NODE
# -----------------------------
def classify(state: AgentState):
    user_input = state["user_input"].lower()

    if any(word in user_input for word in [
        "error", "problem", "issue", "not working"
    ]):
        classification = "complaint"

    elif any(word in user_input for word in [
        "how", "what", "why", "when", "where", "?"
    ]):
        classification = "question"

    elif any(word in user_input for word in [
        "do", "create", "make", "build", "send"
    ]):
        classification = "task"

    else:
        classification = "general"

    print(f"[CLASSIFY] Input classified as: {classification}")

    return {
        "classification": classification
    }


# -----------------------------
# 3. ROUTE NODE
# -----------------------------
def route(state: AgentState):
    classification = state["classification"]

    print(f"[ROUTE] Routing to: {classification}")

    return state


# -----------------------------
# 4. RESPOND NODE
# -----------------------------
def respond(state: AgentState):
    classification = state["classification"]
    user_input = state["user_input"]

    if classification == "question":
        response = f"Here is an answer to your question: {user_input}"

    elif classification == "task":
        response = f"I can help you complete this task: {user_input}"

    elif classification == "complaint":
        response = f"I understand the problem. Let's troubleshoot it: {user_input}"

    else:
        response = f"Thanks for your message: {user_input}"

    print(f"[RESPOND] {response}")

    return {
        "response": response
    }


# -----------------------------
# 5. HUMAN REVIEW NODE
# -----------------------------
def human_review(state: AgentState):

    print("\n" + "=" * 60)
    print("⏸ HUMAN REVIEW REQUIRED")
    print("=" * 60)

    feedback = interrupt({
        "message": "Please review the generated response.",
        "generated_response": state["response"]
    })

    return {
        "human_feedback": feedback
    }


# -----------------------------
# 6. FINAL RESPONSE NODE
# -----------------------------
def final_response(state: AgentState):

    feedback = state.get("human_feedback", "")

    if feedback:
        final = (
            f"Human feedback: {feedback}\n"
            f"Final response: {state['response']}"
        )
    else:
        final = state["response"]

    print("\n[FINAL RESPONSE]")
    print(final)

    return {
        "response": final
    }


# -----------------------------
# 7. CONDITIONAL ROUTING
# -----------------------------
def choose_route(state: AgentState):

    classification = state["classification"]

    if classification == "question":
        return "respond_question"

    elif classification == "task":
        return "respond_task"

    elif classification == "complaint":
        return "respond_complaint"

    else:
        return "respond_general"


# -----------------------------
# 8. BUILD GRAPH
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("respond", respond)
builder.add_node("human_review", human_review)
builder.add_node("final_response", final_response)

# START → classify
builder.add_edge(START, "classify")

# classify → route
builder.add_edge("classify", "route")

# Conditional routing
builder.add_conditional_edges(
    "route",
    choose_route,
    {
        "respond_question": "respond",
        "respond_task": "respond",
        "respond_complaint": "respond",
        "respond_general": "respond",
    },
)

# respond → human review
builder.add_edge("respond", "human_review")

# human review → final response
builder.add_edge("human_review", "final_response")

# final response → END
builder.add_edge("final_response", END)


# -----------------------------
# 9. CHECKPOINT MEMORY
# -----------------------------
memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory
)


# -----------------------------
# 10. HUMAN-IN-THE-LOOP TEST
# -----------------------------
def test_human_interrupt():

    print("\n\n")
    print("#" * 60)
    print("HUMAN-IN-THE-LOOP TEST")
    print("#" * 60)

    config = {
        "configurable": {
            "thread_id": "w10d1-human-test"
        }
    }

    initial_state = {
        "user_input": "Create a research report for me",
        "classification": "",
        "response": "",
        "human_feedback": ""
    }

    print("\nStarting graph...")

    result = graph.invoke(
        initial_state,
        config=config
    )

    # The graph should pause here
    print("\n⏸ Graph paused for human input.")

    print("\nGenerated response:")
    print(result["response"])

    # Get human input
    human_input = input(
        "\nEnter human feedback/approval: "
    )

    print("\n▶ Resuming graph...")

    final_result = graph.invoke(
        Command(resume=human_input),
        config=config
    )

    print("\n" + "=" * 60)
    print("GRAPH COMPLETED")
    print("=" * 60)

    print(final_result["response"])


# -----------------------------
# 11. MAIN
# -----------------------------
if __name__ == "__main__":
    test_human_interrupt()