from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# -----------------------------
# 1. Define State
# -----------------------------
class AgentState(TypedDict):
    user_input: str
    category: str
    response: str
    conversation: list[str]


# -----------------------------
# 2. CLASSIFY NODE
# -----------------------------
def classify(state: AgentState):

    user_input = state["user_input"].lower()

    if any(word in user_input for word in [
        "urgent",
        "emergency",
        "problem",
        "help"
    ]):
        category = "important"
    else:
        category = "general"

    print(f"[CLASSIFY] Category: {category}")

    return {
        "category": category
    }


# -----------------------------
# 3. ROUTE NODE
# -----------------------------
def route(state: AgentState):

    category = state["category"]

    print(f"[ROUTE] Sending to: {category}")

    return {}


# -----------------------------
# 4. HUMAN APPROVAL NODE
# -----------------------------
def human_approval(state: AgentState):

    print("\n[HUMAN APPROVAL]")

    decision = interrupt(
        {
            "message": "Human approval required.",
            "question": f"Should we respond to this request?\n"
                        f"User: {state['user_input']}"
        }
    )

    print(f"Human decision: {decision}")

    return {}


# -----------------------------
# 5. RESPOND NODE
# -----------------------------
def respond(state: AgentState):

    category = state["category"]
    user_input = state["user_input"]

    conversation = state.get("conversation", [])

    conversation.append(
        f"User: {user_input}"
    )

    if category == "important":

        response = (
            "Your important request has been reviewed. "
            "Please provide more details so we can help."
        )

    else:

        response = (
            f"You said: '{user_input}'. "
            "This is a general request."
        )

    conversation.append(
        f"Assistant: {response}"
    )

    print(f"[RESPOND] {response}")

    return {
        "response": response,
        "conversation": conversation
    }


# -----------------------------
# 6. CONDITIONAL ROUTING
# -----------------------------
def conditional_route(state: AgentState):

    if state["category"] == "important":
        return "important"

    return "general"


# -----------------------------
# 7. BUILD GRAPH
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("human_approval", human_approval)
builder.add_node("respond", respond)


# START → CLASSIFY
builder.add_edge(
    START,
    "classify"
)


# CLASSIFY → ROUTE
builder.add_edge(
    "classify",
    "route"
)


# ROUTE → CONDITIONAL ROUTING
builder.add_conditional_edges(
    "route",
    conditional_route,
    {
        "important": "human_approval",
        "general": "respond"
    }
)


# HUMAN APPROVAL → RESPOND
builder.add_edge(
    "human_approval",
    "respond"
)


# RESPOND → END
builder.add_edge(
    "respond",
    END
)


# -----------------------------
# 8. ADD MEMORY
# -----------------------------
memory = InMemorySaver()

graph = builder.compile(
    checkpointer=memory
)


# -----------------------------
# 9. RUN TEST
# -----------------------------
if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "pallavi-human-test"
        }
    }

    user_message = input(
        "\nEnter your message: "
    )

    print("\nStarting LangGraph...\n")

    result = graph.invoke(
        {
            "user_input": user_message,
            "category": "",
            "response": "",
            "conversation": []
        },
        config
    )

    # ---------------------------------
    # CHECK IF GRAPH WAS INTERRUPTED
    # ---------------------------------

    if "__interrupt__" in result:

        print("\n================================")
        print("🛑 GRAPH INTERRUPTED")
        print("================================")

        interrupt_data = result["__interrupt__"][0].value

        print(
            interrupt_data["question"]
        )

        human_input = input(
            "\nEnter human decision (yes/no): "
        )

        # ---------------------------------
        # RESUME GRAPH
        # ---------------------------------

        print("\nResuming graph...\n")

        result = graph.invoke(
            Command(
                resume=human_input
            ),
            config
        )

    # ---------------------------------
    # FINAL RESULT
    # ---------------------------------

    print("\n================================")
    print("FINAL RESULT")
    print("================================")

    print(
        result["response"]
    )

    print("\nConversation memory:")

    final_state = graph.get_state(config)

    print(
        final_state.values.get(
            "conversation",
            []
        )
    )