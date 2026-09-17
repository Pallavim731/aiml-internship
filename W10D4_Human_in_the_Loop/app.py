from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# --------------------------------
# 1. State
# --------------------------------

class AgentState(TypedDict):
    user_input: str
    category: str
    response: str


# --------------------------------
# 2. CLASSIFY NODE
# --------------------------------

def classify(state: AgentState):

    text = state["user_input"].lower()

    important_words = [
        "urgent",
        "emergency",
        "problem",
        "help"
    ]

    if any(word in text for word in important_words):
        category = "important"
    else:
        category = "general"

    print(f"[CLASSIFY] {category}")

    return {
        "category": category
    }


# --------------------------------
# 3. ROUTE NODE
# --------------------------------

def route(state: AgentState):

    print(f"[ROUTE] {state['category']}")

    return {}


# --------------------------------
# 4. HUMAN-IN-THE-LOOP NODE
# --------------------------------

def human_approval(state: AgentState):

    print("\n[HUMAN APPROVAL]")
    print("The graph is waiting for human input.")

    decision = interrupt({
        "question": (
            "Human approval required.\n"
            f"User message: {state['user_input']}\n"
            "Should this request be processed?"
        )
    })

    print(f"Human decision: {decision}")

    return {}


# --------------------------------
# 5. RESPOND NODE
# --------------------------------

def respond(state: AgentState):

    if state["category"] == "important":

        response = (
            "Your important request has been reviewed "
            "and processed."
        )

    else:

        response = (
            f"Your message was received: "
            f"{state['user_input']}"
        )

    print(f"[RESPOND] {response}")

    return {
        "response": response
    }


# --------------------------------
# 6. CONDITIONAL ROUTING
# --------------------------------

def choose_route(state: AgentState):

    if state["category"] == "important":
        return "important"

    return "general"


# --------------------------------
# 7. BUILD GRAPH
# --------------------------------

builder = StateGraph(AgentState)

builder.add_node("classify", classify)
builder.add_node("route", route)
builder.add_node("human_approval", human_approval)
builder.add_node("respond", respond)


builder.add_edge(
    START,
    "classify"
)

builder.add_edge(
    "classify",
    "route"
)


builder.add_conditional_edges(
    "route",
    choose_route,
    {
        "general": "respond",
        "important": "human_approval"
    }
)


builder.add_edge(
    "human_approval",
    "respond"
)

builder.add_edge(
    "respond",
    END
)


# --------------------------------
# 8. MEMORY / CHECKPOINT
# --------------------------------

memory = InMemorySaver()

graph = builder.compile(
    checkpointer=memory
)


# --------------------------------
# 9. INTERACTIVE TEST
# --------------------------------

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "w10d4-human-test"
        }
    }

    message = input(
        "\nEnter your message: "
    )

    print("\nStarting graph...\n")

    result = graph.invoke(
        {
            "user_input": message,
            "category": "",
            "response": ""
        },
        config
    )


    # --------------------------------
    # CHECK FOR INTERRUPT
    # --------------------------------

    if "__interrupt__" in result:

        print("\n==============================")
        print("GRAPH PAUSED")
        print("==============================")

        interrupt_data = result["__interrupt__"][0].value

        print(
            interrupt_data["question"]
        )

        human_input = input(
            "\nEnter human decision (yes/no): "
        )


        # --------------------------------
        # RESUME GRAPH
        # --------------------------------

        print("\nResuming graph...\n")

        result = graph.invoke(
            Command(
                resume=human_input
            ),
            config
        )


    print("\n==============================")
    print("FINAL RESPONSE")
    print("==============================")

    print(
        result["response"]
    )