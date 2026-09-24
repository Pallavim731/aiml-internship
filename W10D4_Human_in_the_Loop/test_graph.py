from app import graph
from langgraph.types import Command


def run_test(message, thread_id):
    print("\n" + "=" * 60)
    print("INPUT:", message)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = graph.invoke(
        {
            "user_input": message,
            "category": "",
            "response": ""
        },
        config
    )

    # If the graph pauses for human approval
    if "__interrupt__" in result:

        print("STATUS: INTERRUPTED")
        print("Human approval required.")

        # Simulate human approval
        result = graph.invoke(
            Command(resume="yes"),
            config
        )

        print("STATUS: RESUMED")

    print("CATEGORY:", result["category"])
    print("RESPONSE:", result["response"])

    return result


def test_five_inputs():

    test_inputs = [
        "Hello, how are you?",
        "What is LangGraph?",
        "I need help with my project",
        "This is an urgent problem",
        "I have an emergency"
    ]

    for number, message in enumerate(test_inputs, start=1):

        run_test(
            message,
            f"w10d4-test-{number}"
        )


if __name__ == "__main__":
    test_five_inputs()