from app import graph


def test_five_inputs():

    test_inputs = [
        "Hello, how are you?",
        "What is LangGraph?",
        "I need help with my project",
        "This is an urgent problem",
        "I have an emergency"
    ]

    config = {
        "configurable": {
            "thread_id": "five-input-test"
        }
    }

    for message in test_inputs:

        print("\n" + "=" * 50)
        print("INPUT:", message)

        result = graph.invoke(
            {
                "user_input": message,
                "category": "",
                "response": "",
                "conversation": []
            },
            config
        )

        # Important requests pause for human approval.
        if "__interrupt__" in result:

            print("STATUS: Human approval required")

            from langgraph.types import Command

            result = graph.invoke(
                Command(resume="yes"),
                config
            )

        print("CATEGORY:", result["category"])
        print("RESPONSE:", result["response"])


if __name__ == "__main__":
    test_five_inputs()