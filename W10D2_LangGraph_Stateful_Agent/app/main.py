from langgraph.types import Command

from app.graph import graph


def run_agent(user_input: str, thread_id: str):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = graph.invoke(
        {"user_input": user_input},
        config=config
    )

    if "__interrupt__" in result:
        print("\n--- HUMAN REVIEW REQUIRED ---")
        print(result["__interrupt__"][0].value)

        approval = input("Enter approval (yes/no): ")

        result = graph.invoke(
            Command(resume=approval),
            config=config
        )

    return result


if __name__ == "__main__":

    print("=" * 45)
    print("     LangGraph Stateful Agent")
    print("=" * 45)

    user_input = input("\nEnter your question: ")

    result = run_agent(
        user_input,
        thread_id="demo-thread"
    )

    print("\n--- FINAL RESULT ---")
    print("Category:", result.get("category"))
    print("Response:", result.get("response"))
    print("Human approval:", result.get("human_approval"))