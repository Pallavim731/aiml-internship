"""
Main application for the Stateful Customer Support Agent.
"""

import mlflow

from .workflow import build_workflow


def main():
    print("=" * 60)
    print("STATEFUL CUSTOMER SUPPORT AGENT")
    print("=" * 60)

    customer_id = input("Enter customer ID: ").strip()

    if not customer_id:
        customer_id = "CUST001"

    workflow = build_workflow()

    # Configure MLflow.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Stateful Customer Support Agent")

    # Initial state.
    state = {
        "customer_id": customer_id,
        "conversation": [],
        "current_message": "",
        "response": "",
        "issue_type": "",
        "resolved": False,
    }

    print("\nType 'exit' to stop the conversation.\n")

    while True:
        message = input("Customer: ").strip()

        if message.lower() == "exit":
            print("\nConversation ended.")
            break

        if not message:
            print("Please enter a message.")
            continue

        state["current_message"] = message

        print("\nProcessing...\n")

        with mlflow.start_run():
            mlflow.log_param("customer_id", customer_id)
            mlflow.log_param("message", message)

            state = workflow.invoke(state)

            response = state["response"]

            mlflow.log_param("issue_type", state["issue_type"])
            mlflow.log_metric("response_length", len(response))
            mlflow.log_metric(
                "conversation_turns",
                len(state["conversation"]) // 2,
            )

            mlflow.log_text(
                response,
                "response.txt",
            )

        print("\nAgent:", response)
        print("\nIssue type:", state["issue_type"])
        print("-" * 60)


if __name__ == "__main__":
    main()