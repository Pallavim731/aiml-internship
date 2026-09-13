import mlflow

from app.workflow import research_graph


def save_output(result):
    with open("outputs/research_report.md", "w", encoding="utf-8") as file:
        file.write(result["report"])

    with open("outputs/review.txt", "w", encoding="utf-8") as file:
        file.write(result["review"])


def main():
    topic = input("Enter a research topic: ").strip()

    if not topic:
        topic = "Generative AI"

    print("\nRunning automated research agent...\n")

    # Start MLflow experiment tracking
    mlflow.set_experiment("Automated Research Report Agent")

    with mlflow.start_run():
        mlflow.log_param("topic", topic)

        result = research_graph.invoke({
            "topic": topic
        })

        save_output(result)

        # Track basic output metrics
        mlflow.log_metric("report_length", len(result["report"]))

        mlflow.log_artifact("outputs/research_report.md")
        mlflow.log_artifact("outputs/review.txt")

    print("Research completed successfully!")
    print("\n--- REVIEW ---")
    print(result["review"])
    print("\nReport saved to:")
    print("outputs/research_report.md")


if __name__ == "__main__":
    main()