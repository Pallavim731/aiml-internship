import mlflow

from app.graph import run_rag


EXPERIMENT_NAME = "W11D5_Tracked_Evaluated_RAG"

CHUNK_SIZE = 500
RETRIEVAL_K = 3
LLM_MODEL = "llama3.2:3b"


def run_and_track(question):
    """Run the RAG pipeline and track the result with MLflow."""

    # Create or select the MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Run the LangGraph RAG pipeline
        result = run_rag(question)

        answer = result["answer"]
        contexts = result["contexts"]

        # Log RAG configuration
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("retrieval_k", RETRIEVAL_K)
        mlflow.log_param("embedding_model", "nomic-embed-text")
        mlflow.log_param("llm_model", LLM_MODEL)

        # Log simple pipeline metrics
        mlflow.log_metric("retrieved_context_count", len(contexts))
        mlflow.log_metric("answer_length", len(answer))

        # Save the answer as an MLflow artifact
        with open("mlflow_answer.txt", "w", encoding="utf-8") as file:
            file.write(f"Question:\n{question}\n\n")
            file.write(f"Answer:\n{answer}\n\n")
            file.write("Retrieved Contexts:\n\n")

            for index, context in enumerate(contexts, start=1):
                file.write(f"--- Context {index} ---\n")
                file.write(context)
                file.write("\n\n")

        mlflow.log_artifact("mlflow_answer.txt")

        print("\nMLflow Run Completed")
        print("Run ID:", mlflow.active_run().info.run_id)
        print("Experiment:", EXPERIMENT_NAME)

        return result


if __name__ == "__main__":
    question = "What is RAG?"

    result = run_and_track(question)

    print("\nFinal Answer:")
    print(result["answer"])