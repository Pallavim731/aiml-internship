import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "W11D2 Iris Model Registry"
MODEL_NAME = "W11D2_Iris_Random_Forest"


# Get experiment
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(
        f"Experiment '{EXPERIMENT_NAME}' was not found."
    )


client = MlflowClient()


# Find the best run by accuracy
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
)


if not runs:
    raise ValueError("No MLflow runs were found.")


best_run = runs[0]

run_id = best_run.info.run_id
accuracy = best_run.data.metrics.get("accuracy")

model_uri = f"runs:/{run_id}/model"


print("\n================================")
print("        BEST RUN SELECTED")
print("================================")

print("Run ID:", run_id)
print("Run Name:", best_run.data.tags.get("mlflow.runName"))
print("Accuracy:", accuracy)

print("\nModel URI:")
print(model_uri)

print("\nRegistering model...")


# Register model
result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)


print("\n================================")
print("   MODEL REGISTERED SUCCESSFULLY")
print("================================")

print("Model Name:", result.name)
print("Model Version:", result.version)
print("Run ID:", run_id)
print("Accuracy:", accuracy)

print("\n================================")