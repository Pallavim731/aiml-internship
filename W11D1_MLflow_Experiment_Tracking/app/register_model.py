import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "W11D1 Iris Random Forest"
MODEL_NAME = "W11D1_Iris_Random_Forest"


# -----------------------------------------
# Find the experiment
# -----------------------------------------

experiment = mlflow.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    raise ValueError(
        f"Experiment '{EXPERIMENT_NAME}' not found."
    )


# -----------------------------------------
# Connect to MLflow
# -----------------------------------------

client = MlflowClient()


# -----------------------------------------
# Get all runs
# Sort by accuracy from highest to lowest
# -----------------------------------------

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)


if not runs:
    raise ValueError("No MLflow runs found.")


# -----------------------------------------
# Select best run
# -----------------------------------------

best_run = runs[0]

run_id = best_run.info.run_id
accuracy = best_run.data.metrics.get("accuracy")


print("\n================================")
print("BEST RUN SELECTED")
print("================================")
print("Run ID:", run_id)
print("Run Name:", best_run.data.tags.get("mlflow.runName"))
print("Accuracy:", accuracy)


# -----------------------------------------
# Model URI
# -----------------------------------------

model_uri = f"runs:/{run_id}/model"

print("\nModel URI:")
print(model_uri)


# -----------------------------------------
# Register model
# -----------------------------------------

print("\nRegistering model...")

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)


# -----------------------------------------
# Result
# -----------------------------------------

print("\n================================")
print("MODEL REGISTERED SUCCESSFULLY")
print("================================")

print("Model Name:", result.name)
print("Model Version:", result.version)
print("Run ID:", run_id)
print("Accuracy:", accuracy)

print("\nYou can now use this model for serving.")