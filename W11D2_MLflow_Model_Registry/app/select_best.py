import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "W11D2 Iris Model Registry"


experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(
        f"Experiment '{EXPERIMENT_NAME}' was not found."
    )


client = MlflowClient()


runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
)


if not runs:
    raise ValueError("No MLflow runs were found.")


print("\n======================================")
print("       W11D2 EXPERIMENT RESULTS")
print("======================================")


for i, run in enumerate(runs, start=1):

    print(f"\nExperiment {i}")
    print("--------------------------------------")

    print(
        "Run Name:",
        run.data.tags.get("mlflow.runName")
    )

    print(
        "Run ID:",
        run.info.run_id
    )

    print(
        "Accuracy:",
        run.data.metrics.get("accuracy")
    )

    print(
        "Precision:",
        run.data.metrics.get("precision")
    )

    print(
        "Recall:",
        run.data.metrics.get("recall")
    )

    print(
        "F1 Score:",
        run.data.metrics.get("f1_score")
    )

    print("Parameters:")

    for key, value in run.data.params.items():
        print(f"  {key}: {value}")


best_run = runs[0]


print("\n======================================")
print("             BEST RUN")
print("======================================")

print("Run ID:", best_run.info.run_id)

print(
    "Run Name:",
    best_run.data.tags.get("mlflow.runName")
)

print(
    "Accuracy:",
    best_run.data.metrics.get("accuracy")
)

print("\nParameters:")

for key, value in best_run.data.params.items():
    print(f"  {key}: {value}")

print("\n======================================")