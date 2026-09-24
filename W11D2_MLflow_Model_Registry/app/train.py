import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# MLflow experiment
mlflow.set_experiment("W11D2 Iris Model Registry")


# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Five different experiments
experiments = [
    {
        "n_estimators": 50,
        "max_depth": 3,
        "min_samples_split": 2,
    },
    {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2,
    },
    {
        "n_estimators": 150,
        "max_depth": 8,
        "min_samples_split": 2,
    },
    {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 4,
    },
    {
        "n_estimators": 250,
        "max_depth": None,
        "min_samples_split": 2,
    },
]


for experiment_number, params in enumerate(experiments, start=1):

    print("\n========================================")
    print(f"Running Experiment {experiment_number}")
    print("========================================")

    with mlflow.start_run(
        run_name=f"Experiment_{experiment_number}"
    ) as run:

        # Create model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=42,
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
        )

        # Log parameters
        mlflow.log_param(
            "n_estimators",
            params["n_estimators"],
        )

        mlflow.log_param(
            "max_depth",
            params["max_depth"],
        )

        mlflow.log_param(
            "min_samples_split",
            params["min_samples_split"],
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model artifact
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        # Display results
        print("Run ID     :", run.info.run_id)
        print("Parameters :", params)
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1 Score   : {f1:.4f}")


print("\n========================================")
print("ALL 5 EXPERIMENTS COMPLETED")
print("========================================")