import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
    stratify=y
)

# MLflow experiment
mlflow.set_experiment("W11D3 MLflow Model Serving")

# Five different hyperparameter combinations
experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
]

best_accuracy = 0
best_run_id = None

for i, params in enumerate(experiments, start=1):

    with mlflow.start_run(run_name=f"experiment_{i}"):

        # Create model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(
    model,
    "model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
        run_id = mlflow.active_run().info.run_id

        print(
            f"Experiment {i}: "
            f"n_estimators={params['n_estimators']}, "
            f"max_depth={params['max_depth']}, "
            f"accuracy={accuracy:.4f}"
        )

        # Track best experiment
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = run_id


print("\nTraining completed.")
print(f"Best accuracy: {best_accuracy:.4f}")
print(f"Best run ID: {best_run_id}")
# Register the best model
model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="W11D3_RandomForest_Best"
)

print("\nBest model registered successfully.")
print(f"Model name: {registered_model.name}")
print(f"Model version: {registered_model.version}")