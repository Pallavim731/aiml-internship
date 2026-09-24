import mlflow
import numpy as np


MODEL_NAME = "W11D1_Iris_Random_Forest"
MODEL_VERSION = 1


print("Loading registered model...")


# Load model from MLflow Model Registry
model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)


print("Model loaded successfully!")


# Example Iris flower
sample = np.array([
    [5.1, 3.5, 1.4, 0.2]
])


# Make prediction
prediction = model.predict(sample)


print("\n================================")
print("PREDICTION")
print("================================")

print("Input:", sample.tolist())
print("Prediction:", prediction.tolist())
print("Predicted class:", int(prediction[0]))