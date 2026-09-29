import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="W12D1 MLOps ML API")


# Train a simple ML model
iris = load_iris()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(iris.data, iris.target)


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {
        "message": "W12D1 MLOps ML API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "class_name": iris.target_names[prediction]
    }