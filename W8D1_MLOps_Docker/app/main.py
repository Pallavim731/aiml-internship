from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="ML Prediction API",
    description="Dockerised ML API for MLOps",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    feature1: float
    feature2: float


class PredictionResponse(BaseModel):
    prediction: float


@app.get("/")
def root():
    return {
        "message": "ML Prediction API is running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    prediction = request.feature1 + request.feature2

    return {
        "prediction": prediction
    }