from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="W12D3 ML Monitoring API")


class PredictionRequest(BaseModel):
    value: float


@app.get("/")
def root():
    return {"message": "W12D3 ML API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = request.value * 2
    return {
        "input": request.value,
        "prediction": prediction,
    }