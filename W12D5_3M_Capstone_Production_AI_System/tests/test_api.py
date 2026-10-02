from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert "W12D5 Production AI API is running" in response.json()["message"]


def test_prediction():
    response = client.post("/predict", json={"value": 5})

    assert response.status_code == 200
    assert response.json()["prediction"] == 10