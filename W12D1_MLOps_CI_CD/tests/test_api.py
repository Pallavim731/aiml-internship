from app.api import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "W12D1 MLOps ML API is running" in response.json()["message"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "features": [5.1, 3.5, 1.4, 0.2]
        }
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "class_name" in response.json()