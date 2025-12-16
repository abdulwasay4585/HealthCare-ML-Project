from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

# Note: Real prediction tests depend on loaded models.
# In a real CI, we might mock the models or ensure they are present.
# Here we just check if the endpoint is reachable (it might return 503 if models missing)
def test_predict_endpoint_availability():
    # Even if 503, the app is running
    data = {"age": "[50-60)"} # Partial data
    response = client.post("/predict/readmission", json=data)
    # We expect either 200 (if model) or 503 (if no model) or 422 (validation)
    # Since we passed partial data, might get defaults or 422 if mismatched? 
    # Our Pydantic model has defaults, so it should be valid input.
    assert response.status_code in [200, 503]
