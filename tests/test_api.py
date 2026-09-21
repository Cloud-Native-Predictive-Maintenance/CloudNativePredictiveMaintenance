"""Integration tests for the FastAPI serving layer. Requires models/ to be
populated first (run `python run_pipeline.py`)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from src.serving.main import app

MODELS_TRAINED = (Path(__file__).resolve().parents[1] / "models" / "failure_classifier.joblib").exists()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.skipif(not MODELS_TRAINED, reason="run run_pipeline.py first to train models")
def test_predict_requires_all_features(client):
    resp = client.post("/predict", json={"machine_id": "M-TEST", "features": {}})
    assert resp.status_code == 422


@pytest.mark.skipif(not MODELS_TRAINED, reason="run run_pipeline.py first to train models")
def test_models_info(client):
    resp = client.get("/models/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "failure_classifier" in body["cmapss"]
    assert len(body["cmapss"]["classifier_leaderboard"]) == 3


AI4I_MODELS_TRAINED = (Path(__file__).resolve().parents[1] / "models_ai4i" / "failure_classifier.joblib").exists()


@pytest.mark.skipif(not AI4I_MODELS_TRAINED, reason="run run_pipeline_ai4i.py first to train models")
def test_predict_ai4i(client):
    resp = client.post("/predict/ai4i", json={
        "machine_id": "TEST-1", "air_temp_k": 298.1, "process_temp_k": 308.6,
        "rotational_speed_rpm": 1551, "torque_nm": 42.8, "tool_wear_min": 0, "type": "M",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["health_status"] in {"HIGH RISK", "MODERATE RISK", "HEALTHY"}
