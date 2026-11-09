import pytest
from fastapi.testclient import TestClient

from txclf import api


@pytest.fixture()
def client(trained, monkeypatch):
    _, _, path = trained
    monkeypatch.setenv("TXCLF_MODEL", str(path))
    api.get_model.cache_clear()
    yield TestClient(api.app)
    api.get_model.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["categories"] == 8


def test_predict_batch(client):
    r = client.post(
        "/predict", json={"descriptions": ["jumbo bag red retrospot", "doormat union flag"]}
    )
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2 and len(preds[0]["top_k"]) == 3


def test_predict_validates_input(client):
    assert client.post("/predict", json={"descriptions": []}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"descriptions": ["a"], "top_k": 0}).status_code == 422


def test_missing_model_returns_503(monkeypatch):
    monkeypatch.setenv("TXCLF_MODEL", "nope/missing.joblib")
    api.get_model.cache_clear()
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "error"
    assert c.post("/predict", json={"descriptions": ["a"]}).status_code == 503
    api.get_model.cache_clear()
