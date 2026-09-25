"""Tests for the /v1/decide API routes (standalone app, no server infra)."""

import importlib.util
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_spec = importlib.util.spec_from_file_location(
    "decide_routes",
    Path(__file__).parents[2] / "api" / "routes" / "decide.py",
)
decide_routes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(decide_routes)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(decide_routes, "_space", None)
    app = FastAPI()
    app.include_router(decide_routes.router)
    return TestClient(app)


def record(client, situation, outcome):
    response = client.post("/v1/decide/record",
                           json={"situation": situation, "outcome": outcome})
    assert response.status_code == 201, response.text
    return response.json()


def test_decide_with_no_precedent_declines(client):
    result = client.post(
        "/v1/decide", json={"situation": {"kind": "novel"}}).json()
    assert result["sufficient"] is False
    assert result["confidence"] == 0.0


def test_record_then_decide(client):
    for _ in range(3):
        record(client, {"amount": "small", "history": "good"}, "approved")
        record(client, {"amount": "large", "history": "poor"}, "declined")
    result = client.post(
        "/v1/decide",
        json={"situation": {"amount": "small", "history": "good"}}).json()
    assert result["sufficient"] is True
    assert result["top_outcome"] == "approved"
    assert result["fact_tree"]["children"]
    precedent_ids = [p["decision_id"] for p in result["precedents"]]
    tree_text = str(result["fact_tree"])
    assert precedent_ids[0] in tree_text  # the tree cites the precedent


def test_stats(client):
    record(client, {"a": "b"}, "x")
    stats = client.get("/v1/decide/stats").json()
    assert stats["precedents"] == 1
