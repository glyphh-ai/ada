"""Tests for the /v1/strand API routes (standalone app, no server infra)."""

import importlib.util
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Load the router module by path: api.routes.__init__ pulls in DB-backed
# routes whose dependencies this unit test doesn't need.
_spec = importlib.util.spec_from_file_location(
    "strand_routes",
    Path(__file__).parents[2] / "api" / "routes" / "strand.py",
)
strand_routes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strand_routes)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(strand_routes, "_service", None)  # fresh deployment
    app = FastAPI()
    app.include_router(strand_routes.router)
    return TestClient(app)


def post_event(client, sid, actor, action, attributes=None):
    response = client.post(
        f"/v1/strand/sessions/{sid}/events",
        json={"actor": actor, "action": action,
              "attributes": attributes or {}},
    )
    assert response.status_code == 200, response.text
    return response.json()


WORKFLOW = [
    ("user", "ask_balance", {"account": "checking"}),
    ("agent", "authenticate", {}),
    ("user", "provide_credentials", {}),
    ("agent", "show_balance", {"account": "checking"}),
]


def run_workflow(client, steps=WORKFLOW):
    sid = client.post("/v1/strand/sessions").json()["session_id"]
    result = None
    for actor, action, attrs in steps:
        result = post_event(client, sid, actor, action, attrs)
    return sid, result


def test_create_and_summarize_session(client):
    sid = client.post("/v1/strand/sessions").json()["session_id"]
    summary = client.get(f"/v1/strand/sessions/{sid}").json()
    assert summary == {"session_id": sid, "turns": 0, "drift": 0.0}


def test_unknown_session_404(client):
    assert client.get("/v1/strand/sessions/nope").status_code == 404


def test_event_returns_prediction_shape(client):
    sid = client.post("/v1/strand/sessions").json()["session_id"]
    result = post_event(client, sid, "user", "ask_balance",
                        {"account": "checking"})
    assert result["turn"] == 1
    assert result["event_margin"] == 0.0  # first turn of first session
    assert isinstance(result["predicted_next"], list)


def test_predicts_learned_workflow(client):
    """After watching a workflow, the API anticipates its next step."""
    for _ in range(3):
        run_workflow(client)
    sid = client.post("/v1/strand/sessions").json()["session_id"]
    post_event(client, sid, "user", "ask_balance", {"account": "checking"})
    result = post_event(client, sid, "agent", "authenticate")
    top = result["predicted_next"][0]
    assert (top["actor"], top["action"]) == ("user", "provide_credentials")


def test_hijacked_session_drifts_more(client):
    """A trajectory that swerves mid-session scores higher drift."""
    for _ in range(5):
        run_workflow(client)
    _, normal = run_workflow(client)

    hijacked_steps = WORKFLOW[:2] + [
        ("user", "wire_transfer", {"amount": "all", "dest": "external"}),
        ("agent", "execute_transfer", {}),
    ]
    _, hijacked = run_workflow(client, hijacked_steps)
    assert hijacked["drift"] > normal["drift"]


def test_end_session_keeps_learning(client):
    sid, result = run_workflow(client)
    summary = client.delete(f"/v1/strand/sessions/{sid}").json()
    assert summary["turns"] == len(WORKFLOW)
    assert client.get(f"/v1/strand/sessions/{sid}").status_code == 404
    # a new session still benefits from what the ended one taught
    sid2, _ = run_workflow(client)
    post_sid = client.get(f"/v1/strand/sessions/{sid2}").json()
    assert post_sid["turns"] == len(WORKFLOW)
