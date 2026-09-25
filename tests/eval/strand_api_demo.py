"""
Live demo of the /v1/strand API: trajectory intelligence as a service.

Simulates a deployment at a bank's support desk. The service watches a
handful of normal sessions (no training jobs — it learns as it watches),
then two things happen:

1. A new customer starts the same workflow — the API anticipates each
   next step before it happens.
2. A session gets hijacked mid-flow (credential phase pivots straight
   into a full external wire transfer) — the drift score separates it
   from the normal session.

Run:
    PYTHONPATH=. python tests/eval/strand_api_demo.py
"""

import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

_spec = importlib.util.spec_from_file_location(
    "strand_routes",
    Path(__file__).parents[2] / "api" / "routes" / "strand.py",
)
strand_routes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strand_routes)

WORKFLOW = [
    ("user", "ask_balance", {"account": "checking"}),
    ("agent", "authenticate", {}),
    ("user", "provide_credentials", {}),
    ("agent", "show_balance", {"account": "checking"}),
    ("user", "ask_transactions", {"period": "month"}),
    ("agent", "show_transactions", {"period": "month"}),
]

HIJACKED = WORKFLOW[:3] + [
    ("user", "wire_transfer", {"amount": "all", "dest": "external"}),
    ("agent", "execute_transfer", {}),
]


def open_session(client):
    return client.post("/v1/strand/sessions").json()["session_id"]


def send(client, sid, actor, action, attrs):
    return client.post(
        f"/v1/strand/sessions/{sid}/events",
        json={"actor": actor, "action": action, "attributes": attrs},
    ).json()


def main():
    app = FastAPI()
    app.include_router(strand_routes.router)
    client = TestClient(app)

    print("=== phase 1: the service watches 5 normal sessions ===")
    for _ in range(5):
        sid = open_session(client)
        for actor, action, attrs in WORKFLOW:
            send(client, sid, actor, action, attrs)
    print("(no training job ran — it learned by watching)\n")

    print("=== phase 2: a new customer calls in ===")
    sid = open_session(client)
    for actor, action, attrs in WORKFLOW:
        result = send(client, sid, actor, action, attrs)
        predicted = result["predicted_next"][0] if result["predicted_next"] else None
        line = f"  {actor:>5}: {action:<22}"
        if predicted:
            line += (f"-> api expects next: {predicted['actor']}."
                     f"{predicted['action']} "
                     f"(confidence {predicted['confidence']:.2f})")
        print(line)
    normal_drift = result["drift"]
    print(f"  session drift: {normal_drift:.3f}\n")

    print("=== phase 3: a session gets hijacked mid-flow ===")
    sid = open_session(client)
    for actor, action, attrs in HIJACKED:
        result = send(client, sid, actor, action, attrs)
        flag = "  <-- off-trajectory" if result["event_margin"] > 0.25 else ""
        print(f"  {actor:>5}: {action:<22}margin {result['event_margin']:.3f}{flag}")
    print(f"  session drift: {result['drift']:.3f} "
          f"(normal session was {normal_drift:.3f})")
    verdict = "FLAGGED" if result["drift"] > 2 * max(normal_drift, 0.01) else "not separated"
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
