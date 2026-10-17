"""The client against the contract's fixtures: every request as the runtime expects it, every answer read right,
every refusal an AdaError with the runtime's code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import validate

from glyphh_ada import Ada, AdaError, Result

CONTRACT = json.loads((Path(__file__).parents[3] / "contract" / "contract.json").read_text())
FIXTURES = json.loads((Path(__file__).parents[3] / "contract" / "fixtures.json").read_text())


class Fake:
    """A runtime that answers from a script of (method, path, body) -> (status, body)."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, dict(headers), json.loads(body) if body else None))
        status, answer = self.answers.pop(0)
        return status, json.dumps(answer).encode()


def case(name):
    return next(c for c in FIXTURES["cases"] if c["name"] == name)


def ada(*answers):
    fake = Fake(*answers)
    return Ada("https://m1.glyphh.app/", "glyphh_rk_test", transport=fake), fake


def test_a_decide_that_is_done_with_a_clearance():
    c = case("a call that is done")
    client, fake = ada((200, c["response"]["body"]))
    r = client.model("incidents").decide(c["request"]["body"]["inputs"]["data"], clearance=0.5)
    validate(c["response"]["body"], CONTRACT["result"])
    assert isinstance(r, Result) and r.done and not r.waiting and r.ended
    assert (r.run_id, r.data["fix"], r.seconds) == ("7e2a1c", "raise circuit breaker", 0.21)
    method, url, headers, body = fake.calls[0]
    assert (method, url) == ("POST", "https://m1.glyphh.app/v1/runs")
    assert headers["authorization"] == "Bearer glyphh_rk_test" and headers["content-type"] == "application/json"
    assert body == c["request"]["body"]


def test_a_run_that_needs_input_is_resumed():
    c = case("a call that needs input, then continued")
    client, fake = ada((200, c["response"]["body"]), (200, c["then"]["response"]["body"]))
    m = client.model("incidents")
    r = m.decide(c["request"]["body"]["inputs"]["data"])
    assert r.waiting and r.status == "needs_input" and "How was this one fixed" in r.prompt
    done = m.resume(r.run_id, "rotate certificate")
    assert done.done and done.data["fix"] == "rotate certificate" and done.data["recorded"]
    assert fake.calls[1][1] == "https://m1.glyphh.app/v1/runs/9b0d44/continue"
    assert fake.calls[1][3] == {"answer": "rotate certificate"}


def test_a_query_answers_the_contracts_answer_and_the_other_functions_name_themselves():
    c = case("a query")
    body = c["response"]["body"]
    client, fake = ada(*[(200, body)] * 7)
    m = client.model("incidents")
    r = m.query(c["request"]["body"]["inputs"]["data"])
    validate(r.data, CONTRACT["answer"])
    assert r.data["top"] == "rotate certificate" and r.data["sufficient"]
    m.record({"a": 1}, "fixed")
    m.record({"a": 1})
    m.facts({"a": 1}, top=2, clearance=0.5)
    m.probabilities({"a": 1})
    m.learn()
    m.learn(sample=50, clearance=0.5)
    sent = [(call[3]["function"], call[3]["inputs"], call[3].get("clearance")) for call in fake.calls]
    assert sent == [
        ("query", {"data": c["request"]["body"]["inputs"]["data"]}, None),
        ("record", {"data": {"a": 1}, "outcome": "fixed"}, None),
        ("record", {"data": {"a": 1}}, None),
        ("facts", {"data": {"a": 1}, "top": 2}, 0.5),
        ("probabilities", {"data": {"a": 1}}, None),
        ("learn", {}, None),
        ("learn", {"sample": 50}, 0.5),
    ]


@pytest.mark.parametrize("name", ["a wrong key", "a paused runtime", "a model that is not there"])
def test_a_refusal_is_an_ada_error_with_the_runtimes_code(name):
    c = case(name)
    client, _ = ada((c["response"]["status"], c["response"]["body"]))
    validate(c["response"]["body"], CONTRACT["error"])
    with pytest.raises(AdaError) as caught:
        client.model(c["request"]["body"]["model"]).call("start")
    err = caught.value
    assert (err.code, err.status) == (c["response"]["body"]["error"]["code"], c["response"]["status"])
    assert err.message == c["response"]["body"]["error"]["message"]
    if name == "a paused runtime":
        assert err.detail == {"limit": "runtime_hours"}
    assert repr(err).startswith("AdaError(")


def test_the_runtime_and_the_rest_of_the_api():
    models = {"models": [{"name": "incidents", "active": 1}]}
    run = {"id": "r1", "status": "done"}
    client, fake = ada((200, models), (201, {"model": "incidents", "version": 2, "commit_sha": "a" * 40}), (200, run), (200, {"lines": [1]}), (200, case("a call that is done")["response"]["body"]))
    assert client.models() == models["models"]
    assert client.deploy("octo/incidents", "a" * 40, "ghp_x")["version"] == 2
    assert client.run("r1") == run and client.log("r1") == [1]
    assert client.cancel("r1").done
    assert [(c[0], c[1].removeprefix("https://m1.glyphh.app")) for c in fake.calls] == [
        ("GET", "/v1/models"),
        ("POST", "/v1/models/deploy"),
        ("GET", "/v1/runs/r1"),
        ("GET", "/v1/runs/r1/log"),
        ("POST", "/v1/runs/r1/cancel"),
    ]
    assert fake.calls[1][3] == {"repo": "octo/incidents", "commit": "a" * 40, "token": "ghp_x"}
    assert fake.calls[0][3] is None and "content-type" not in fake.calls[0][2]


def test_an_answer_that_is_not_json_or_no_code_and_a_runtime_that_cannot_be_reached():
    def broken(method, url, headers, body):
        return 502, b"<html>bad gateway</html>"

    with pytest.raises(AdaError) as caught:
        Ada("https://m1.glyphh.app", "k", transport=broken).models()
    assert (caught.value.code, caught.value.status) == ("E_RUNTIME", 502)

    def gone(method, url, headers, body):
        raise OSError("no route to host")

    with pytest.raises(AdaError) as caught:
        Ada("https://m1.glyphh.app", "k", transport=gone).models()
    assert caught.value.code == "E_UNREACHABLE"
    for status, code in ((401, "E_AUTH"), (402, "E_LIMIT"), (404, "E_NOT_FOUND")):
        with pytest.raises(AdaError) as caught:
            Ada("https://m1.glyphh.app", "k", transport=lambda *a, s=status: (s, b"{}")).models()
        assert caught.value.code == code


def test_the_address_and_key_come_from_the_environment_or_are_refused(monkeypatch):
    monkeypatch.delenv("GLYPHH_RUNTIME_URL", raising=False)
    monkeypatch.delenv("GLYPHH_RUNTIME_KEY", raising=False)
    with pytest.raises(AdaError) as caught:
        Ada()
    assert caught.value.code == "E_CONFIG"
    monkeypatch.setenv("GLYPHH_RUNTIME_URL", "https://m2.glyphh.app/")
    monkeypatch.setenv("GLYPHH_RUNTIME_KEY", "glyphh_rk_env")
    assert Ada().url == "https://m2.glyphh.app"
