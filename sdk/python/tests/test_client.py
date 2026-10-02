"""The client against what the server returns: contract/fixtures.json holds
real request and response pairs, and contract/contract.json the schemas the
server publishes."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jsonschema
import pytest

from glyphh_ada import DEFAULT_URL, Ada, AdaError, AdaModel

ROOT = Path(__file__).resolve().parents[3] / "contract"
FIXTURES: Dict[str, Any] = json.loads((ROOT / "fixtures.json").read_text())
CONTRACT: Dict[str, Any] = json.loads((ROOT / "contract.json").read_text())
STATUS = {"E_VALIDATION": 400, "E_UNAUTHENTICATED": 401, "E_FORBIDDEN": 403, "E_NOT_FOUND": 404, "E_FAILED_PRECONDITION": 409}
Sent = List[Tuple[str, Dict[str, str], Any]]


def replaying(name: str, **options: Any) -> Tuple[Ada, Sent]:
    """An Ada whose transport answers with one fixture's response, and keeps what it was sent."""
    sent: Sent = []
    response = FIXTURES[name]["response"]

    def transport(url: str, headers: Any, body: bytes, timeout: float) -> Tuple[int, bytes]:
        sent.append((url, dict(headers), json.loads(body)))
        return (STATUS.get(response["error"]["code"], 400) if "error" in response else 200), json.dumps(response).encode()

    return Ada(api_key="sk-test", transport=transport, **options), sent


def m(ada: Ada, r: Dict[str, Any]) -> AdaModel:
    return ada.model(r["model_id"])


CALLS: Dict[str, Callable[[Ada, Dict[str, Any]], Any]] = {
    "create_model": lambda ada, r: ada.create_model(r["name"], r["storage"], spec=r["spec"]).created,
    "create_accounts": lambda ada, r: ada.create_model(r["name"], r["storage"], spec=r["spec"]).created,
    "record": lambda ada, r: m(ada, r).record(r["situation"], r["outcome"]),
    "models": lambda ada, r: {"models": ada.models()},
    "models_one": lambda ada, r: {"models": [m(ada, r).info()]},
    "query_act": lambda ada, r: m(ada, r).query(r["situation"]),
    "query_unseen": lambda ada, r: m(ada, r).query(r["situation"]),
    "update_model": lambda ada, r: m(ada, r).update(scale=r["scale"]),
    "facts": lambda ada, r: m(ada, r).facts(r["situation"], top=r["top"]),
    "check": lambda ada, r: m(ada, r).check(r["situation"], r["outcome"]),
    "history": lambda ada, r: m(ada, r).history(r["situation"]),
    "trend": lambda ada, r: m(ada, r).trend(r["situation"]),
    "predict": lambda ada, r: m(ada, r).predict(r["situation"]),
    "edges": lambda ada, r: m(ada, r).edges(r["situation"], level=r["level"]),
    "gql_find": lambda ada, r: m(ada, r).gql(r["query"]),
    "gql_count": lambda ada, r: m(ada, r).gql(r["query"], r["args"]),
    "gql_compare": lambda ada, r: m(ada, r).gql(r["query"]),
    "gql_follow": lambda ada, r: m(ada, r).gql(r["query"]),
    "gql_path": lambda ada, r: m(ada, r).gql(r["query"]),
    "save_procedure": lambda ada, r: m(ada, r).save_procedure(r["name"], r["query"], r["description"]),
    "procedures": lambda ada, r: {"model_id": r["model_id"], "procedures": m(ada, r).procedures()},
    "call": lambda ada, r: m(ada, r).call(r["name"], r["args"]),
    "records": lambda ada, r: m(ada, r).records(limit=r["limit"]),
    "contract_model": lambda ada, r: m(ada, r).contract(),
    "delete_model": lambda ada, r: m(ada, r).delete(),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_call_sends_what_the_server_takes_and_returns_what_it_answered(name: str) -> None:
    ada, sent = replaying(name)
    got = CALLS[name](ada, FIXTURES[name]["request"])
    assert len(sent) == 1
    url, headers, body = sent[0]
    assert body == FIXTURES[name]["request"], "the request body is the op and its arguments, nothing else"
    assert (url, headers["x-glyphh-api-key"], headers["content-type"]) == (f"{DEFAULT_URL}/ada", "Bearer sk-test", "application/json")
    assert got == FIXTURES[name]["response"]["data"]


def test_every_fixture_that_is_not_an_error_has_a_call_here() -> None:
    assert [n for n in FIXTURES if not n.startswith("error_") and n not in CALLS] == []


@pytest.mark.parametrize("name, code, status", [("error_validation", "E_VALIDATION", 400), ("error_not_found", "E_NOT_FOUND", 404),
                                               ("error_forbidden", "E_FORBIDDEN", 403)])
def test_a_refusal_is_an_ada_error_that_says_which_kind(name: str, code: str, status: int) -> None:
    ada, _ = replaying(name)
    r = FIXTURES[name]["request"]
    with pytest.raises(AdaError) as e:
        m(ada, r).query(r["situation"]) if r["op"] == "query" else m(ada, r).delete_procedure(r["name"])
    assert (e.value.code, e.value.status, e.value.message) == (code, status, FIXTURES[name]["response"]["error"]["message"])
    assert str(e.value) == e.value.message and code in repr(e.value)


@pytest.mark.parametrize("schema, names", [("query", ["query_act", "query_unseen"]), ("facts", ["facts"]), ("history", ["history"]),
                                          ("trend", ["trend"]), ("predict", ["predict"]), ("edges", ["edges"]),
                                          ("gql", ["gql_find", "gql_count", "gql_compare", "gql_follow", "gql_path", "call"])])
def test_what_the_server_returns_is_what_its_contract_says(schema: str, names: List[str]) -> None:
    jsonschema.Draft202012Validator.check_schema(CONTRACT[schema])
    for name in names:
        jsonschema.Draft202012Validator(CONTRACT[schema]).validate(FIXTURES[name]["response"]["data"])
    data = FIXTURES["contract_model"]["response"]["data"]["data"]
    jsonschema.Draft202012Validator(data).validate(FIXTURES["record"]["request"]["situation"])
    assert not jsonschema.Draft202012Validator(data).is_valid({"deal": {"metrics": {"meetings": "many"}}})


def test_an_answer_is_typed_data_a_program_branches_on() -> None:
    model_id = FIXTURES["record"]["request"]["model_id"]
    acted = replaying("query_act")[0].model(model_id).query({})
    assert (acted["act"], acted["reason"], acted["top"], acted["unseen"]) == (True, "reflex", "open", [])
    refused = replaying("query_unseen")[0].model(model_id).query({})
    assert refused["act"] is False and refused["reason"] in ("unseen", "insufficient") and "deal.metrics.meetings=19" in refused["unseen"]
    followed = replaying("gql_follow")[0].model(model_id).gql("")
    assert followed["statement"] == "follow" and sorted(n["hops"] for n in followed["nodes"]) == [1, 1, 2]
    ahead = replaying("predict")[0].model(model_id).predict({})
    assert ahead["roles"]["deal.metrics.meetings"]["basis"] == "trend" and ahead["answer"]["contract"] == CONTRACT["contract"] == 2


def test_the_key_and_the_url_come_from_the_environment_and_none_is_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLYPHH_API_KEY", raising=False)
    with pytest.raises(AdaError) as e:
        Ada()
    assert e.value.code == "E_UNAUTHENTICATED"
    monkeypatch.setenv("GLYPHH_API_KEY", "sk-env")
    monkeypatch.setenv("GLYPHH_URL", "https://ada.example/")
    sent: Sent = []

    def transport(url: str, headers: Any, body: bytes, timeout: float) -> Tuple[int, bytes]:
        sent.append((url, dict(headers), json.loads(body)))
        assert timeout == 5
        return 200, b'{"data": {"models": [{"model_id": "am_0123456789ab"}]}}'

    ada = Ada(headers={"x-trace": "1"}, timeout=5, transport=transport)
    ada.models()
    model = ada.model("am_0123456789ab")
    model.query({"a": 1}, weights=None)
    model.update(name="renamed", clear_weights=True)
    assert (sent[0][0], sent[0][1]["x-glyphh-api-key"], sent[0][1]["x-trace"]) == ("https://ada.example/ada", "Bearer sk-env", "1")
    assert sent[1][2] == {"op": "query", "model_id": "am_0123456789ab", "situation": {"a": 1}}
    assert sent[2][2] == {"op": "update_model", "model_id": "am_0123456789ab", "name": "renamed", "weights": None}, "cleared weights are sent as null"
    assert repr(model) == "AdaModel('am_0123456789ab')"


def test_a_server_that_cannot_be_reached_or_answers_without_a_body_is_an_ada_error_too() -> None:
    def down(*_: Any) -> Tuple[int, bytes]:
        raise OSError("connection refused")

    with pytest.raises(AdaError) as e:
        Ada("k", transport=down).models()
    assert (e.value.code, e.value.status) == ("E_UNREACHABLE", 0) and "connection refused" in e.value.message
    for status, body, code, message in ((402, b"no plan", "E_PAYMENT", "Ada answered 402"), (401, b'{"detail": "bad key"}', "E_UNAUTHENTICATED", "bad key"),
                                        (500, b"", "E_ADA", "Ada answered 500"), (200, b'{"nope": 1}', "E_ADA", "Ada answered 200")):
        with pytest.raises(AdaError) as e:
            Ada("k", transport=lambda *_, s=status, b=body: (s, b)).models()
        assert (e.value.code, e.value.status, e.value.message) == (code, status, message)
