"""The client against what the server returns: contract/fixtures.json holds
real request and response pairs, and contract/contract.json the schemas the
server publishes."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jsonschema
import pytest

from glyphh_ada import DEFAULT_URL, LOAD_BATCH, Ada, AdaError, AdaModel

ROOT = Path(__file__).resolve().parents[3] / "contract"
FIXTURES: Dict[str, Any] = json.loads((ROOT / "fixtures.json").read_text())
CONTRACT: Dict[str, Any] = json.loads((ROOT / "contract.json").read_text())
STATUS = {"E_VALIDATION": 400, "E_UNAUTHENTICATED": 401, "E_FORBIDDEN": 403, "E_NOT_FOUND": 404, "E_FAILED_PRECONDITION": 409}

#: What the gateway in front of Ada answers, taken from a real server: a key it does not know, and a rate limit.
GATEWAY_401 = b'{"error":{"message":"Authentication Error, Invalid proxy server token passed.","type":"token_not_found_in_db","param":"key","code":"401"}}'
GATEWAY_429 = b'{"detail":"Rate limit exceeded for team: t. Limit type: requests. Current limit: 60, Remaining: 0."}'
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
    "load": lambda ada, r: m(ada, r).load(r["records"]),
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


def test_the_gateways_own_refusals_get_adas_codes_and_a_rate_limit_says_how_long_to_wait() -> None:
    def answering(status: int, body: bytes, headers: Dict[str, str]) -> Ada:
        return Ada("k", "http://ada", transport=lambda *_: (status, body, headers))

    with pytest.raises(AdaError) as unknown:
        answering(401, GATEWAY_401, {}).models()
    assert (unknown.value.code, unknown.value.status, unknown.value.retry_after) == ("E_UNAUTHENTICATED", 401, None)
    assert "Invalid proxy server token" in unknown.value.message
    with pytest.raises(AdaError) as limited:
        answering(429, GATEWAY_429, {"Retry-After": "60"}).models()
    assert (limited.value.code, limited.value.status, limited.value.retry_after) == ("E_RATE_LIMITED", 429, 60.0)
    assert limited.value.message.startswith("Rate limit exceeded")
    with pytest.raises(AdaError) as named:
        answering(429, b'{"error": {"code": "E_RATE_LIMITED", "message": "slow down"}}', {"retry-after": "soon"}).models()
    assert (named.value.code, named.value.retry_after) == ("E_RATE_LIMITED", None)
    with pytest.raises(AdaError) as old:
        Ada("k", "http://ada", transport=lambda *_: (403, b'{"error": {"code": "403", "message": "no"}}')).models()
    assert (old.value.code, old.value.retry_after) == ("E_FORBIDDEN", None), "a transport that gives no headers still works"



def test_a_long_load_goes_in_turns_of_load_batch_and_the_answer_sums_what_landed() -> None:
    ada, sent = replaying("load")
    records: List[Any] = [{"situation": {"n": i}, "outcome": "x"} for i in range(LOAD_BATCH + 1)]
    got = m(ada, FIXTURES["load"]["request"]).load(records)
    assert [len(body["records"]) for _, _, body in sent] == [LOAD_BATCH, 1] and got["loaded"] == 4
    with pytest.raises(AdaError) as e:
        m(ada, FIXTURES["load"]["request"]).load([])
    assert e.value.code == "E_VALIDATION"


def test_a_streamed_load_is_ndjson_to_ada_load_the_model_first_and_a_refusal_says_how_many_landed() -> None:
    model_id = FIXTURES["load"]["request"]["model_id"]
    sent: List[Tuple[str, Dict[str, str], bytes]] = []
    answers = [(200, {"data": {"model_id": model_id, "loaded": 2, "wins": 2, "vetoes": 0, "lines": 3}}),
               (400, {"error": {"code": "E_VALIDATION", "message": "line 3: this record: situation must be a non-empty object", "loaded": 1}})]

    def transport(url: str, headers: Any, body: Any, timeout: float) -> Tuple[int, bytes]:
        assert not isinstance(body, bytes), "a streamed body is an iterable of lines, sent as it is read"
        sent.append((url, dict(headers), b"".join(body)))
        status, answer = answers[len(sent) - 1]
        return status, json.dumps(answer).encode()

    ada = Ada(api_key="sk-test", transport=transport)
    rows: List[Any] = [{"situation": {"a": 1}, "outcome": "x"}, {"situation": {"a": 2}, "outcome": "y", "store": "vetoes"}]
    assert ada.model(model_id).stream(rows) == answers[0][1]["data"]
    url, headers, body = sent[0]
    assert (url, headers["content-type"]) == (f"{DEFAULT_URL}/ada/load", "application/x-ndjson")
    assert [json.loads(line) for line in body.decode().splitlines()] == [{"model_id": model_id}, *rows]
    with pytest.raises(AdaError) as e:
        ada.upload(model_id, iter(rows))
    assert (e.value.code, e.value.status, e.value.loaded) == ("E_VALIDATION", 400, 1) and sent[1][2] == body
