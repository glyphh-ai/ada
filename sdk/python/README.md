# glyphh-ada

Ada for Python. An agent builds an Ada model: its types, its records, its rules. This package is what your software calls to use it. Every answer is typed, the same input always gives the same answer, and no language model is in the path.

Docs: [glyphh.ai/docs/ada](https://glyphh.ai/docs/ada). No dependencies on Python 3.11 and later. Python 3.9 or later.

```bash
pip install glyphh-ada
```

## Ask a model

```python
from glyphh_ada import Ada

ada = Ada()  # GLYPHH_API_KEY from the environment
tickets = ada.model("am_0123456789ab")

answer = tickets.query({"ticket": {"issue": {"component": "kubelet", "symptom": "crash_on_boot"}}})

if answer["act"]:
    route(answer["top"])
elif answer["reason"] == "unseen":
    ask_someone(answer["unseen"])  # ["ticket.issue.component=etcd"]
```

`act` is the one thing to branch on. When it is false, `reason` says why, and the nearest answer and the records behind it are still there. [When Ada acts](https://glyphh.ai/docs/ada-act) covers every case.

An answer is the JSON the server sent, as a dictionary. The package's `TypedDict`s (`Answer`, `Facts`, `History`, `Trend`, `Prediction`, `Edges`, `GqlResult`) say what is in each, so a type checker knows the keys.

## Record what happened

```python
tickets.record({"ticket": {"issue": {"component": "kubelet", "symptom": "crash_on_boot", "severity": 7}}}, "route_to_platform")
tickets.veto(situation, "route_to_network")  # this outcome failed here
tickets.record(situation, "route_to_platform", at="2026-01-03T10:00:00Z")  # when it happened, for past records
```

## Create a typed model

```python
model = ada.create_model("tickets", spec={
    "name": "tickets",
    "layers": [{"name": "ticket", "segments": [{"name": "issue", "roles": [
        {"name": "component", "type": "category"},
        {"name": "severity", "type": "number", "numeric_config": {"min_value": 0, "max_value": 10, "bin_width": 1}},
    ]}]}],
})
print(model.id)
```

Data that does not fit a role's type is refused before it is stored or scored.

## Everything a model does

| Method | Returns |
|---|---|
| `query(situation)` | The answer: `act`, `reason`, `top`, `unseen`, `receipts` |
| `facts(situation, top=3)` | The nearest situations as fact trees: which parts matched, and how closely |
| `check(situation, outcome)` | How close the nearest win and failure of one outcome are |
| `record(situation, outcome, at=None)`, `veto(...)` | The record's id, key and version |
| `history(thing)` | Every version of a thing, each with what changed |
| `trend(thing)` | How far it has moved, how fast, and each number's slope |
| `predict(thing, at=None)` | Its next version, and what the model's other things say of it |
| `edges(thing, level=None, top=None)` | Its nearest neighbours at a level, the change between versions, and its relations |
| `gql(query, args=None)` | One GQL statement: `FIND SIMILAR`, `LIST`, `COUNT`, `AGGREGATE`, `COMPARE`, `DETECT DRIFT`, `INTROSPECT`, `TREND`, `PREDICT`, `FOLLOW`, `PATH` |
| `call(name, args=None)` | A stored procedure |
| `procedures()`, `save_procedure(name, query)`, `delete_procedure(name)` | Stored procedures |
| `info()`, `update(...)`, `delete()` | The model itself |
| `records(...)`, `edit_record(id, ...)`, `delete_record(id)` | Its records |
| `calibrate()`, `learn()` | Fit its probabilities; learn a flat model's weights |
| `contract()` | The JSON Schema of every answer and of a typed model's data |

On the client: `ada.models()`, `ada.create_model(...)`, `ada.devices()`, `ada.contract()`, and `ada.op(name, **args)` for any operation by name.

## GQL and the graph

```python
near = tickets.gql("FIND SIMILAR TO $like WHERE outcome = $outcome LIMIT 5", {"like": situation, "outcome": "won"})
for match in near["matches"]:
    print(match["key"], match["score"])

impact = services.gql('FOLLOW glyph("db") IN DEPTH 3')  # everything that depends on db
```

## Errors

A refusal is an `AdaError` with a `code`, a `message` and the HTTP `status`.

```python
from glyphh_ada import AdaError

try:
    tickets.query({"ticket": {"issue": {"severity": "high"}}})
except AdaError as e:
    if e.code == "E_VALIDATION":
        print(e.message)  # situation: ticket.issue.severity: a numeric role takes a finite number
```

| Code | Means |
|---|---|
| `E_VALIDATION` | An argument, or data that does not fit the model's types |
| `E_NOT_FOUND` | No such model, record, thing or procedure |
| `E_FORBIDDEN` | For org admins, or above the reader's clearance |
| `E_FAILED_PRECONDITION` | The model cannot do that as it is |
| `E_UNAUTHENTICATED`, `E_PAYMENT` | The key, or the organization's plan or budget |
| `E_UNREACHABLE` | The request never got an answer |

## Options

```python
Ada(api_key=None, base_url=None, timeout=30.0, headers=None, transport=None)
```

`api_key` defaults to `GLYPHH_API_KEY`, `base_url` to `GLYPHH_URL` and then `https://api.glyphh.ai`. `transport` is a function `(url, headers, body, timeout) -> (status, body)`: give your own to use another HTTP library. The client is synchronous; call it from a thread in async code. Create a key on the Virtual Keys page of the [Glyphh console](https://platform.glyphh.ai). Every call is metered on your organization's plan.

## License

MIT
