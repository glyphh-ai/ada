# @glyphh-ai/ada

Ada for TypeScript and JavaScript. An agent builds an Ada model: its types, its records, its rules. This package is what your software calls to use it. Every answer is typed, the same input always gives the same answer, and no language model is in the path.

Docs: [docs.glyphh.ai/ada](https://docs.glyphh.ai/ada). No dependencies. Node 18 or later, or any runtime with `fetch`.

```bash
npm install @glyphh-ai/ada
```

## Ask a model

```ts
import { Ada } from "@glyphh-ai/ada";

const ada = new Ada({ apiKey: process.env.GLYPHH_API_KEY });
const tickets = ada.model("am_0123456789ab");

const answer = await tickets.query({ ticket: { issue: { component: "kubelet", symptom: "crash_on_boot" } } });

if (answer.act) {
  route(answer.top);
} else if (answer.reason === "unseen") {
  askSomeone(answer.unseen); // ["ticket.issue.component=etcd"]
}
```

`act` is the one thing to branch on. When it is false, `reason` says why, and the nearest answer and the records behind it are still there. [When Ada acts](https://docs.glyphh.ai/ada-act) covers every case.

## Record what happened

```ts
await tickets.record({ ticket: { issue: { component: "kubelet", symptom: "crash_on_boot", severity: 7 } } }, "route_to_platform");
await tickets.veto(situation, "route_to_network"); // this outcome failed here
await tickets.record(situation, "route_to_platform", { at: "2026-01-03T10:00:00Z" }); // when it happened, for past records
await tickets.load(history.map((h) => ({ situation: h.situation, outcome: h.outcome, at: h.when }))); // the seed, all or none
await tickets.stream(readRecords("tickets.ndjson")); // a file, streamed; a refusal names the line and says how many landed (error.loaded)
```

## Create a typed model

```ts
const model = await ada.createModel({
  name: "tickets",
  storage: "cloud",
  spec: {
    name: "tickets",
    layers: [{ name: "ticket", segments: [{ name: "issue", roles: [
      { name: "component", type: "category" },
      { name: "severity", type: "number", numeric_config: { min_value: 0, max_value: 10, bin_width: 1 } },
    ] }] }],
  },
});
```

Data that does not fit a role's type is refused before it is stored or scored.

## Everything a model does

| Method | Returns |
|---|---|
| `query(situation)` | The answer: `act`, `reason`, `top`, `unseen`, `receipts` |
| `facts(situation, { top })` | The nearest situations as fact trees: which parts matched, and how closely |
| `check(situation, outcome)` | How close the nearest win and failure of one outcome are |
| `record(situation, outcome, { at })`, `veto(...)` | The record's id, key and version |
| `load(records)` | Up to 500 records in one call, all or none; a longer list goes in turns. `{ loaded, wins, vetoes }` |
| `stream(records)` | A file of records as NDJSON to `POST /ada/load`, written in batches as it arrives; an async iterable is sent as it is read |
| `history(thing)` | Every version of a thing, each with what changed |
| `trend(thing)` | How far it has moved, how fast, and each number's slope |
| `predict(thing, { at })` | Its next version, and what the model's other things say of it |
| `edges(thing, { level, top })` | Its nearest neighbours at a level, the change between versions, and its relations |
| `gql(query, args)` | One GQL statement: `FIND SIMILAR`, `LIST`, `COUNT`, `AGGREGATE`, `COMPARE`, `DETECT DRIFT`, `INTROSPECT`, `TREND`, `PREDICT`, `FOLLOW`, `PATH` |
| `call(name, args)` | A stored procedure |
| `procedures()`, `saveProcedure(name, query)`, `deleteProcedure(name)` | Stored procedures |
| `info()`, `update(change)`, `delete()` | The model itself |
| `records(options)`, `editRecord(id, change)`, `deleteRecord(id)` | Its records |
| `calibrate()`, `learn()` | Fit its probabilities; learn which keys or roles decide |
| `contract()` | The JSON Schema of every answer and of a typed model's data |

On the client: `ada.models()`, `ada.createModel(model)`, `ada.devices()`, `ada.contract()`, and `ada.op(name, args)` for any operation by name.

## GQL and the graph

```ts
const near = await tickets.gql('FIND SIMILAR TO $like WHERE outcome = $outcome LIMIT 5', { like: situation, outcome: "won" });
if (near.statement === "find") near.matches.forEach((m) => console.log(m.key, m.score));

const impact = await services.gql('FOLLOW glyph("db") IN DEPTH 3'); // everything that depends on db
```

`gql` returns a union. Narrow it on `statement` and the fields are typed.

## Worked examples

Two programs that go from an empty model to a function your software calls, each with a step-by-step guide: [route support tickets](https://docs.glyphh.ai/ada-guide-tickets) (Python) and [customer churn](https://docs.glyphh.ai/ada-guide-churn) (TypeScript). The source is in [`examples/`](https://github.com/glyphh-ai/ada/tree/main/examples).

## Errors

A refusal is an `AdaError` with a `code`, the HTTP `status`, and `retryAfter` when Ada says how long to wait.

```ts
import { AdaError } from "@glyphh-ai/ada";

try {
  await tickets.query({ ticket: { issue: { severity: "high" } } });
} catch (e) {
  if (e instanceof AdaError && e.code === "E_VALIDATION") console.error(e.message);
  // situation: ticket.issue.severity: a numeric role takes a finite number
}
```

| Code | Means |
|---|---|
| `E_VALIDATION` | An argument, or data that does not fit the model's types |
| `E_NOT_FOUND` | No such model, record, thing or procedure |
| `E_FORBIDDEN` | For org admins, or above the reader's clearance |
| `E_FAILED_PRECONDITION` | The model cannot do that as it is |
| `E_UNAUTHENTICATED`, `E_PAYMENT` | The key, or the organization's plan or budget |
| `E_RATE_LIMITED` | Too many calls for the plan. `retryAfter` says how many seconds to wait |
| `E_UNREACHABLE` | The request never got an answer |

## Options

```ts
new Ada({ apiKey, baseUrl, fetch, headers });
```

`apiKey` defaults to `GLYPHH_API_KEY`, `baseUrl` to `GLYPHH_URL` and then `https://api.glyphh.ai`. Create a key on the Virtual Keys page of the [Glyphh console](https://platform.glyphh.ai). Every call is metered on your organization's plan.

## License

MIT
