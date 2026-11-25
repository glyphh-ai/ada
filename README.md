# Ada

Ada for your app. A model lives in your Glyphh runtime — a typed spec and a precedent store, deployed by commit —
and your software calls it at the runtime's own address with a runtime key. Every answer is deterministic, nothing
of Glyphh's is between your app and your runtime, and Ada says when it does not know.

| | |
|---|---|
| [`sdk/typescript`](sdk/typescript) | `@glyphh-ai/ada` for TypeScript and JavaScript |
| [`sdk/python`](sdk/python) | `glyphh-ada` for Python |
| [`contract`](contract) | The JSON Schema of a run's `Result` and of an Ada model's answers, and the fixtures both SDKs are tested against |

In Claude, the `ada` plugin from the Glyphh marketplace (`claude plugin install ada@glyphh`) carries the skill:
how to build a model, deploy it and call it. Sign in, start your runtime and make keys on the [Glyphh
platform](https://app.glyphh.ai).

## Install

```bash
npm install @glyphh-ai/ada
pip install glyphh-ada
```

## Use

Two settings: your runtime's address and a runtime key (`GLYPHH_RUNTIME_URL`, `GLYPHH_RUNTIME_KEY`).

```ts
import { Ada } from "@glyphh-ai/ada";

const incidents = new Ada().model("incidents");
const r = await incidents.decide({ error: { what: { class: "TimeoutError", message: "payment gateway timed out", status: 504 } } });
if (r.status === "done") apply(r.data.fix);
else if (r.status === "needs_input") await incidents.resume(r.run_id, await askSomeone(r.prompt));
```

```python
from glyphh_ada import Ada

incidents = Ada().model("incidents")
r = incidents.decide({"error": {"what": {"class": "TimeoutError", "message": "payment gateway timed out", "status": 504}}})
if r.status == "done":
    apply(r.data["fix"])
```

## What a call is

`POST https://<machine>.glyphh.app/v1/runs` `{model, function, inputs, version?, clearance?}` with `Authorization:
Bearer <runtime key>`, answered with the first `Result`:

| Field | |
|---|---|
| `run_id` | The run; `resume(run_id, answer)` continues one that `needs_input` or `needs_approval` |
| `status` | `done`, `needs_input`, `needs_approval`, `failed`, `expired`, `cancelled` |
| `data` | What the function answered (`done`), or why it failed (`{code, reason}`) |
| `step`, `prompt` | What a waiting run asks |
| `seconds` | How long it ran |

An Ada model's functions (`record`, `query`, `facts`, `probabilities`, `learn`, `decide`) are what `spec.json` and
`model.py` make them; the SDKs name them for convenience, and `call(function, inputs)` reaches any function. `query`
answers `{sufficient, confidence, support, count, top, outcomes, receipts}`; `facts` the nearest precedents as fact
trees; `learn` makes the model weigh its roles by what its own precedents say and calibrate its probabilities.
`clearance` (0 to 1) is who asks: the model's secured layers are cut from the answer below their weight.

A refusal is `{"error": {"code", "message"}}`: `E_AUTH` (401, the key), `E_LIMIT` (402, the runtime is paused: its
plan's hours are spent this month), `E_MODEL_MISSING`, `E_RUN_MISSING`, `E_BAD_REQUEST`. The SDKs raise `AdaError`
with the code.

## License

MIT.


## License

Apache-2.0. See LICENSE and NOTICE. Copyright 2026 Chris Timmerman / Glyphh AI LLC. More about it: https://glyphh.ai/projects/ada-runtime
