# @glyphh-ai/ada

Ada for TypeScript and JavaScript. Your models live in your Glyphh runtime; this calls them at the runtime's address with a runtime key.

```bash
npm install @glyphh-ai/ada
```

```ts
import { Ada, AdaError } from "@glyphh-ai/ada";

const incidents = new Ada().model("incidents");   // GLYPHH_RUNTIME_URL, GLYPHH_RUNTIME_KEY; or new Ada({ url, key })

await incidents.record({ error: {...}, resolution: {...} }, "raise circuit breaker");

let r = await incidents.decide<{ fix: string }>({ error: { what: { class: "TimeoutError", message: "payment gateway timed out" } } });
if (r.status === "done") apply(r.data.fix);
else if (r.status === "needs_input") r = await incidents.resume(r.run_id, await askSomeone(r.prompt));

const bot = await incidents.query({...}, { clearance: 0.5 });   // a caller who may not see the secured layers
```

`ada.models()`, `ada.deploy(repo, commit)`, `ada.run(runId)`, `ada.log(runId)`, `ada.cancel(runId)`;
`model.call(fn, inputs, { version, clearance })` reaches any function. A refusal throws `AdaError` (`code`, `message`,
`status`): `E_AUTH`, `E_LIMIT` (the runtime is paused: its month's hours are spent), `E_MODEL_MISSING`,
`E_RUN_MISSING`, `E_BAD_REQUEST`, `E_UNREACHABLE`. The contract both SDKs are tested against is `../../contract`.
