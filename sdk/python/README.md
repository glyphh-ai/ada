# glyphh-ada

Ada for Python. Your models live in your Glyphh runtime; this calls them at the runtime's address with a runtime key.

```bash
pip install glyphh-ada
```

```python
from glyphh_ada import Ada, AdaError

incidents = Ada().model("incidents")   # GLYPHH_RUNTIME_URL, GLYPHH_RUNTIME_KEY; or Ada(url, key)

incidents.record({"error": {...}, "resolution": {...}}, outcome="raise circuit breaker")

r = incidents.decide({"error": {"what": {"class": "TimeoutError", "message": "payment gateway timed out"}}})
if r.done:
    apply(r.data["fix"])
elif r.waiting:                        # needs_input or needs_approval: r.prompt says what
    r = incidents.resume(r.run_id, ask_someone(r.prompt))

bot = incidents.query({...}, clearance=0.5)   # a caller who may not see the secured layers
```

`Ada.models()`, `Ada.deploy(repo, commit)`, `Ada.run(run_id)`, `Ada.log(run_id)`, `Ada.cancel(run_id)`;
`Model.call(function, inputs, version=, clearance=)` reaches any function. A refusal raises `AdaError(code, message,
status)`: `E_AUTH`, `E_LIMIT` (the runtime is paused: its month's hours are spent), `E_MODEL_MISSING`,
`E_RUN_MISSING`, `E_BAD_REQUEST`, `E_UNREACHABLE`. The contract both SDKs are tested against is `../../contract`.
