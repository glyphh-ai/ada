// The client against the contract's fixtures: every request as the runtime expects it, every answer read right,
// every refusal an AdaError with the runtime's code.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { Ada, AdaError } from "../dist/index.js";

const fixtures = JSON.parse(readFileSync(new URL("../../../contract/fixtures.json", import.meta.url), "utf8"));
const byName = (name) => fixtures.cases.find((c) => c.name === name);

function fake(...answers) {
  const calls = [];
  const transport = async (method, url, headers, body) => {
    calls.push({ method, url, headers, body: body === undefined ? undefined : JSON.parse(body) });
    const [status, answer] = answers.shift();
    return { status, body: JSON.stringify(answer) };
  };
  return { ada: new Ada({ url: "https://m1.glyphh.app/", key: "glyphh_rk_test", transport }), calls };
}

test("a decide that is done, with a clearance", async () => {
  const c = byName("a call that is done");
  const { ada, calls } = fake([200, c.response.body]);
  const r = await ada.model("incidents").decide(c.request.body.inputs.data, { clearance: 0.5 });
  assert.equal(r.status, "done");
  assert.equal(r.data.fix, "raise circuit breaker");
  assert.deepEqual(calls[0].body, c.request.body);
  assert.equal(calls[0].url, "https://m1.glyphh.app/v1/runs");
  assert.equal(calls[0].headers.authorization, "Bearer glyphh_rk_test");
});

test("a run that needs input is resumed", async () => {
  const c = byName("a call that needs input, then continued");
  const { ada, calls } = fake([200, c.response.body], [200, c.then.response.body]);
  const m = ada.model("incidents");
  const r = await m.decide(c.request.body.inputs.data);
  assert.equal(r.status, "needs_input");
  const done = await m.resume(r.run_id, "rotate certificate");
  assert.equal(done.data.fix, "rotate certificate");
  assert.equal(calls[1].url, "https://m1.glyphh.app/v1/runs/9b0d44/continue");
  assert.deepEqual(calls[1].body, { answer: "rotate certificate" });
});

test("a query, and the other functions name themselves", async () => {
  const c = byName("a query");
  const { ada, calls } = fake(...Array(7).fill([200, c.response.body]));
  const m = ada.model("incidents");
  const r = await m.query(c.request.body.inputs.data);
  assert.equal(r.data.top, "rotate certificate");
  await m.record({ a: 1 }, "fixed");
  await m.record({ a: 1 });
  await m.facts({ a: 1 }, { top: 2, clearance: 0.5 });
  await m.probabilities({ a: 1 });
  await m.learn();
  await m.learn({ sample: 50, clearance: 0.5 });
  assert.deepEqual(
    calls.map((k) => [k.body.function, k.body.inputs, k.body.clearance]),
    [
      ["query", { data: c.request.body.inputs.data }, undefined],
      ["record", { data: { a: 1 }, outcome: "fixed" }, undefined],
      ["record", { data: { a: 1 } }, undefined],
      ["facts", { data: { a: 1 }, top: 2 }, 0.5],
      ["probabilities", { data: { a: 1 } }, undefined],
      ["learn", {}, undefined],
      ["learn", { sample: 50 }, 0.5],
    ],
  );
});

for (const name of ["a wrong key", "a paused runtime", "a model that is not there"]) {
  test(`a refusal is an AdaError: ${name}`, async () => {
    const c = byName(name);
    const { ada } = fake([c.response.status, c.response.body]);
    await assert.rejects(ada.model(c.request.body.model).call("start"), (e) => {
      assert.ok(e instanceof AdaError);
      assert.equal(e.code, c.response.body.error.code);
      assert.equal(e.status, c.response.status);
      assert.equal(e.message, c.response.body.error.message);
      if (name === "a paused runtime") assert.deepEqual(e.detail, { limit: "runtime_hours" });
      return true;
    });
  });
}

test("the runtime and the rest of the api", async () => {
  const models = { models: [{ name: "incidents", active: 1 }] };
  const { ada, calls } = fake([200, models], [201, { model: "incidents", version: 2, commit_sha: "a".repeat(40) }], [200, { id: "r1" }], [200, { lines: [1] }], [200, byName("a call that is done").response.body]);
  assert.deepEqual(await ada.models(), models.models);
  assert.equal((await ada.deploy("octo/incidents", "a".repeat(40), "ghp_x")).version, 2);
  assert.deepEqual(await ada.run("r1"), { id: "r1" });
  assert.deepEqual(await ada.log("r1"), [1]);
  assert.equal((await ada.cancel("r1")).status, "done");
  assert.deepEqual(
    calls.map((k) => [k.method, k.url.replace("https://m1.glyphh.app", "")]),
    [["GET", "/v1/models"], ["POST", "/v1/models/deploy"], ["GET", "/v1/runs/r1"], ["GET", "/v1/runs/r1/log"], ["POST", "/v1/runs/r1/cancel"]],
  );
  assert.deepEqual(calls[1].body, { repo: "octo/incidents", commit: "a".repeat(40), token: "ghp_x" });
  assert.equal(calls[0].body, undefined);
});

test("an answer that is not json, no code, and a runtime that cannot be reached", async () => {
  const broken = new Ada({ url: "https://m1.glyphh.app", key: "k", transport: async () => ({ status: 502, body: "<html>" }) });
  await assert.rejects(broken.models(), (e) => e.code === "E_RUNTIME" && e.status === 502);
  const gone = new Ada({ url: "https://m1.glyphh.app", key: "k", transport: async () => { throw new Error("no route"); } });
  await assert.rejects(gone.models(), (e) => e.code === "E_UNREACHABLE");
  for (const [status, code] of [[401, "E_AUTH"], [402, "E_LIMIT"], [404, "E_NOT_FOUND"]]) {
    const a = new Ada({ url: "https://m1.glyphh.app", key: "k", transport: async () => ({ status, body: "{}" }) });
    await assert.rejects(a.models(), (e) => e.code === code);
  }
});

test("the address and key are needed", () => {
  delete process.env.GLYPHH_RUNTIME_URL;
  delete process.env.GLYPHH_RUNTIME_KEY;
  assert.throws(() => new Ada(), (e) => e.code === "E_CONFIG");
  process.env.GLYPHH_RUNTIME_URL = "https://m2.glyphh.app/";
  process.env.GLYPHH_RUNTIME_KEY = "glyphh_rk_env";
  assert.equal(new Ada().url, "https://m2.glyphh.app");
});
