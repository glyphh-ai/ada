// The client against what the server returns: contract/fixtures.json holds real request and response pairs.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { Ada, AdaError, DEFAULT_URL } from "../dist/index.js";

const fixtures = JSON.parse(readFileSync(new URL("../../../contract/fixtures.json", import.meta.url), "utf8"));
const STATUS = { E_VALIDATION: 400, E_UNAUTHENTICATED: 401, E_FORBIDDEN: 403, E_NOT_FOUND: 404, E_FAILED_PRECONDITION: 409 };

/** An Ada whose fetch answers with one fixture's response, and keeps what it was sent. */
function replaying(name, options = {}) {
  const sent = [];
  const { response } = fixtures[name];
  const ada = new Ada({
    apiKey: "sk-test",
    ...options,
    fetch: async (url, init) => {
      sent.push({ url, headers: init.headers, body: JSON.parse(init.body) });
      return new Response(JSON.stringify(response), { status: response.error ? STATUS[response.error.code] ?? 400 : 200 });
    },
  });
  return { ada, sent, expected: fixtures[name] };
}

const DEALS = fixtures.record.request.model_id;
const d1 = { deal: { identity: { deal_id: "d1" } } };

/** Each call the client makes, by the fixture it should reproduce. */
const calls = {
  create_model: (ada, r) => ada.createModel({ name: r.name, storage: r.storage, spec: r.spec }).then((m) => m.created),
  record: (ada, r) => ada.model(r.model_id).record(r.situation, r.outcome),
  models: (ada) => ada.models().then((models) => ({ models })),
  models_one: (ada, r) => ada.model(r.model_id).info().then((m) => ({ models: [m] })),
  query_act: (ada, r) => ada.model(r.model_id).query(r.situation),
  query_unseen: (ada, r) => ada.model(r.model_id).query(r.situation),
  update_model: (ada, r) => ada.model(r.model_id).update({ scale: r.scale }),
  facts: (ada, r) => ada.model(r.model_id).facts(r.situation, { top: r.top }),
  check: (ada, r) => ada.model(r.model_id).check(r.situation, r.outcome),
  history: (ada, r) => ada.model(r.model_id).history(r.situation),
  trend: (ada, r) => ada.model(r.model_id).trend(r.situation),
  predict: (ada, r) => ada.model(r.model_id).predict(r.situation),
  edges: (ada, r) => ada.model(r.model_id).edges(r.situation, { level: r.level }),
  gql_find: (ada, r) => ada.model(r.model_id).gql(r.query),
  gql_count: (ada, r) => ada.model(r.model_id).gql(r.query, r.args),
  gql_compare: (ada, r) => ada.model(r.model_id).gql(r.query),
  gql_follow: (ada, r) => ada.model(r.model_id).gql(r.query),
  gql_path: (ada, r) => ada.model(r.model_id).gql(r.query),
  save_procedure: (ada, r) => ada.model(r.model_id).saveProcedure(r.name, r.query, r.description),
  procedures: (ada, r) => ada.model(r.model_id).procedures().then((procedures) => ({ model_id: r.model_id, procedures })),
  call: (ada, r) => ada.model(r.model_id).call(r.name, r.args),
  records: (ada, r) => ada.model(r.model_id).records({ limit: r.limit }),
  contract_model: (ada, r) => ada.model(r.model_id).contract(),
  delete_model: (ada, r) => ada.model(r.model_id).delete(),
  create_accounts: (ada, r) => ada.createModel({ name: r.name, storage: r.storage, spec: r.spec }).then((m) => m.created),
};

for (const [name, run] of Object.entries(calls)) {
  test(`${name}: sends what the server takes and returns what it answered`, async () => {
    const { ada, sent, expected } = replaying(name);
    const got = await run(ada, expected.request);
    assert.deepEqual(sent.length, 1);
    assert.deepEqual(sent[0].body, expected.request, "the request body is the op and its arguments, nothing else");
    assert.equal(sent[0].url, `${DEFAULT_URL}/ada`);
    assert.equal(sent[0].headers["x-glyphh-api-key"], "Bearer sk-test");
    assert.deepEqual(got, expected.response.data);
  });
}

test("every fixture that is not an error has a call here", () => {
  const untested = Object.keys(fixtures).filter((n) => !n.startsWith("error_") && !(n in calls));
  assert.deepEqual(untested, []);
});

for (const [name, code, status] of [["error_validation", "E_VALIDATION", 400], ["error_not_found", "E_NOT_FOUND", 404], ["error_forbidden", "E_FORBIDDEN", 403]]) {
  test(`${name}: a refusal is an AdaError that says which kind`, async () => {
    const { ada, expected } = replaying(name);
    const r = expected.request;
    const call = r.op === "query" ? ada.model(r.model_id).query(r.situation) : ada.model(r.model_id).deleteProcedure(r.name);
    await assert.rejects(call, (e) => {
      assert.ok(e instanceof AdaError);
      assert.deepEqual([e.code, e.status, e.message], [code, status, expected.response.error.message]);
      return true;
    });
  });
}

test("an answer is typed data a program branches on", async () => {
  const acted = await replaying("query_act").ada.model(DEALS).query({});
  assert.deepEqual([acted.act, acted.reason, acted.top, acted.unseen], [true, "reflex", "open", []]);
  const refused = await replaying("query_unseen").ada.model(DEALS).query({});
  assert.equal(refused.act, false);
  assert.ok(["unseen", "insufficient"].includes(refused.reason) && refused.unseen.includes("deal.metrics.meetings=19"),
    "a refusal says why, and what it was never shown");
  const followed = await replaying("gql_follow").ada.model(DEALS).gql("");
  assert.equal(followed.statement, "follow");
  assert.deepEqual(followed.nodes.map((n) => n.hops).sort(), [1, 1, 2]);
  const ahead = await replaying("predict").ada.model(DEALS).predict(d1);
  assert.equal(ahead.roles["deal.metrics.meetings"].basis, "trend");
});

test("options: the key and the URL come from the environment, and undefined arguments are not sent", async () => {
  const before = { ...process.env };
  delete process.env.GLYPHH_API_KEY;
  assert.throws(() => new Ada(), (e) => e instanceof AdaError && e.code === "E_UNAUTHENTICATED");
  process.env.GLYPHH_API_KEY = "sk-env";
  process.env.GLYPHH_URL = "https://ada.example/";
  const sent = [];
  const ada = new Ada({ headers: { "x-trace": "1" }, fetch: async (url, init) => { sent.push({ url, init }); return new Response('{"data": {"models": []}}'); } });
  await ada.models();
  await ada.model("am_0123456789ab").query({ a: 1 }, { weights: undefined });
  assert.equal(sent[0].url, "https://ada.example/ada");
  assert.deepEqual([sent[0].init.headers["x-glyphh-api-key"], sent[0].init.headers["x-trace"]], ["Bearer sk-env", "1"]);
  assert.deepEqual(JSON.parse(sent[1].init.body), { op: "query", model_id: "am_0123456789ab", situation: { a: 1 } });
  process.env = before;
});

test("a server that cannot be reached, or answers without a body, is an AdaError too", async () => {
  const down = new Ada({ apiKey: "k", fetch: async () => { throw new Error("ECONNREFUSED"); } });
  await assert.rejects(down.models(), (e) => e instanceof AdaError && e.code === "E_UNREACHABLE" && e.status === 0);
  const broke = new Ada({ apiKey: "k", fetch: async () => new Response("no plan", { status: 402 }) });
  await assert.rejects(broke.models(), (e) => e instanceof AdaError && e.code === "E_PAYMENT" && e.status === 402);
  const unsigned = new Ada({ apiKey: "k", fetch: async () => new Response('{"detail": "bad key"}', { status: 401 }) });
  await assert.rejects(unsigned.models(), (e) => e instanceof AdaError && e.code === "E_UNAUTHENTICATED" && e.message === "bad key");
});

test("the gateway's own refusals get Ada's codes, and a rate limit says how long to wait", async () => {
  const answering = (status, body, headers = {}) => new Ada({ apiKey: "k", fetch: async () => new Response(body, { status, headers }) });
  const unknown = '{"error":{"message":"Authentication Error, Invalid proxy server token passed.","type":"token_not_found_in_db","param":"key","code":"401"}}';
  await assert.rejects(answering(401, unknown).models(),
    (e) => e instanceof AdaError && e.code === "E_UNAUTHENTICATED" && e.status === 401 && e.retryAfter === null && e.message.includes("Invalid proxy server token"));
  const limited = '{"detail":"Rate limit exceeded for team: t. Limit type: requests. Current limit: 60, Remaining: 0."}';
  await assert.rejects(answering(429, limited, { "retry-after": "60" }).models(),
    (e) => e instanceof AdaError && e.code === "E_RATE_LIMITED" && e.retryAfter === 60 && e.message.startsWith("Rate limit exceeded"));
  await assert.rejects(answering(429, '{"error": {"code": "E_RATE_LIMITED", "message": "slow down"}}', { "retry-after": "soon" }).models(),
    (e) => e.code === "E_RATE_LIMITED" && e.retryAfter === null);
  await assert.rejects(answering(403, '{"error": {"code": "403", "message": "no"}}').models(), (e) => e.code === "E_FORBIDDEN");
});
