/**
 * Customer churn with Ada: past accounts and how each term ended, then a verdict on a live account.
 *
 *   npm install @glyphh-ai/ada
 *   export GLYPHH_API_KEY=sk-...        # an org admin's key, to create the model
 */
import { Ada, AdaError, type Answer, type Spec } from "@glyphh-ai/ada";

const ada = new Ada();

// 1. An account, month by month. account_id says which account a record is about.
const SPEC: Spec = {
  name: "renewals",
  layers: [{ name: "account", segments: [
    { name: "identity", roles: [{ name: "account_id", type: "category", key_part: true, similarity_weight: 0 }] },
    { name: "profile", similarity_weight: 0.5, roles: [
      { name: "plan", type: "category", values: ["starter", "growth", "scale"] },
      { name: "industry", type: "category" },
    ] },
    { name: "usage", roles: [
      { name: "seats", type: "number", numeric_config: { min_value: 0, max_value: 200, bin_width: 10 } },
      { name: "logins_per_seat", type: "number", numeric_config: { min_value: 0, max_value: 10, bin_width: 1 } },
      { name: "open_tickets", type: "number", numeric_config: { min_value: 0, max_value: 10, bin_width: 1 } },
      { name: "features_used", type: "number", numeric_config: { min_value: 0, max_value: 12, bin_width: 1 } },
    ] },
  ] }],
};

type Usage = { seats: number; logins_per_seat: number; open_tickets: number; features_used: number };
type Account = { id?: string; plan: string; industry: string; usage: Usage };

const snapshot = (a: Account) => ({
  account: {
    ...(a.id ? { identity: { account_id: a.id } } : {}),
    profile: { plan: a.plan, industry: a.industry },
    usage: a.usage,
  },
});

// 2. Sixty past accounts, three monthly snapshots each, and how the term ended.
//    In your code this is a query on your warehouse.
function* past(seed = 11): Generator<{ account: Account; month: number; outcome: "renewed" | "churned" }> {
  let s = seed;
  const rnd = (n: number) => {  // a small seeded generator, so every run loads the same accounts
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) % n;
  };
  const plans = ["starter", "growth", "scale"], industries = ["saas", "retail", "fintech", "health"];
  for (let i = 0; i < 60; i++) {
    const churned = i % 3 === 0;
    const plan = plans[rnd(3)], industry = industries[rnd(4)], seats = 10 + rnd(15) * 10;
    const logins = 5 + rnd(4), features = 5 + rnd(6);
    for (let month = 0; month < 3; month++) {
      const slide = churned ? month : 0;
      yield {
        account: { id: `acct-${String(i).padStart(2, "0")}`, plan, industry, usage: {
          seats: Math.max(10, seats - slide * 10),
          logins_per_seat: Math.max(0, logins - slide * 2 - (churned ? rnd(2) : 0)),
          open_tickets: Math.min(10, (churned ? 2 + slide * 2 : 0) + rnd(2)),
          features_used: Math.max(1, features - slide * 2),
        } },
        month, outcome: churned ? "churned" : "renewed",
      };
    }
  }
}

const show = (title: string, value: unknown) => console.log(`\n--- ${title}\n${JSON.stringify(value, null, 2)}`);
const brief = (a: Answer) => ({ act: a.act, reason: a.reason, top: a.top, confidence: a.confidence, count: a.count, outcomes: a.outcomes, unseen: a.unseen });
const T0 = Date.UTC(2026, 0, 1) / 1000, MONTH = 30 * 86400;

const model = await ada.createModel({ name: "renewals", storage: "cloud", spec: SPEC });
console.log("model", model.id);

// 3. Record each snapshot with when it was taken. Records that share an account_id are versions of one account.
for (const { account, month, outcome } of past()) await model.record(snapshot(account), outcome, { at: T0 + month * MONTH });
const info = await model.info();
show("the model", { things: info.things, wins: info.wins });

// Fit the model's probabilities to its own records, once it has ten or more.
show("calibrated", await model.calibrate());

// 4. A live account, as it is today. No id: the question is "what happened to accounts like this".
const sliding: Account = { plan: "growth", industry: "saas", usage: { seats: 40, logins_per_seat: 2, open_tickets: 6, features_used: 3 } };
const healthy: Account = { plan: "growth", industry: "saas", usage: { seats: 60, logins_per_seat: 7, open_tickets: 0, features_used: 9 } };
show("an account that is sliding", brief(await model.query(snapshot(sliding))));
show("a healthy account", brief(await model.query(snapshot(healthy))));

// 5. Why: the nearest past account, role by role.
const why = (await model.facts(snapshot(sliding), { top: 1 })).wins[0];
const usage = why.tree.children![0].children!.find((c) => c.key === "usage")!;
show(`nearest past account (${why.outcome}, score ${why.score})`,
  usage.children!.map((c) => ({ role: c.key, today: c.query, then: c.record, match: c.match, score: c.score })));

// 6. Which churned accounts does it look like? One GQL statement, with parameters. Each match names its account.
const alike = await model.gql("FIND SIMILAR TO $account WHERE outcome = $outcome LIMIT 3", { account: snapshot(sliding), outcome: "churned" });
if (alike.statement !== "find") throw new Error("expected a find");
show("churned accounts it resembles", alike.matches.map((m) => ({ account: m.parts?.[0], score: m.score, versions: m.count })));

// 7. How one of those slid: its versions, its trend, and how far it drifted.
const lost = { account: { identity: { account_id: String(alike.matches[0].parts?.[0]) } } };
const history = await model.history(lost);
show(`${lost.account.identity.account_id}, month by month`, history.observations.map((o) => ({ version: o.version, usage: (o.data as ReturnType<typeof snapshot>).account.usage, changed: o.changes?.map((c) => c.path.split(".").pop()) })));
const trend = await model.trend(lost);
show("its trend", { drift: trend.drift, logins: trend.roles["account.usage.logins_per_seat"], tickets: trend.roles["account.usage.open_tickets"] });
const drift = await model.gql("DETECT DRIFT FROM glyph($first) TO glyph($latest) THRESHOLD 0.2",
  { first: `${lost.account.identity.account_id}@1`, latest: lost.account.identity.account_id });
if (drift.statement === "drift") show("first month to last", { drift: drift.drift, drifted: drift.drifted, changed: drift.changes.map((c) => `${c.path.split(".").pop()}: ${c.from} to ${c.to}`) });

// 8. The whole book in one line: what usage looks like at the end of a term, by how it ended.
const logins = await model.gql("AGGREGATE AVG account.usage.logins_per_seat GROUP BY outcome");
if (logins.statement === "aggregate") show("average logins per seat in the last month", Object.fromEntries(logins.groups.map((g) => [g.by.outcome, g.value])));

// 9. Save a question with the model, and call it by name from anywhere.
await model.saveProcedure("usage_by_plan", "AGGREGATE AVG account.usage.logins_per_seat WHERE account.profile.plan = $plan GROUP BY outcome",
  "Average logins per seat at the end of a term, for one plan, by how the term ended.");
const byPlan = await model.call("usage_by_plan", { plan: "growth" });
if (byPlan.statement === "aggregate") show("growth plan, logins per seat", Object.fromEntries(byPlan.groups.map((g) => [g.by.outcome, g.value])));

// 10. The function your success team's tooling calls.
async function risk(account: Account) {
  try {
    const a = await model.query(snapshot(account));
    if (!a.act) return { risk: "unknown", because: a.reason, new: a.unseen };
    return { risk: a.top === "churned" ? "high" : "low", because: `${Math.round(a.confidence * 100)}% of the vote of ${a.count} past accounts says ${a.top}` };
  } catch (e) {
    if (e instanceof AdaError && e.code === "E_RATE_LIMITED") return { risk: "unknown", because: `retry in ${e.retryAfter}s` };
    throw e;
  }
}
show("risk()", [await risk(sliding), await risk(healthy), await risk({ ...healthy, industry: "gaming" })]);

// 11. When the live account's term ends, record how it ended. The model has one more account to stand on.
const ended = await model.record(snapshot({ ...sliding, id: "acct-live-1" }), "churned");
show("recorded", { key: ended.key, version: ended.version });
