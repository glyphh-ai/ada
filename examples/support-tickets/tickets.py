"""Route support tickets with Ada: build a typed model from past tickets, then let software route new ones.

    pip install glyphh-ada
    export GLYPHH_API_KEY=sk-...        # an org admin's key, to create the model
    python tickets.py
"""
import json
import random

from glyphh_ada import Ada, AdaError

ada = Ada()

# 1. The shape of a ticket: every role has a type.
SPEC = {
    "name": "support-tickets",
    "layers": [{"name": "ticket", "segments": [
        {"name": "issue", "roles": [
            {"name": "product", "type": "category", "values": ["api", "billing", "dashboard", "mobile"]},
            {"name": "component", "type": "category"},
            {"name": "severity", "type": "number", "numeric_config": {"min_value": 1, "max_value": 5, "bin_width": 1}},
            {"name": "plan", "type": "category", "values": ["free", "pro", "enterprise"]},
        ]},
        {"name": "text", "similarity_weight": 0.5, "roles": [
            {"name": "summary", "type": "text", "text_encoding": "ngrams"},
        ]},
    ]}],
}


def ticket(product, component, severity, plan, summary=None):
    issue = {"product": product, "component": component, "severity": severity, "plan": plan}
    return {"ticket": {"issue": issue, **({"text": {"summary": summary}} if summary else {})}}


# 2. Past tickets, and the team that resolved each. In your code this is a query on your helpdesk.
COMPONENTS = {
    "webhooks": ("api", "platform", ["webhook deliveries failing", "webhooks arrive twice", "webhook signature rejected"]),
    "rate-limits": ("api", "platform", ["hitting 429 too early", "rate limit not resetting", "burst limit unclear"]),
    "invoices": ("billing", "billing", ["invoice shows wrong total", "missing invoice for March", "invoice not emailed"]),
    "refunds": ("billing", "billing", ["refund not received", "double charge needs refund", "partial refund request"]),
    "charts": ("dashboard", "frontend", ["chart will not load", "chart shows stale data", "chart legend overlaps"]),
    "exports": ("dashboard", "frontend", ["csv export times out", "export missing columns", "export encoding broken"]),
    "push": ("mobile", "mobile", ["push notifications stopped", "push arrives late", "duplicate push alerts"]),
    "login": ("mobile", "identity", ["cannot log in on phone", "login loop after update", "biometric login fails"]),
}


def history(n=96, seed=7):
    rng = random.Random(seed)
    for i in range(n):
        component = list(COMPONENTS)[i % len(COMPONENTS)]
        product, team, summaries = COMPONENTS[component]
        yield ticket(product, component, rng.randint(1, 5), rng.choice(["free", "pro", "enterprise"]), rng.choice(summaries)), team, 1_750_000_000 + i * 3600


def show(title, value):
    print(f"\n--- {title}\n{json.dumps(value, indent=2)}")


def main():
    model = ada.create_model("support-tickets", spec=SPEC)
    print("model", model.id)

    # 3. Record what happened: this ticket was resolved by this team, at this time.
    for situation, team, when in history():
        model.record(situation, team, at=when)
    info = model.info()
    show("the model", {k: info[k] for k in ("name", "wins", "vetoes")})

    # 4. A new ticket. Branch on `act`.
    new = ticket("api", "webhooks", 4, "pro", "webhook delivery fails with a 500")
    answer = model.query(new)
    show("a ticket like ones on record", {k: answer[k] for k in ("act", "reason", "top", "confidence", "count", "unseen")})

    # 5. Something the model was never shown: it says so, and still names the nearest answer.
    odd = ticket("api", "sso", 3, "enterprise", "saml login fails for one tenant")
    unknown = model.query(odd)
    show("a component no ticket ever had", {k: unknown[k] for k in ("act", "reason", "top", "unseen")})

    # 6. A person routes those; record what they decided. After three, the model acts on its own.
    for severity in (3, 4, 2):
        model.record(ticket("api", "sso", severity, "enterprise", "saml login fails for one tenant"), "identity")
    learned = model.query(odd)
    show("the same ticket, after three were routed by hand", {k: learned[k] for k in ("act", "reason", "top", "confidence")})

    # 7. A route that was wrong: record the failure. The model stops making that call for this situation.
    bounced = ticket("billing", "refunds", 5, "enterprise", "double charge needs refund")
    before = model.query(bounced)
    model.veto(bounced, before["top"])
    after = model.query(bounced)
    show("a route that bounced", {"before": {k: before[k] for k in ("act", "top")}, "after": {k: after[k] for k in ("act", "reason", "top", "veto_match")}})

    # 8. Why: the nearest ticket on record, part by part.
    why = model.facts(new, top=1)["wins"][0]
    leaves = [{"role": leaf["key"], "asked": leaf.get("query"), "on_record": leaf.get("record"), "match": leaf.get("match"), "score": leaf["score"]}
              for segment in why["tree"]["children"][0]["children"] for leaf in segment["children"]]
    show("why the first ticket went to " + why["outcome"], {"score": why["score"], "recorded": why["count"], "parts": leaves})

    # 9. Which roles decide? Learn it from the tickets themselves (org admins).
    fit = model.learn()
    show("learned", {"weights": fit["weights"], "held_out_accuracy": fit["held_out_accuracy"]})

    # 10. A question you ask every day, saved with the model and called by name.
    model.save_procedure("load_by_team", "AGGREGATE COUNT WHERE store = wins AND ticket.issue.severity >= $severity GROUP BY outcome",
                         "How many tickets at or above a severity each team resolved.")
    load = model.call("load_by_team", {"severity": 4})
    show("tickets at severity 4 or 5, by team", {g["by"]["outcome"]: g["value"] for g in load["groups"]})

    # 11. The function your helpdesk calls.
    def route(new_ticket):
        try:
            answer = model.query(new_ticket)
        except AdaError as e:
            if e.code == "E_RATE_LIMITED":
                return {"queue": "triage", "because": f"retry in {e.retry_after}s"}
            raise
        if answer["act"]:
            return {"queue": answer["top"], "because": f"{answer['count']} tickets like it, confidence {answer['confidence']}"}
        return {"queue": "triage", "because": answer["reason"], "suggest": answer["top"], "new": answer["unseen"]}

    show("route()", [route(new), route(ticket("dashboard", "charts", 2, "free")), route(ticket("mobile", "widgets", 1, "free"))])
    return model


if __name__ == "__main__":
    main()
