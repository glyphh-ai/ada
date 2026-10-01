---
name: ada
description: Use Ada, Glyphh's typed language for AI work, through its ada_* tools - to build a model for a recurring kind of work, load data into it, and query it for answers explained as fact trees. Use it before repeating a decision this organization has made before (how to run something, how to fix a recurring error, which approach worked), whenever an outcome is known and worth keeping ("that worked", "that failed", "always/never do X when ..."), and whenever structured memory or classification would beat reasoning from scratch.
---

# Ada

Ada is a typed language for AI work. A model holds typed data; a query in
the same shape comes back as a fact tree: what matched, how closely, and
why, part by part. Graded experience is one use (what was done in which
situation, and whether it worked); classification and structured memory
are others. Ada cites the records it stands on and says when it does not
know. It runs on Glyphh's servers; these tools reach it as the signed-in
Glyphh member.

## Before acting on a recurring kind of decision

1. `ada_models` to find the model that fits, if you do not know its id.
2. `ada_query` with the situation. Act on the answer only when `act`
   is true. A single record is not consensus unless it matches the
   situation exactly.
3. `ada_facts` when you need to show why: each close record explained as
   a fact tree, per attribute its score, weight and share.

## Once the world has graded an outcome

- It worked: `ada_record` (situation, outcome).
- It failed or was rejected: `ada_veto`.

Record only outcomes the user or the world confirmed. Never grade your
own work as a win.

## How answers are decided

- `act` is true only when confidence is at least 0.7, three or more
  records stand behind the answer and no recorded failure vetoes it.
- A record that matches the situation exactly is the answer and acts
  alone.
- A query may use another word for a recorded value (`production` for
  `prod`): cloud models match those through a lexicon built when the
  record was written. Such a match is never exact, so it still needs
  three records.
- When `act` is false, read `reason` and ask the user or fall back. Do
  not act on `top` anyway.

## Describing a situation

Give only the attributes that should decide, as short canonical values:
`{"task": "deploy_service", "constraint": "canary_required"}`. Leave out
ids, paths, names and timestamps: they make unrelated situations look
alike. Use the same keys every time for the same kind of situation.

## Other tools

`ada_check` (how close the nearest win and failure of one outcome are),
`ada_calibrate` (fit the model's probabilities once it has 10+ wins),
`ada_records` (browse), and for org admins `ada_learn` (which keys
decide, learned from the wins), `ada_create_model`,
`ada_update_model`, `ada_delete_model`, `ada_edit_record`,
`ada_delete_record`, `ada_devices`.
