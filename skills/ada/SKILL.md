---
name: ada
description: Use Ada, Glyphh's typed language for AI work, through its ada_* tools - to build a typed model for a recurring kind of work, load records into it, and query it for answers explained as fact trees. Use it before repeating a decision this organization has made before (how to run something, how to fix a recurring error, which approach worked), whenever an outcome is known and worth keeping ("that worked", "that failed", "always/never do X when ..."), when things relate to each other or change over time and that should be asked about (what depends on this, what changed, where is it heading), and when building a model that software will call with no agent in the path.
---

# Ada

Ada is a typed language for AI work. A model holds typed records; a
question in the same shape comes back as a fact tree: what matched, how
closely, and why, part by part. The same question always gets the same
answer, no language model is in the path, and Ada says when it does not
know. It runs on Glyphh's servers; these tools reach it as the signed-in
Glyphh member.

There are two callers. You build a model: choose its types, load its
records, test it. Software then calls it (the `@glyphh-ai/ada` and
`glyphh-ada` packages, or `POST /ada`) and branches on the answer. So
build models a program can trust: typed, with the same shape every time.

## Before acting on a recurring kind of decision

1. `ada_models` to find the model that fits, if you do not know its id.
2. `ada_query` with the situation. Act on the answer only when `act`
   is true.
3. `ada_facts` when you need to show why: each close situation as a
   fact tree, per part its score, weight and share.

## Once the world has graded an outcome

- It worked: `ada_record` (situation, outcome).
- It failed or was rejected: `ada_veto`.
- Loading something that happened earlier: pass `at`, when it happened.

Record only outcomes the user or the world confirmed. Never grade your
own work as a win.

## How answers are decided

`act` is true only when `reason` is `reflex`: confidence of 0.7 or
more, three or more records behind the answer and no recorded failure
against it, or the same situation on record. Otherwise `reason` says
why not:

- `unseen`: the situation carries a key or a value no win has recorded.
  `unseen` lists them (`target=main`). The nearest answer is still in
  `top` and `receipts`, but nothing on record covers this case.
- `insufficient`: nothing on record is close enough to answer from.
- `uncalibrated`: fewer than three records stand behind the answer.
- `mismatch`: the close records do not agree with the situation on the
  keys the model says must match.
- `vetoed`: a recorded failure stands against it.
- `low_confidence`: the vote is split between outcomes.

When `act` is false, ask the user or fall back. Do not act on `top`
anyway. Two different values score 0, never "a little alike". A query
may use another word for a recorded value (`production` for `prod`), in
a flat model's top-level keys and a typed model's open categories; such
a match is never exact.

## Building a model

Prefer a typed model: `ada_create_model` with a `spec` of layers,
segments and roles, each role with a type.

- `category` (one value, same or not; `values` lists what it may hold),
  `set`, `number` (a `numeric_config` scale, near numbers score near),
  `text` (words, or n-grams so another spelling is near), `boolean`,
  `time`, and `ref` (the key of another thing: a relation).
- Roles marked `key_part` are a thing's identity. Records that share
  them are versions of one thing, and the model answers from the newest.
- A segment with `"together": true` scores its roles jointly: use it
  when the roles decide only in combination.
- A category that lists its `values` is closed: nothing else is
  accepted. One that lists none is open, and learns other ways its
  values are said.
- A spec is set before the first record. After that only its
  similarity weights change, by `ada_learn` or by hand.
- Data that does not fit is refused with the path of the role that was
  wrong. `ada_contract` with the model's id returns the JSON Schema of
  its data and of every answer.

A flat model (no spec) takes any JSON. Give it only the attributes that
should decide, as short canonical values:
`{"task": "deploy_service", "constraint": "canary_required"}`. Leave out
ids, paths, names and timestamps, and use the same keys every time.

## Modelling: the data is the model

Ada compares exactly, by type. The shape you give a record is the
model, and the fact tree shows at once when the shape is wrong. Learned
the hard way; follow these before the first record:

- **An absent value is not a zero.** Record only what a thing has. A
  field set to 0 on every record agrees with every question, and the
  vote fills with records that share nothing real. Twelve hue bins of
  which ten are 0 made every image look alike.
- **A histogram or a profile is a set, not a row of numbers.** Encode
  each part as tokens by its size (`orange1 ... orange8`, `red1`) in a
  `set` role: two things then score by what they share over what
  either has. A row of independent numbers scores a 1% trace against a
  missing value as 0 and rewards agreeing on the small bins.
- **Drop traces before you compare.** A 1% presence should not be in
  the record or the question. Threshold on the way in, the same way
  both times.
- **A number needs the right scale.** `bin_width` is the difference
  that counts as "near": one step apart scores 1 - 1/bins. Too fine and
  nothing is near; too coarse and everything is.
- **A category that may grow is open** (no `values`): a new value is
  `unseen`, the model says so, and a person decides. A category that
  must not grow is closed: list its `values`.
- **Identity is a key part with weight 0**, and a thing that changes is
  recorded again under the same key parts: versions, not new things.
  Ask about the thing by its key parts for history, trend and predict;
  leave them out to ask what the thing is like.
- **Roles that only mean something together go in one segment with
  `together: true`** (door open and alarm armed).
- **Weigh what decides.** Give identity, timestamps and noise weight 0.
  Once there are 10 or more wins, `ada_calibrate` fits the vote and
  `ada_learn` finds the weights from the records; read what it learned,
  a role at 0 no longer guards `unseen`.
- **Read the fact tree when an answer is off.** It names the role, both
  values and the score. A wrong answer with the right nearest records
  is a weighting or `together` problem; wrong nearest records are a
  representation problem; `unseen` on something that is not new is a
  threshold or a closed category.
- **Test with what it must refuse** as well as what it must answer, and
  iterate the spec before loading everything: a spec is fixed after the
  first record, bar its weights.

## Time, likeness and the graph

For a thing in a model with key parts:

- `ada_history`: every version, each with what changed.
- `ada_trend`: how far it has moved, how fast, each number's slope.
- `ada_predict`: its next version, each role with what it stands on.
- `ada_edges`: the things nearest it at a level (the whole, a layer, a
  segment, a role), the change between its versions, and its relations
  out and in.

`ada_gql` runs one statement: `FIND SIMILAR TO`, `LIST`, `COUNT`,
`AGGREGATE`, `COMPARE`, `DETECT DRIFT`, `INTROSPECT`, `TREND`,
`PREDICT`, and over `ref` roles `FOLLOW glyph("db") IN DEPTH 3` (what
depends on db) and `PATH FROM glyph("a") TO glyph("b")`. Name a thing
with `glyph("key")`, a version with `glyph("key@2")`. Pass values as
`$name` with `args`, never by building the statement's text.

A statement worth running again is a stored procedure:
`ada_save_procedure`, then `ada_call` with `args`. `ada_procedures`
lists them.

## Other tools

`ada_check` (how close the nearest win and failure of one outcome are),
`ada_calibrate` (fit the model's probabilities once it has 10+ wins),
`ada_records` (browse), and for org admins `ada_learn` (which keys or
roles decide, learned from the model's wins and saved: a typed model's
role weights go into its spec), `ada_update_model`,
`ada_delete_model`, `ada_edit_record`, `ada_delete_record`,
`ada_delete_procedure`, `ada_devices`.
