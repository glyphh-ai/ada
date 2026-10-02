# Glyphh Ada

[Glyphh](https://glyphh.ai) Ada is a typed language for AI work: typed models your agents build, and your software then calls with no agent in the path. Every answer is a fact tree, the same question always gets the same answer, and Ada says when it does not know.

This repository holds the agent skill and Claude Code plugin, and the SDKs:

| | |
|---|---|
| [`skills/ada`](skills/ada/SKILL.md) | The skill: when to query Ada, how to build a model, when to record |
| [`sdk/typescript`](sdk/typescript) | `@glyphh-ai/ada` for TypeScript and JavaScript |
| [`sdk/python`](sdk/python) | `glyphh-ada` for Python |
| [`contract`](contract) | The JSON Schema of every answer, and the request and response pairs both SDKs are tested against |

Docs: [Ada models](https://glyphh.ai/docs/ada), [Install Ada](https://glyphh.ai/docs/ada-install), [Ada tools and API](https://glyphh.ai/docs/ada-tools).

Sign in, manage models and create API keys in the [Glyphh app](https://platform.glyphh.ai).

## Install

Claude Code plugin (the skill and the Ada connection):

```bash
claude plugin marketplace add glyphh-ai/ada
claude plugin install ada@glyphh-ai
```

Then run `/mcp` in Claude Code, pick `ada` and choose Authenticate to sign in with your Glyphh account.

Other agents via [skills.sh](https://skills.sh):

```bash
npx skills add glyphh-ai/ada --skill ada
```

This installs the skill in the current project; add `-g` to install it globally. Then connect the agent to Ada:

- Sign-in: add `https://api.glyphh.ai/mcp` as a remote MCP server or custom connector and sign in with your Glyphh account.
- API key (CI, headless): send the header `x-glyphh-api-key: Bearer <key>` to the same URL. In Claude Code:

```bash
claude mcp add --transport http ada https://api.glyphh.ai/mcp --header "x-glyphh-api-key: Bearer $GLYPHH_API_KEY"
```

## Use

Ask your agent for the work in plain words:

> Use Ada to remember how we fixed this deploy failure, and check it before the next deploy.

In Claude Code you can invoke the skill explicitly with `/ada:ada`.

| Skill | Purpose |
|-------|---------|
| `ada` | Decide when to query Ada, how to build a typed model, and when to record a win or a failure |

## SDKs

Software calls a model through `POST /ada` with an API key. The SDKs wrap it, typed:

```bash
npm install @glyphh-ai/ada
pip install glyphh-ada
```

```ts
import { Ada } from "@glyphh-ai/ada";

const tickets = new Ada().model("am_0123456789ab");
const answer = await tickets.query({ ticket: { issue: { component: "kubelet" } } });
if (answer.act) route(answer.top);
```

```python
from glyphh_ada import Ada

tickets = Ada().model("am_0123456789ab")
answer = tickets.query({"ticket": {"issue": {"component": "kubelet"}}})
if answer["act"]:
    route(answer["top"])
```

Each has its own README: [TypeScript](sdk/typescript/README.md), [Python](sdk/python/README.md).

## How Ada decides

- `act` is true only when confidence is at least 0.7, three or more records stand behind the answer and no recorded failure vetoes it, or when the same situation is on record.
- A situation that carries a key or a value no win has recorded is not acted on. The answer's `reason` is `unseen`, `unseen` lists what was new, and the nearest answer is still returned. [When Ada acts](https://glyphh.ai/docs/ada-act) covers every case.
- Two different values score 0. A typed model says how each role is compared: categories, sets, numbers on a scale, text, booleans, times and relations.
- `ada_calibrate` fits a model's probabilities, and `ada_learn` works out what decides from a model's own wins: a flat model's key weights, a typed model's role weights.

## Tools

| Tool | Purpose |
|------|---------|
| `ada_models` | List the organization's Ada models, or one model with its record counts |
| `ada_query` | Ask a model what worked in situations like this one: `act`, `reason`, and the records behind it |
| `ada_facts` | Explain the nearest situations as fact trees |
| `ada_record`, `ada_veto` | Record a graded win, or a failure |
| `ada_check` | How close the nearest win and failure of one outcome are |
| `ada_history`, `ada_trend`, `ada_predict` | A thing's versions and what changed, where it is heading, and its next version |
| `ada_edges` | The things nearest a thing at a level, the change between its versions, and its relations |
| `ada_gql` | One GQL statement: find, list, count, aggregate, compare, drift, and `FOLLOW` and `PATH` over the graph |
| `ada_call`, `ada_procedures`, `ada_save_procedure`, `ada_delete_procedure` | Stored procedures: a GQL statement saved under a name, with parameters |
| `ada_contract` | The JSON Schema of every answer, and of a typed model's data |
| `ada_calibrate` | Fit a model's probabilities to its own records (10+ wins) |
| `ada_learn` | Learn which keys or roles decide from a model's wins (10+), save them as its weights or in its spec, and calibrate it (org admins) |
| `ada_records` | Browse a model's records, newest first |
| `ada_devices` | The registered devices that can hold a local model's records |
| `ada_create_model`, `ada_update_model`, `ada_delete_model` | Manage models and their specs (organization admins) |
| `ada_edit_record`, `ada_delete_record` | Correct or delete a record (organization admins) |

Every call names a `model_id` and is metered on your organization's Glyphh plan. Arguments, answer fields and the `POST /ada` endpoint are in the [tools reference](https://glyphh.ai/docs/ada-tools).

## License

MIT. See [LICENSE](LICENSE).
