# Glyphh Ada Agent Skills

Agent skills for building with [Glyphh](https://glyphh.ai) Ada: typed models your agents build, feed and query, with every answer a fact tree.

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
| `ada` | Decide when to query Ada, how to describe a situation, and when to record a win or a failure |

## How Ada decides

- `act` is true only when confidence is at least 0.7, three or more records stand behind the answer, and no recorded failure vetoes it.
- A record that matches the situation exactly is the answer and acts alone.
- A query can use another word for a recorded value (`production` for `prod`). Cloud models match these through a lexicon built when the record is written; no model runs at query time, and such a match still needs three records to act.
- `ada_learn` works out which keys decide from a model's own wins, and `ada_calibrate` fits its probabilities.

## Tools

| Tool | Purpose |
|------|---------|
| `ada_models` | List the organization's Ada models, or one model with its record counts |
| `ada_query` | Ask a model what worked in situations like this one, with confidence |
| `ada_facts` | Explain the nearest wins and failures as fact trees |
| `ada_record` | Record a graded win: this outcome worked in this situation |
| `ada_veto` | Record a graded failure: this outcome failed or was rejected |
| `ada_check` | How close the nearest win and failure of one outcome are |
| `ada_calibrate` | Fit a model's probabilities to its own records (10+ wins) |
| `ada_learn` | Learn which keys decide from a model's wins (10+), save them as its weights and calibrate it (org admins) |
| `ada_records` | Browse a model's records, newest first |
| `ada_devices` | The registered devices that can hold a local model's records |
| `ada_create_model`, `ada_update_model`, `ada_delete_model` | Manage models (organization admins) |
| `ada_edit_record`, `ada_delete_record` | Correct or delete a record (organization admins) |

Every call names a `model_id` and is metered on your organization's Glyphh plan. Arguments, answer fields and the `POST /ada` endpoint are in the [tools reference](https://glyphh.ai/docs/ada-tools).

## License

MIT. See [LICENSE](LICENSE).
