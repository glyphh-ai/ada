# Ada for Claude Code

Ada is Glyphh's typed language for AI work. You define a model with a
typed spec (layers, segments and roles, each weighted), stream data into
it, and ask it anything in that shape. Every answer comes back as a fact
tree: what matched, how closely, and why, scored at every level of the
spec. Use it to classify tickets, route work across an agent swarm, keep
what worked and what failed, or remember anything an agent should reason
over later. This plugin connects Claude Code to Ada on Glyphh's servers
over MCP.

## Install

```
/plugin marketplace add glyphh-ai/ada
/plugin install ada@ada
```

Set your Glyphh API key (Glyphh console, API Keys) in the environment
Claude Code starts from:

```bash
export GLYPHH_API_KEY=sk-...
```

`GLYPHH_URL` points the plugin at another Glyphh server; it defaults to
`https://api.glyphh.ai`.

## What it adds

- The `ada_*` tools: query, facts, record, veto, check, calibrate,
  models and records.
- The `ada` skill, which tells Claude when to reach for Ada: to build a
  model for recurring work, load what it learns, and query it before
  deciding.

Every call is metered on your organization's Glyphh plan.

## License

MIT. See [LICENSE](LICENSE).
