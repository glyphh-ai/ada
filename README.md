# Ada for Claude Code

Ada is Glyphh's graded precedent: record what worked and what failed,
ask what worked in situations like this one, and see why as a fact tree.
This plugin connects Claude Code to Ada on Glyphh's servers over MCP.

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
- The `ada` skill, which tells Claude when to consult Ada and when to
  record an outcome.

Every call is metered on your organization's Glyphh plan.

## License

Proprietary. Copyright glyphh.ai.
