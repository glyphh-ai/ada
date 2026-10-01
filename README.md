# Ada

Ada is Glyphh's typed language for AI work. You define a model with a
typed spec (layers, segments and roles, each weighted), stream data into
it, and ask it anything in that shape. Every answer comes back as a fact
tree: what matched, how closely, and why, scored at every level of the
spec. Use it to classify tickets, route work across an agent swarm, keep
what worked and what failed, or remember anything an agent should reason
over later.

Ada runs on Glyphh's servers and speaks MCP, so any MCP client can use
it: Claude Code, Claude Desktop, Cursor, Codex, VS Code, your own agents.
This repo holds the connection details and a ready-made Claude Code
plugin.

## Connect any MCP client

Ada is a streamable HTTP MCP server:

- URL: `https://api.glyphh.ai/mcp`
- Header: `x-glyphh-api-key: Bearer <your Glyphh API key>`

Get a key in the Glyphh console under API Keys. Most clients take a config
like this:

```json
{
  "mcpServers": {
    "ada": {
      "type": "http",
      "url": "https://api.glyphh.ai/mcp",
      "headers": { "x-glyphh-api-key": "Bearer sk-..." }
    }
  }
}
```

## Claude Code plugin

The plugin adds the same connection plus the `ada` skill, which tells
Claude when to reach for Ada.

```
/plugin marketplace add glyphh-ai/ada
/plugin install ada@ada
```

Set your key in the environment Claude Code starts from:

```bash
export GLYPHH_API_KEY=sk-...
```

`GLYPHH_URL` points it at another Glyphh server; it defaults to
`https://api.glyphh.ai`.

## Tools

The `ada_*` tools: models, query, facts, record, veto, check, calibrate
and records, plus model and record management for organization admins.
Every call is metered on your organization's Glyphh plan.

## License

MIT. See [LICENSE](LICENSE).
