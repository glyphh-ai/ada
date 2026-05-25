# Glyphh skills for Claude Code

## glyphh-guard

Compiles a plain-English enforcement rule ("never push to main") into a
structured `add_guard` MCP call, using Claude Code's own reasoning (the user's
subscription) — Glyphh stays LLM-free. The resulting guard is enforced by the
`PreToolUse` hook (see `../hooks/`).

**Install** (project or user scope):

```bash
mkdir -p .claude/skills
cp -r skills/glyphh-guard .claude/skills/
```

Then, with the Glyphh runtime running and the `glyphh` MCP server connected,
just say: *"never push directly to main"* — Claude Code compiles it to a guard
and any matching `git push ... main` is blocked.
