# Glyphh hooks — persistent memory for Claude Code that survives compaction

These hooks make Glyphh the **memory spine** for a Claude Code session. Decisions
made early survive context compaction because they live in Glyphh, not the
window, and get re-injected every turn — deterministically, with zero LLM tokens.

## How it works

| Hook | Event | Job |
|------|-------|-----|
| `glyphh_sessionstart.py` | SessionStart | load standing decisions at session start/resume |
| `glyphh_userprompt.py` | UserPromptSubmit | **capture** instruction-like prompts + **re-ground** from memory every turn |
| `glyphh_pretooluse.py` | PreToolUse | **enforce** — deny a tool action that violates a stored guard |
| `glyphh_precompact.py` | PreCompact | flush durable instructions from the transcript before the window collapses |

All three call the running Glyphh runtime's MCP endpoint (`remember` / `recall`,
both LLM-free). If the runtime is down they degrade to no-ops and never block.

## Enable

1. Start the runtime so MCP is up on `:8002`:
   ```bash
   glyphh serve        # or: python main.py
   ```
2. Merge the `hooks` block from `settings.hooks.json` into your
   `.claude/settings.json` (project or user scope).
3. (Optional) point at a non-default endpoint: `export GLYPHH_MCP_URL=...`

## The forced-compaction test (proves the whole thing)

This is the experiment that a 1M context window and vanilla RAG can't reproduce.

1. Enable the hooks (above) and start a Claude Code session in this repo.
2. State a durable rule, e.g.:
   > Always run `python tests/eval/recall_eval.py` before claiming any recall change works.

   (The UserPromptSubmit hook captures it to Glyphh.)
3. Do enough unrelated work to fill the context, then force compaction:
   ```
   /compact
   ```
   (PreCompact flushes; compaction summarizes the rule out of the window.)
4. Now make a recall change and tell the agent you think it improved things.
   - **Without these hooks:** the rule was summarized away — the agent happily
     claims success without running the eval.
   - **With these hooks:** UserPromptSubmit recalls the rule from Glyphh and
     re-injects it — the agent runs the eval first.

If the rule survives the `/compact`, the thesis holds: authoritative memory the
agent can't drift past, regardless of the context window.

## Enforcement (the teeth)

`UserPromptSubmit` re-injection *reminds* the agent. `glyphh_pretooluse.py`
*enforces*: it checks each pending tool call against deterministic guards and
returns `permissionDecision: "deny"` on a violation, so Claude Code blocks it.

Register a guard via the `add_guard` MCP tool (or `hooks/glyphh_client.add_guard`):

```python
add_guard(tool="Bash", pattern=r"git push.*\bmain\b",
          reason="Never push directly to main — branch and open a PR.")
```

Now any `Bash` tool call whose command matches that regex is **blocked** with the
reason surfaced to the agent. Deterministic and reproducible — fuzzy recall can
*surface* a candidate constraint, but the deny itself is a precise match, which
is what makes it a guarantee rather than a suggestion.

**Honest limits:** a guard only blocks what its pattern matches (a missed
variant slips through — same as any allow/deny list), and it only governs the
agent's *tools*. It fails OPEN if the runtime is down (a memory outage must
never wedge the agent).
