# Glyphh hooks — persistent memory for Claude Code that survives compaction

These hooks make Glyphh the **memory spine** for a Claude Code session. Decisions
made early survive context compaction because they live in Glyphh, not the
window, and get re-injected every turn — deterministically, with zero LLM tokens.

## How it works

| Hook | Event | Job |
|------|-------|-----|
| `glyphh_sessionstart.py` | SessionStart | load standing decisions at session start/resume |
| `glyphh_userprompt.py` | UserPromptSubmit | **capture** instruction-like prompts + **re-ground** from memory every turn |
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

## Note on enforcement

These hooks **re-inject** memory (raises adherence). For a hard guarantee that the
agent can't *act* against a stored rule, add a `PreToolUse` hook that recalls
relevant constraints and returns `permissionDecision: "deny"` when an action
violates one. That's the difference between "reminded" and "can't get around it".
