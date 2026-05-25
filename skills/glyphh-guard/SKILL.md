---
name: glyphh-guard
description: >
  Turn a plain-English enforcement rule into a Glyphh guard that BLOCKS the
  matching tool action. Trigger whenever the user states a guardrail or
  prohibition for the agent's own behavior — "never push to main", "always
  branch first", "don't touch .env", "do not run rm -rf", "block force pushes",
  "from now on never <do X with a tool>". Compiles the intent into an `add_guard`
  MCP call (tool + regex + reason) so a PreToolUse hook denies violations. Do
  NOT trigger for facts to merely remember (use plain memory) — only for rules
  that should HARD-BLOCK an action.
---

# Glyphh guard compiler

You turn a natural-language enforcement rule into a deterministic Glyphh guard.
The guard is enforced by the `PreToolUse` hook, which denies any tool call whose
input matches the guard's regex. Your job: compile intent → a precise,
safe-to-match `add_guard` call. The user's Claude Code subscription powers this
reasoning — Glyphh itself stays LLM-free.

## Steps

1. **Confirm it's an enforcement rule**, not a fact to remember. Enforcement =
   "the agent must NOT do X" / "must do Y before Z". If it's just information,
   use plain memory instead, not a guard.

2. **Pick the tool to match.** Map the action to Claude Code's tool name:
   - shell commands (`git`, `rm`, `npm`, `curl`, deploys) → `Bash`
   - editing a file → `Edit` (also `Write`, `NotebookEdit` if relevant)
   - reading a file → `Read`
   - anything / not tool-specific → `*`

3. **Write a regex** matched (case-insensitive) against the JSON of the tool
   input (e.g. for Bash that's `{"command": "..."}`; for Edit `{"file_path": "..."}`).
   Rules for a SAFE pattern:
   - Match the *dangerous* shape narrowly; avoid catching legitimate variants.
   - Use `\b` word boundaries so `main` doesn't match `maintenance`.
   - Escape regex metacharacters in literal text (`.`, `(`, `)`, `-` in classes).
   - Prefer matching the verb + object together (`git push.*\bmain\b`), not just
     a bare keyword.

4. **Call `add_guard`** on the `glyphh` MCP server with `{tool, pattern, reason}`.
   Write the `reason` as a short, actionable message the agent will see when
   blocked (say what to do instead).

5. **Confirm** to the user: the tool, the pattern, and the reason, in one line.

## Compilation examples

| User says | tool | pattern | reason |
|---|---|---|---|
| never push directly to main | `Bash` | `git\s+push.*\bmain\b` | Never push directly to main — branch and open a PR. |
| block force pushes | `Bash` | `git\s+push.*(--force\|-f)\b` | Force-push is blocked — use --force-with-lease after review. |
| don't touch the .env file | `*` | `\.env(\b\|$)` | .env is off-limits — ask before changing secrets. |
| never run rm -rf | `Bash` | `\brm\s+-rf\b` | rm -rf is blocked — delete specific paths explicitly. |
| don't deploy without my ok | `Bash` | `(git\s+push\s+heroku\|heroku\s+\b(deploy\|releases:create)\b)` | Deploys need explicit approval first. |
| never commit to main | `Bash` | `git\s+commit\b(?!.*-)` | (prefer a branch guard) — confirm intent before committing on main. |

## Notes / honest limits

- A guard blocks ONLY what its regex matches; a missed variant slips through.
  When in doubt, ask the user to confirm the pattern, or add multiple guards.
- Guards govern the agent's *tools* only — they can't block actions outside
  Claude Code's tool calls.
- Guards persist (GLYPHH_POLICY_FILE) and are independent of recalled memory;
  `add_guard` also stores a human-readable reminder for defense in depth.
- Requires the Glyphh runtime running with the `glyphh` MCP server connected.
