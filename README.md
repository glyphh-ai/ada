# Glyphh Runtime

Ada's cognitive brain — a hyperdimensional computing runtime that gives LLMs persistent memory, deterministic routing, and background reasoning.

Ada sits between the user and any LLM (Claude, Gemini, GPT). She absorbs everything, recalls what's relevant, and controls what the LLM sees. The LLM is Ada's language center — she is the information boundary.

## Architecture

```
User input
    |
    v
+---------------------------------------------------+
|  Ada's Think Pipeline (per request, 1-3 cycles)   |
|                                                    |
|  1. PERCEIVE   CognitiveGlyph classification       |
|                 STORE | RECALL | FEEL | CONTRADICT |
|                                                    |
|  2. GUARD      Prompt injection firewall            |
|                 HDC cosine similarity, no LLM       |
|                 200+ attack patterns, 16 families   |
|                                                    |
|  3. RECALL     Cognitive gradient search            |
|                 Multi-step vector descent through    |
|                 thought space (see below)            |
|                                                    |
|  4. REASON     Brain router — which capability?     |
|                 HDC routing, sub-10ms deterministic  |
|                                                    |
|  5. ACT        Execute capability query             |
|                                                    |
|  6. EVALUATE   Confidence scoring                   |
|                 < 0.5 → retry or "I don't know"     |
|                 >= 0.5 → LLM synthesizes from facts  |
|                                                    |
|  7. RESPOND    Formulate response                   |
|                 STORE → "Got it."                    |
|                 FEEL → empathy                       |
|                 RECALL → grounded answer or fallback |
+---------------------------------------------------+
    |
    v
Absorb input + response → persist queue → SQLite
Record observation → dream loop
```

### The Cognitive Gradient

The RECALL step is not one-shot similarity search. It's gradient descent through HDC vector space:

1. Encode the query into a search vector
2. Find the best matching thought in memory
3. Compute the **residual** — the part of the query that wasn't answered
4. Update the search vector toward the residual
5. Search again with the updated vector
6. Stop when the residual norm < 0.15 (converged) or 5 steps exhausted

This is how Ada *thinks through* a question rather than just pattern-matching. Example: "who am i?" first matches identity patterns broadly, then the residual captures the self-referential "me, not you" signal, and the next step finds the user's name.

### The Hallucination Gate

Ada's confidence threshold (0.5) controls whether the LLM is involved:

- **confidence >= 0.5** — LLM synthesizes a natural response from grounded facts. The LLM only sees facts that survived the cognitive gradient. It cannot add information.
- **confidence < 0.5** — LLM is cut out entirely. Ada returns the top fact raw, or says "I don't have information about that in my memory." No confabulation possible.

### The Dream Loop

Ada dreams continuously in the background while awake:

- **REM phase** (every 3s) — Local pattern mining. Discovers connections between recent thoughts, reinforces frequently-accessed memories (Hebbian strengthening), decays unused ones.
- **Slow-Wave phase** (every 30s) — Deep reasoning. Multi-hop inference, contradiction detection, knowledge gap identification. Can crystallize new compound primitives from discovered patterns.

Dreams produce **insights** — connections, contradictions, convergences, questions, crystallizations — that accumulate and can be drained via the CLI.

### Memory

Ada's memory is a `ThoughtGlyphSpace` — an in-memory HDC vector store backed by SQLite for persistence:

- **Absorb** — Natural language is encoded into 5-layer Thought Glyphs (perspective, semantic, relational, temporal, direction) with 2000-dimensional vectors
- **Recall** — Three-signal search: content cosine (0.50 weight), per-layer role cosine (0.35), structural Jaccard overlap (0.15)
- **Persistence** — Background worker flushes new thoughts to SQLite every 2 seconds. Strengths are flushed on shutdown.
- **Hebbian reinforcement** — Recalled memories get stronger. Unused ones decay. Strength caps at 3.0, decay floor at 0.01 (archived, never deleted).

### Capabilities

Ada loads capabilities from `capabilities/` at boot. Each capability is an HDC model with exemplars:

- **firewall** — Prompt injection detection (200+ patterns, 16 attack families)
- **cognitive-router** — Routes requests to the right capability
- **speaker-id** — Voice identification via MFCC encoding
- **code-index** — Semantic codebase search with 4-layer encoding

Ada can build new capabilities autonomously: "build a capability for X" triggers the capability builder, which generates exemplars via LLM, encodes them into HDC vectors, and hot-loads the new model.

## MCP Interface

Ada exposes a single MCP tool:

```
think(input: string) → response
```

Available at `http://localhost:8002/mcp` via Streamable HTTP.

### Connect from Claude Code

Create `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "ada": {
      "type": "http",
      "url": "http://localhost:8002/mcp"
    }
  }
}
```

### Connect from other tools

Any MCP-compatible client can connect to the same endpoint. Ada maintains shared memory across all connected tools — context flows between them.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install glyphh[runtime]
```

## Quick Start

```bash
# Start the interactive shell (boots Ada's brain)
glyphh

# Or start as a headless server
glyphh serve
```

Ada boots in sequence: validate config → load license → init database → load capabilities → encode exemplars → seed identity memories → load persisted thoughts → start dream loop → open MCP endpoint.

## CLI Commands

The interactive shell (`glyphh`) accepts natural language — everything goes through Ada's think pipeline. Prefixed commands:

```
dream status          Background reasoning stats
dream start           Start the dream loop
dream stop            Stop the dream loop
dream insights        Show recent insights

memory reset          Clear all memory (in-memory + SQLite)

config show           Show current configuration
auth login            Authenticate with Glyphh platform
auth logout           Log out
```

## Testing

```bash
pip install glyphh[dev]
pytest tests/unit/ -v
```

The brain pipeline test suite covers:
- **ThoughtGlyphSpace** — absorb, dedup, recall ranking, user vs Ada identity isolation
- **AdaCognitive** — StoredThought returns, sentence splitting, recall gating, dream lifecycle
- **ThoughtProcess** — STORE/FEEL/RECALL paths, confidence gate, hallucination prevention
- **Brain** — seed loading, user identity, persistence queue, end-to-end think

## License

Glyphh AI Community License — Copyright (c) 2026 Glyphh AI LLC. All rights reserved.

See [LICENSE](https://github.com/glyphh-ai/glyphh-runtime/blob/main/LICENSE) for full terms. Patent pending (Application No. 63/969,729).
