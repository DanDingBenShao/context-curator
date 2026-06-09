# Context Curator

Pluggable LLM context lifecycle management middleware. Add it to any AI agent in 3 lines of code.

## Features

- **Dual-score dormancy**: Long-term (future value) + short-term (immediate relevance) — inactive topics sleep, active ones stay
- **Knowledge gap detection**: Step 0 reasoning chain — analyzes user intent, decomposes prerequisite knowledge, compares against existing context, triggers memory recall when gaps found
- **Multi-query memory recall**: LLM generates main query + synonym variants for better retrieval coverage; single-query systems use `queries[0]` as fallback
- **Auto-compression**: Long outputs auto-summarized, originals indexed for pull-back
- **Pin protection**: Mark critical content to survive budget pressure
- **Budget-aware**: 4-level pressure adapts scoring/compression/eviction based on token usage
- **Script-enforced**: LLM scores/detects gaps, script handles decay/dormancy/deletion — deterministic and safe
- **Persist marking**: LLM marks cross-session facts (preferences, decisions, key facts) via `persist_segments` — external Agent Loop handles batch writes to long-term memory

## Install

```bash
pip install git+https://github.com/DanDingBenShao/context-curator.git
```

## Quick Start

```python
from context_curator import create_curator

# Zero-config setup (reads CURATOR_API_KEY from env)
curator = create_curator(
    max_tokens=80000,
    memory_fn=lambda queries: my_memory_system.search(queries),
)

# Call after each user turn
result = curator.curate(user_message)

# result.context → clean segment list, inject into next system prompt
for seg in result.context:
    print(f"[{seg.id}] {seg.content[:80]}...")
```

Or bring your own LLM client:

```python
from context_curator import ContextCurator, CuratorConfig

config = CuratorConfig(
    llm_client=my_llm,           # any object with .chat(messages) → str
    db_path="curator.db",
    max_context_tokens=80000,
    memory_fn=my_memory.search,     # (List[str]) → str, multi-query recall
)
curator = ContextCurator(config)
result = curator.curate(user_message)
```

## Configuration

| Param | Default | Description |
|-------|---------|-------------|
| `llm_client` | (required) | LLM for scoring, any `.chat(messages) → str` |
| `db_path` | `curator.db` | SQLite persistence path |
| `max_context_tokens` | 80000 | Target context budget |
| `dormant_threshold` | 3 | short_term ≤ N → sleep |
| `decay_per_turn` | 1 | Long-term score decay/turn |
| `short_term_decay` | 2 | Short-term score decay/turn |
| `host_model` | "" | Auto-detect window size (model name) |
| `log_path` | "" | JSONL log path (empty = no log) |
| `memory_fn` | None | `(List[str]) → str`, called when LLM detects knowledge gaps |
| `search_fn` | None | `(str) → str`, web search callback |
| `file_read_fn` | None | `(str) → str`, local file read callback |
| `host_info_fn` | None | `() → str`, host model info callback |

## How It Works

Each turn:

1. **Decay**: All scores auto-decay (long -1, short -2)
2. **Add message**: New user message added with default scores
3. **Step 0 — Gap & Persist**: LLM reasons through user intent → prerequisite knowledge → gap check → triggers memory recall if needed; also marks cross-session facts (persist_segments) for external batching
4. **Score/Compress**: LLM adjusts scores, compresses long outputs
5. **Trim**: Low-score segments evicted, dormant pool capped
6. **Return**: Clean context for the host agent, with persist_segment IDs for Agent Loop to batch-write

See `context_curator/` source for full details.
