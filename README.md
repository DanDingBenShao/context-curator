# Context Curator

Pluggable LLM context lifecycle management middleware. Add it to any AI agent in 3 lines of code.

## Features

- **Dual-score dormancy**: Long-term (future value) + short-term (immediate relevance) — inactive topics sleep, active ones stay
- **Auto-compression**: Long outputs auto-summarized, originals indexed for pull-back
- **Pin protection**: Mark critical content to survive budget pressure
- **Budget-aware**: Adapts scoring/compression/eviction based on token usage
- **Script-enforced**: LLM scores, script handles decay/dormancy/deletion — deterministic and safe

## Install

```bash
pip install git+https://github.com/DanDingBenShao/context-curator.git
```

## Quick Start

```python
from context_curator import create_curator

# One line setup (reads CURATOR_API_KEY from env)
curator = create_curator(max_tokens=80000)

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
    llm_client=my_llm,       # any object with .chat(messages) → str
    db_path="curator.db",
    max_context_tokens=80000,
)
curator = ContextCurator(config)
result = curator.curate(user_message)
```

## Configuration

| Param | Default | Description |
|-------|---------|-------------|
| `max_context_tokens` | 80000 | Target context budget |
| `dormant_threshold` | 3 | short_term ≤ N → sleep |
| `decay_per_turn` | 1 | Long-term score decay/turn |
| `short_term_decay` | 2 | Short-term score decay/turn |
| `host_model` | "" | Auto-detect window size (model name) |
| `log_path` | "" | JSONL log path (empty = no log) |

## How It Works

Each turn: decay → add message → LLM scores/compresses → trim → return clean context

See `context_curator/` source for full details.
