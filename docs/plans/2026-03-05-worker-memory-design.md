# Worker Memory System Design

**Date**: 2026-03-05
**Status**: Approved

## Problem

WorkerAgent has no memory across turns. Each turn is stateless — the worker doesn't know what it did last turn, leading to:
- Repeated failed actions (e.g., moving into a wall repeatedly)
- No awareness of position changes (or lack thereof)
- Deadlocks between workers who can't learn from collisions

## Solution

Add a compact memory system to WorkerAgent, following the proven LLMAgent pattern but optimized for minimal token cost.

## Design

### History Entry Format

Each entry stores:
- `timestep`: int
- `position`: tuple (x, y)
- `action`: human-readable action char
- `held`: what the worker is holding (or "nothing")
- `task`: current task description (for boundary detection)

### Prompt Injection Format (compact, ~15 tokens/entry)

```
RECENT HISTORY:
- Step 12: at (2,1) → ↑ [Task: Pick up onion]
- Step 13: at (2,0) → interact [Task: Pick up onion]
--- New task ---
- Step 14: at (2,0) holding onion → → [Task: Deliver to pot]
```

Task boundary markers (`--- New task ---`) are inserted when the task description changes between consecutive entries.

### Parameters

- `history_size`: default 5 (covers one full replan cycle at default `replan_interval=5`)
- Resets on `reset()` (new episode)
- Persists across task reassignments with boundary markers

### Token Budget

- ~15 tokens per entry x 5 entries = ~75 extra tokens per turn
- Negligible compared to system prompt (~600 tokens) + state (~300 tokens)

## Files Modified

1. **`worker_agent.py`** — add `history_size`, `_history`, `_format_history()`, `_add_to_history()`, update `action()` and `reset()`

## What Doesn't Change

- `ToolState`, `worker_tools.py`, `state_serializer.py`, `graph_builder.py` — untouched
- System prompt stays the same
- Graph builder / ReAct loop unchanged
