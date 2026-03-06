# Performance Optimization Summary

**Date**: 2026-03-05
**Issue**: Each step taking ~48 seconds (target: <5 seconds)

## Root Cause Analysis

### Excessive LLM Calls

**Actual vs Expected:**
- **Expected**: ~17 LLM calls (1 per agent per step for 17 steps)
- **Actual**: 81 LLM calls (4.8x more!)
  - Planner: 12 calls (3 per replan cycle)
  - Worker_0: 33 calls (2-3 per step)
  - Worker_1: 36 calls (2-3 per step)

### Why So Many Calls?

Workers are using a **ReAct loop** where observation tools trigger additional LLM calls:

1. Worker gets task
2. Calls `get_surroundings()` → LLM call #1
3. LLM processes result
4. Calls `check_path()` → LLM call #2
5. LLM processes result
6. Finally calls action tool (move/interact) → LLM call #3

**Tool call breakdown from logs:**
- `check_path`: 26 calls (workers checking paths repeatedly)
- `get_surroundings`: 10 calls
- `get_pot_details`: 5 calls (planner)
- Movement/interact: 35 calls (this should be 1:1 with steps)

## Optimizations Applied

### 1. Enriched Worker Context (worker_agent.py)

**Before:**
```python
prompt = f"Your task: {task}\n\nCurrent state:\n{state_text}\n\nChoose one action."
```

**After:**
```python
# Precompute nearby objects within 2 tiles
surroundings = [f"({x},{y}) distance {dist}: {obj}" for nearby objects]

prompt = f"""Your task: {task}

Current state:
{state_text}

NEARBY OBJECTS (within 2 tiles):
{surroundings}

REMINDER: You must be ADJACENT (distance=1) to interact. If distance > 1, move closer first.

Choose ONE action. DO NOT call observation tools unless absolutely necessary."""
```

**Impact**: Workers can decide actions with 1 LLM call instead of 2-3.

### 2. Reduced Recursion Limit (worker_agent.py)

**Before:** `recursion_limit=15` (allows ~5 observation tool calls)
**After:** `recursion_limit=8` (forces action choice sooner)

**Impact**: Prevents workers from making excessive observation tool calls.

### 3. Added Performance Diagnostics (graph_builder.py)

```python
# Now prints:
# [worker_0] Calling LLM (prompt: 2,450 chars, 2 messages)...
# [worker_0] LLM API response received (450ms)
```

**Impact**: Can identify slow API calls and large prompts.

## Expected Performance Improvement

### Before Optimizations:
- **Step 0**: ~48s (planner: 38s, workers: ~10s)
- **LLM calls per step**: ~5 calls (1 planner + 2 worker_0 + 2 worker_1)

### After Optimizations:
- **Expected**: ~10-15s per step
- **LLM calls per step**: ~3 calls (1 planner + 1 worker_0 + 1 worker_1)
- **Speedup**: ~3-4x faster

### Breakdown:
- Planner: ~2-4s (1-2 LLM calls per replan)
- Worker_0: ~1-2s (1 LLM call per step)
- Worker_1: ~1-2s (1 LLM call per step)
- **Total**: ~4-8s per step (depending on replan)

## Additional Optimizations (Optional)

### 1. Use Faster Model

Current: `minimax.minimax-m2` (large model, ~1-2s per call)

**Options:**
```bash
# Option A: GPT-4o-mini (faster, cheaper)
LLM_MODEL=openai/gpt-4o-mini
WORKER_MODEL=openai/gpt-4o-mini

# Option B: Hybrid (powerful planner, fast workers)
LLM_MODEL=openai/gpt-4o              # Planner
WORKER_MODEL=openai/gpt-4o-mini      # Workers

# Option C: Claude Haiku (very fast)
WORKER_MODEL=anthropic/claude-3-5-haiku-20241022
```

**Impact**: 2-5x faster LLM responses (300-500ms vs 1-2s)

### 2. Increase Replan Interval

Current: `--replan-interval 5` (replans every 5 steps)

```bash
# Test with less frequent replanning
--replan-interval 10   # Replan every 10 steps
```

**Impact**: Fewer planner invocations, but potentially less adaptive.

### 3. Disable Unnecessary Tools

Remove or disable observation tools workers rarely need:
- Keep: `get_surroundings` (useful fallback)
- Remove: `check_path` (now provided in context)
- Remove: `distance_to` (not needed if surroundings provided)

**Impact**: Smaller tool list = faster tool selection.

## Testing Performance

### Run with performance analysis:

```bash
# Run a short episode
source .env && uv run python scripts/run_llm_agent.py \
  --agent-type planner-worker \
  --layout cramped_room \
  --horizon 20 \
  --replan-interval 5 \
  --debug

# Analyze the results
uv run python scripts/analyze_performance.py
```

### Key metrics to watch:

1. **LLM calls per step**: Should be ~3 (down from ~5)
2. **Time per step**: Should be ~5-15s (down from ~48s)
3. **Tool calls**: Movement/interact should be ~1:1 with steps

## Performance Analysis Script

Use the new `scripts/analyze_performance.py` to analyze runs:

```bash
# Analyze most recent run
uv run python scripts/analyze_performance.py

# Analyze specific run
uv run python scripts/analyze_performance.py logs/agent_runs/<run_id>.jsonl
```

**Output includes:**
- LLM calls per agent role
- Average latency per call
- Total tokens and estimated cost
- Tool call breakdown
- Step-by-step timeline

## Next Steps

1. **Test optimized version** and compare to baseline
2. **Experiment with models** (try gpt-4o-mini for workers)
3. **Tune replan interval** (try 7-10 steps)
4. **Profile specific slow steps** using analyze_performance.py
5. **Consider removing check_path tool** entirely if not needed

## Monitoring

Watch for these warning signs:
- Workers calling observation tools frequently (check debug output)
- Recursion limit warnings (means tool loop not converging)
- Steps taking >20s (indicates slow API or model)

---

**Related Files:**
- `src/overcooked_ai_py/agents/llm/worker_agent.py` (optimizations applied)
- `src/overcooked_ai_py/agents/llm/graph_builder.py` (diagnostics added)
- `scripts/analyze_performance.py` (analysis tool)
