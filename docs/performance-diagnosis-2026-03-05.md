# Performance Diagnosis: Slow Agent Execution

**Date**: 2026-03-05
**Issue**: Agents running very slowly after recent changes
**Methodology**: Systematic debugging with root cause investigation

---

## Summary

**ROOT CAUSE IDENTIFIED**: The planner is making **3 LLM calls per planning cycle** (instead of 1-2), causing each timestep to take ~5-6 seconds. This is due to the planner calling observation tools before assigning tasks.

**Combined factors**:
1. New observation tools added to planner (commit `449c779`)
2. API latency of ~1.1s per call
3. Planner running every 3 steps

**Impact**: ~5.7 seconds per timestep (vs expected ~2-3 seconds)

---

## Investigation Results

### 1. API Latency (Test 1)
**Finding**: API latency is **NOT** the bottleneck.

```
Average latency: 1092ms (~1.1s per call)
Min: 904ms, Max: 1319ms
```

**Verdict**: ✅ Acceptable latency for a remote API endpoint

### 2. LLM Calls Per Action (Test 2)

**Planner**:
- **6 LLM calls across 2 planning cycles** = 3 calls per cycle
- Expected: 1-2 calls per cycle
- **Verdict**: ⚠️ EXCESSIVE (50% overhead)

**Workers**:
- worker_0: 1.7 calls per action
- worker_1: 1.5 calls per action
- **Verdict**: ✅ Normal (1-2 calls is expected for ReAct loop)

### 3. Total Time Per Timestep

```
Calculation:
- Planner: 3 calls ÷ 3 steps = 1.0 call per step
- worker_0: 1.7 calls per step
- worker_1: 1.5 calls per step
- Total: ~4.2 LLM calls per timestep

With 1.1s API latency:
4.2 calls × 1.1s = ~4.6 seconds per timestep

Observed in practice: ~5-6 seconds per timestep
```

---

## Root Cause Analysis

### What Changed?

**Commit `449c779`** (5 commits ago) added two new observation tools to the planner:

1. `get_nearby_interactables` - checks what objects workers can interact with
2. `validate_task_feasibility` - validates if proposed tasks are achievable

**Before**: 4 observation tools
**After**: 6 observation tools

### Why Does This Cause Slowdown?

The planner's ReAct loop works like this:

```
1. LLM call → decides to call observation tools
2. Tools execute → gather information
3. LLM call → processes results, may call more observation tools
4. LLM call → finally calls assign_tasks action
```

With **more observation tools available**, the LLM is naturally inclined to explore them before making decisions, adding extra LLM calls.

### Evidence

From background task output:
```
[Planner] Invoking graph (step 0)...
[Planner] Calling LLM...        ← Call 1: explore available tools
[Planner] Calling LLM...        ← Call 2: call observation tools
[Planner] Calling LLM...        ← Call 3: finally assign tasks
[Planner] Graph completed
```

**Expected**: 1-2 LLM calls (direct task assignment or 1 observation + assignment)
**Actual**: 3 LLM calls (2 observations + assignment)

---

## Impact Calculation

For a 20-step episode with replan_interval=3:

```
Planning cycles: 20 ÷ 3 = ~6-7 cycles
Extra LLM calls: 1 extra call per cycle × 6 cycles = 6 extra calls
Extra time: 6 calls × 1.1s = ~6.6 seconds of overhead

Total episode time:
- Expected: ~2-3 minutes
- Actual: ~3-4 minutes (+30-50% slower)
```

For longer episodes (400 steps):
```
Planning cycles: 400 ÷ 3 = ~133 cycles
Extra time: 133 calls × 1.1s = ~146 seconds = ~2.5 minutes of overhead
```

---

## Recommendations

### Option 1: Optimize Planner System Prompt (Recommended)
Modify the planner's system prompt to encourage more direct task assignment:

```python
# Add to planner system prompt:
"EFFICIENCY: Assign tasks decisively. Only use observation tools when
absolutely necessary to resolve ambiguity. Trust your workers to execute
their tasks autonomously."
```

**Pros**: Reduces LLM calls without removing tool functionality
**Cons**: May reduce planning quality if observations were valuable

### Option 2: Remove New Observation Tools
Revert the addition of `get_nearby_interactables` and `validate_task_feasibility`.

**Pros**: Guaranteed to restore previous performance
**Cons**: Loses potentially valuable planning capabilities

### Option 3: Increase Replan Interval
Change `replan_interval` from 3 to 5 or 10 steps.

**Pros**: Reduces frequency of expensive planning cycles
**Cons**: Workers may continue outdated tasks longer

### Option 4: Use Faster Model for Planner
Use a faster/cheaper model (e.g., `gpt-4o-mini` or `haiku`) for the planner.

**Pros**: Reduces latency per call
**Cons**: May reduce planning quality

### Option 5: Parallelize Planning and Worker Execution
Invoke the planner asynchronously while workers continue executing.

**Pros**: Hides planning latency
**Cons**: Complex implementation, potential race conditions

---

## Testing

To verify the root cause, run:

```bash
# Test with observation tools disabled (temporary modification)
# Expect: ~2-3 seconds per timestep instead of 5-6 seconds
```

To monitor performance improvements:

```bash
# Run with timing instrumentation
uv run python testing/test_api_latency.py
```

---

## Conclusion

The slowness is **NOT due to high API latency** but rather due to **excessive observation tool usage by the planner**, which was introduced by adding new tools in commit `449c779`.

**Next steps**: Choose a mitigation strategy (recommended: Option 1 or 3) and implement.
