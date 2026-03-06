# Planner-Worker Latency Reduction Design

**Date:** 2026-03-06  
**Status:** Approved design (brainstorming phase)  
**Primary Goal:** Reduce per-step runtime to under 10 seconds by removing planner/worker observability bloat while preserving planner control.

## Problem Summary

Current planner-worker execution is too slow for long horizons because:
- Planner spends multiple LLM turns exploring observability tools before assigning tasks.
- Workers spend multiple LLM turns in ReAct observation loops before selecting one action.
- End-to-end per-step latency becomes too high to iterate on experiments.

## Design Decisions

### 1. Planner keeps control but drops observability tools

- Keep planner in the existing ReAct graph.
- Keep only planner action tool: `assign_tasks`.
- Remove planner observability tools from planner tool registry:
  - `get_surroundings`
  - `get_pot_details`
  - `check_path`
  - `get_worker_status`
  - `get_nearby_interactables`
  - `validate_task_feasibility`
- Replace tool-based state gathering with a richer deterministic context block in planner prompt.

### 2. Planner always assigns both workers each replan

- On every replan cycle, planner must assign tasks to both `worker_0` and `worker_1`.
- Assignments can re-issue same objective or provide new objectives.
- Replan output is always correlated to current state at that timestep.

### 3. Planner must be strictly state-grounded

Add explicit planner instruction:
- Plan from each worker’s current observed state at this timestep.
- Do not assume current/previous task will be completed first.
- Do not chain “after completion then …” assumptions.

### 4. Worker execution becomes one-shot

- Worker execution path should produce one action from one model response per step.
- Default behavior should avoid worker observation tool usage.
- Rich worker context must be provided up front to reduce need for tool calls.
- Keep deterministic parse/validation with fallback action on invalid output.

## Planner Context Payload (Deterministic, Tool-Free)

Each planner cycle should include:
- Full serialized game state (existing serializer output)
- Per-worker snapshot:
  - position
  - orientation
  - held object
  - current assigned task text
  - steps active on current task
- Pot summary:
  - empty / partial / full-not-cooking / cooking / ready
- Counter object summary
- Recent planning history (already present)

This context replaces planner-side observability tool calls.

## Runtime Flow

1. `maybe_replan` decides if planning is needed.
2. Build deterministic planner context snapshot.
3. Planner receives prompt and calls `assign_tasks` with both worker assignments.
4. Assigned tasks overwrite previous tasks for both workers at each replan cycle.
5. Each worker executes one-shot action selection from current state + task + local context.
6. Worker output is validated and converted to one action.

## Error Handling and Safety

### Planner
- Planner output must include both workers via `assign_tasks`.
- If planner output is malformed or incomplete:
  - Preserve last valid task for missing worker(s) for that cycle.
  - Emit structured debug/error event.

### Worker
- Worker output must map to one valid action.
- If parsing fails, use deterministic fallback (default: `wait`).
- If repeated blocked motion is detected, fallback to safe action and rely on next replan.

## Non-Goals (Phase 1)

- No semantic task-completion inference system.
- No heuristic completion labeling (`likely_completed`, `stuck`, etc.).
- No large planner redesign beyond tool-surface reduction and context strengthening.

## Success Criteria

- Planner observability tool call count: effectively zero (removed from planner registry).
- Planner LLM calls per replan: typically one.
- Worker LLM calls per step: one-shot behavior as default.
- End-to-end average step time: under 10 seconds in target evaluation runs.

## Validation Plan

- Run planner-worker episodes with debug + observability logs.
- Use existing performance analysis pipeline to compare:
  - calls per role
  - tool-call distribution
  - per-step latency
- Confirm task assignment quality remains serviceable for iterative experiments.

## Risks and Mitigations

- Risk: Planner quality drops without live tool probes.  
  Mitigation: strengthen deterministic planner context payload and prompt constraints.

- Risk: Worker one-shot decisions make occasional local mistakes.  
  Mitigation: strict fallback behavior and frequent replanning.

- Risk: malformed planner outputs can stall coordination.  
  Mitigation: enforce both-worker assignment contract and fallback retention logic.
