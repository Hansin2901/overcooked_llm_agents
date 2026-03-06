# Planner-Worker Latency Reduction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce planner-worker runtime to under 10 seconds per environment step by removing planner observability tool loops and converting workers to one-shot action selection.

**Architecture:** Keep planner orchestration and task assignment flow intact, but prune planner tools to assignment-only and move planner observability into deterministic prompt context. Replace worker ReAct tool-loop execution with one-shot structured action output plus strict parsing/fallback. Preserve compatibility with existing `Task`, `Planner`, and `WorkerAgent` interfaces.

**Tech Stack:** Python 3, `langchain_core` messages/tools, `langgraph`, existing Overcooked planner/worker modules, `unittest` test suite.

---

## Skill References

- Use `@test-driven-development` for each code change.
- Use `@verification-before-completion` before claiming final success.
- Use `@systematic-debugging` if any test fails unexpectedly.

### Task 1: Prune Planner Tool Surface to Assignment-Only

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/planner_tools.py`
- Test: `testing/test_planner_tools.py`

**Step 1: Write the failing test**

Add a test asserting planner observation tool list is empty and only `assign_tasks` remains:

```python
def test_factory_creates_assignment_only_planner_tools(self):
    obs_tools, action_tools, action_tool_names = create_planner_tools(
        self.planner_tool_state, self.worker_registry
    )
    self.assertEqual(obs_tools, [])
    self.assertEqual({t.name for t in action_tools}, {"assign_tasks"})
    self.assertEqual(action_tool_names, {"assign_tasks"})
```

**Step 2: Run test to verify it fails**

Run:
```bash
uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_factory_creates_assignment_only_planner_tools -v
```

Expected: FAIL because observation tools currently include multiple entries.

**Step 3: Write minimal implementation**

In `create_planner_tools`, keep `assign_tasks` and return an empty observation list:

```python
observation_tools = []
action_tools = [assign_tasks]
action_tool_names = {"assign_tasks"}
return observation_tools, action_tools, action_tool_names
```

**Step 4: Run test to verify it passes**

Run:
```bash
uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_factory_creates_assignment_only_planner_tools -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add testing/test_planner_tools.py src/overcooked_ai_py/agents/llm/planner_tools.py
git commit -m "refactor(planner): limit planner tools to assign_tasks"
```

### Task 2: Add Deterministic Planner Context Snapshot and State-Grounded Prompt Rules

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/planner.py`
- Modify: `src/overcooked_ai_py/agents/llm/state_serializer.py`
- Test: `testing/test_planner.py`
- Test: `testing/test_state_serializer_prompts.py`

**Step 1: Write the failing tests**

Add planner prompt tests for state-grounded instruction and explicit no-completion-assumption rule:

```python
def test_planner_prompt_forbids_completion_assumptions(self):
    prompt = build_planner_system_prompt(self.mdp, self.worker_ids)
    self.assertIn("Do not assume a worker will complete", prompt)
    self.assertIn("current observed state", prompt)
```

Add planner context formatting test for worker snapshots:

```python
def test_maybe_replan_prompt_includes_worker_state_snapshot(self):
    # assert prompt contains worker position/holding/current task fields
    self.assertIn("Worker snapshots:", sent_prompt)
    self.assertIn("holding=", sent_prompt)
```

**Step 2: Run tests to verify they fail**

Run:
```bash
uv run python -m unittest \
  testing.test_state_serializer_prompts.TestPlannerSystemPrompt.test_planner_prompt_forbids_completion_assumptions \
  testing.test_planner.TestMaybeReplan.test_maybe_replan_prompt_includes_worker_state_snapshot -v
```

Expected: FAIL because prompt/context content does not yet include these constraints/sections.

**Step 3: Write minimal implementation**

- In `state_serializer.py`, update `build_planner_system_prompt` to include:
  - mandatory state-grounded planning language
  - explicit prohibition on assuming previous task completion
  - requirement to assign both workers every replan cycle
- In `planner.py`, add deterministic snapshot builder used by `maybe_replan`:

```python
def _build_worker_snapshot_text(self) -> str:
    lines = ["Worker snapshots:"]
    for wid, ts in sorted(self._worker_registry.items()):
        player = ts.state.players[ts.agent_index]
        held = player.held_object.name if player.held_object else "nothing"
        task = ts.current_task.description if ts.current_task else "none"
        steps = ts.current_task.steps_active if ts.current_task else 0
        lines.append(
            f"  {wid}: pos={player.position}, facing={player.orientation}, holding={held}, task={task}, steps_active={steps}"
        )
    return "\n".join(lines)
```

Then include this snapshot in planner `HumanMessage` prompt.

**Step 4: Run tests to verify they pass**

Run:
```bash
uv run python -m unittest \
  testing.test_state_serializer_prompts.TestPlannerSystemPrompt.test_planner_prompt_forbids_completion_assumptions \
  testing.test_planner.TestMaybeReplan.test_maybe_replan_prompt_includes_worker_state_snapshot -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/planner.py src/overcooked_ai_py/agents/llm/state_serializer.py testing/test_planner.py testing/test_state_serializer_prompts.py
git commit -m "feat(planner): add state-grounded context snapshot and prompt constraints"
```

### Task 3: Convert Worker Decision Path to One-Shot Structured Action Output

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/worker_agent.py`
- Optionally modify: `src/overcooked_ai_py/agents/llm/graph_builder.py` (only if needed to support one-shot mode cleanly)
- Test: `testing/test_worker_agent_unit.py`

**Step 1: Write the failing tests**

Add test verifying worker action path performs one-shot parse without requiring observation tool loop:

```python
def test_worker_one_shot_action_parsing(self):
    # mock one model response with {"action":"move_up"}
    action, _ = self.worker.action(state)
    self.assertEqual(action, Direction.NORTH)
```

Add test verifying invalid output falls back safely:

```python
def test_worker_invalid_one_shot_output_falls_back_to_stay(self):
    # mock malformed output
    action, _ = self.worker.action(state)
    self.assertEqual(action, Action.STAY)
```

**Step 2: Run tests to verify they fail**

Run:
```bash
uv run python -m unittest \
  testing.test_worker_agent_unit.TestWorkerAgentUnit.test_worker_one_shot_action_parsing \
  testing.test_worker_agent_unit.TestWorkerAgentUnit.test_worker_invalid_one_shot_output_falls_back_to_stay -v
```

Expected: FAIL because worker still relies on graph tool-loop behavior.

**Step 3: Write minimal implementation**

Implement one-shot action parsing helpers in `worker_agent.py`:

```python
_ACTION_MAP = {
    "move_up": Direction.NORTH,
    "move_down": Direction.SOUTH,
    "move_left": Direction.WEST,
    "move_right": Direction.EAST,
    "interact": Action.INTERACT,
    "wait": Action.STAY,
}

def _parse_worker_action(self, text: str):
    # parse strict JSON {"action":"..."}; return mapped action or None
```

In `action()`:
- build current rich context as before
- invoke one worker model turn
- parse action
- fallback to `Action.STAY` when parsing fails

Keep existing history bookkeeping and observability emits.

**Step 4: Run tests to verify they pass**

Run:
```bash
uv run python -m unittest \
  testing.test_worker_agent_unit.TestWorkerAgentUnit.test_worker_one_shot_action_parsing \
  testing.test_worker_agent_unit.TestWorkerAgentUnit.test_worker_invalid_one_shot_output_falls_back_to_stay -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/worker_agent.py testing/test_worker_agent_unit.py
git commit -m "feat(worker): switch to one-shot structured action selection"
```

### Task 4: Ensure Planner Assigns Both Workers Every Replan

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/planner_tools.py`
- Modify: `src/overcooked_ai_py/agents/llm/state_serializer.py`
- Test: `testing/test_planner_tools.py`

**Step 1: Write the failing test**

Add validation test for partial assignment payload:

```python
def test_assign_tasks_requires_both_workers(self):
    assign_tasks = next(t for t in self.action_tools if t.name == "assign_tasks")
    result = assign_tasks.invoke({"assignments": json.dumps({"worker_0": "only one task"})})
    self.assertIn("Error", result)
    self.assertIn("both workers", result.lower())
```

**Step 2: Run test to verify it fails**

Run:
```bash
uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_assign_tasks_requires_both_workers -v
```

Expected: FAIL because partial assignment is currently accepted.

**Step 3: Write minimal implementation**

Inside `assign_tasks`, require exactly all registered worker IDs:

```python
expected_workers = set(worker_registry.keys())
if set(parsed.keys()) != expected_workers:
    return (
        "Error: assign_tasks must provide tasks for both workers. "
        f"Expected keys: {sorted(expected_workers)}"
    )
```

Also reinforce this requirement in planner system prompt output-format rules.

**Step 4: Run test to verify it passes**

Run:
```bash
uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_assign_tasks_requires_both_workers -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/planner_tools.py src/overcooked_ai_py/agents/llm/state_serializer.py testing/test_planner_tools.py
git commit -m "feat(planner): enforce two-worker assignment contract"
```

### Task 5: End-to-End Verification and Latency Check

**Files:**
- No code changes required unless regressions are found.
- Validate against existing logs/scripts:
  - `scripts/run_llm_agent.py`
  - `scripts/analyze_performance.py`

**Step 1: Run targeted unit suite**

Run:
```bash
uv run python -m unittest \
  testing.test_planner_tools \
  testing.test_planner \
  testing.test_worker_agent_unit \
  testing.test_state_serializer_prompts -v
```

Expected: PASS.

**Step 2: Run short planner-worker smoke episode**

Run:
```bash
source .env && uv run python scripts/run_llm_agent.py \
  --agent-type planner-worker \
  --layout cramped_room \
  --horizon 20 \
  --replan-interval 5 \
  --debug \
  --run-name latency-reduction-smoke \
  --tags bench,latency
```

Expected: episode completes without hangs; planner assigns both workers each replan.

**Step 3: Analyze performance output**

Run:
```bash
uv run python scripts/analyze_performance.py
```

Expected:
- planner observation tools absent
- planner LLM calls per replan near 1
- worker LLM calls per step near 1
- step time trending toward <10s target

**Step 4: If metrics regress, debug before merge**

Run focused diagnosis commands and patch only root-cause files, then re-run Task 5 Steps 1-3.

**Step 5: Final commit (if verification produced fixes)**

```bash
git add <only-files-modified-for-fixes>
git commit -m "fix: address regressions from latency optimization rollout"
```

