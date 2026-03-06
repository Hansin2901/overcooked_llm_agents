# Worker Memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a compact memory system to WorkerAgent so workers remember their recent actions, positions, and task boundaries across turns.

**Architecture:** Add `_history` list and helper methods to `WorkerAgent`, inject formatted history into the per-turn prompt. Follows the `LLMAgent` memory pattern but uses a compact format (~15 tokens/entry) to minimize context window impact.

**Tech Stack:** Python, unittest, existing WorkerAgent/ToolState/Action classes

---

## Dependency Graph

```
Task 1: Write ALL tests
    ↓
Task 2A: Implement _add_to_history + _format_history  ──┐
Task 2B: Wire into action() and reset()                 ├── PARALLEL (separate files)
    ↓                                                    │
    └────────────────────────────────────────────────────┘
    ↓
Task 3: Integration — run full test suite
```

- **Task 1** → sequential (must be done first to follow TDD)
- **Task 2A and 2B** → **PARALLEL** — 2A adds methods to `worker_agent.py`, 2B only wires the `action()` and `reset()` callsites. Since 2A defines the methods and 2B calls them, **2A must merge first**, then 2B applies on top. However, both can be *written* in parallel since 2B's code is fully specified in the plan.
- **Task 3** → sequential (depends on 2A + 2B both complete)

---

### Task 1: Write ALL failing tests (do this first)

**Parallelism:** SEQUENTIAL — must complete before Tasks 2A/2B.

**Files:**
- Test: `testing/test_worker_agent_unit.py`

**Step 1: Write all test cases**

Add the entire `TestWorkerMemory` class to `testing/test_worker_agent_unit.py`. This covers all three categories: storage, formatting, and integration.

```python
class TestWorkerMemory(unittest.TestCase):
    """Tests for WorkerAgent memory system."""

    def setUp(self):
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.planner = Planner(
            model_name="gpt-4o-mini",
            replan_interval=5,
            debug=False,
            horizon=400,
        )
        self.worker = WorkerAgent(
            planner=self.planner,
            worker_id="worker_0",
            model_name="gpt-4o-mini",
            debug=False,
            horizon=400,
        )

    # --- Storage tests (for Task 2A) ---

    def test_default_history_size(self):
        """WorkerAgent defaults to history_size=5."""
        self.assertEqual(self.worker.history_size, 5)
        self.assertEqual(self.worker._history, [])

    def test_custom_history_size(self):
        """WorkerAgent accepts custom history_size."""
        w = WorkerAgent(
            planner=self.planner,
            worker_id="w",
            model_name="gpt-4o-mini",
            history_size=3,
        )
        self.assertEqual(w.history_size, 3)

    def test_add_to_history(self):
        """_add_to_history stores entry with correct fields."""
        self.worker._add_to_history(
            timestep=5,
            position=(2, 1),
            action=Direction.NORTH,
            held="onion",
            task_description="Pick up onion",
        )
        self.assertEqual(len(self.worker._history), 1)
        entry = self.worker._history[0]
        self.assertEqual(entry["timestep"], 5)
        self.assertEqual(entry["position"], (2, 1))
        self.assertEqual(entry["action"], "↑")
        self.assertEqual(entry["held"], "onion")
        self.assertEqual(entry["task"], "Pick up onion")

    def test_history_trimmed_to_size(self):
        """History is trimmed when it exceeds history_size."""
        self.worker.history_size = 3
        for i in range(5):
            self.worker._add_to_history(
                timestep=i,
                position=(0, 0),
                action=Action.STAY,
                held="nothing",
                task_description="task",
            )
        self.assertEqual(len(self.worker._history), 3)
        self.assertEqual(self.worker._history[0]["timestep"], 2)

    def test_history_disabled_when_size_zero(self):
        """No entries stored when history_size=0."""
        self.worker.history_size = 0
        self.worker._add_to_history(
            timestep=0, position=(0, 0), action=Action.STAY,
            held="nothing", task_description="t",
        )
        self.assertEqual(len(self.worker._history), 0)

    # --- Formatting tests (for Task 2A) ---

    def test_format_history_empty(self):
        """Empty history returns empty string."""
        self.assertEqual(self.worker._format_history(), "")

    def test_format_history_single_entry(self):
        """Single entry formats correctly."""
        self.worker._add_to_history(
            timestep=5, position=(2, 1), action=Direction.NORTH,
            held="nothing", task_description="Pick up onion",
        )
        result = self.worker._format_history()
        self.assertIn("RECENT HISTORY:", result)
        self.assertIn("Step 5", result)
        self.assertIn("(2, 1)", result)
        self.assertIn("Pick up onion", result)

    def test_format_history_with_held_item(self):
        """Entry with held item includes it."""
        self.worker._add_to_history(
            timestep=5, position=(2, 1), action=Direction.NORTH,
            held="onion", task_description="Deliver to pot",
        )
        result = self.worker._format_history()
        self.assertIn("holding onion", result)

    def test_format_history_no_held_item_omitted(self):
        """Entry with 'nothing' held does not say 'holding nothing'."""
        self.worker._add_to_history(
            timestep=5, position=(2, 1), action=Direction.NORTH,
            held="nothing", task_description="Pick up onion",
        )
        result = self.worker._format_history()
        self.assertNotIn("holding", result)

    def test_format_history_task_boundary(self):
        """Task change inserts boundary marker."""
        self.worker._add_to_history(
            timestep=5, position=(2, 1), action=Direction.NORTH,
            held="nothing", task_description="Pick up onion",
        )
        self.worker._add_to_history(
            timestep=6, position=(2, 0), action=Action.INTERACT,
            held="onion", task_description="Deliver to pot",
        )
        result = self.worker._format_history()
        self.assertIn("--- New task ---", result)

    def test_format_history_same_task_no_boundary(self):
        """Same task across entries has no boundary marker."""
        self.worker._add_to_history(
            timestep=5, position=(2, 1), action=Direction.NORTH,
            held="nothing", task_description="Pick up onion",
        )
        self.worker._add_to_history(
            timestep=6, position=(2, 0), action=Action.INTERACT,
            held="nothing", task_description="Pick up onion",
        )
        result = self.worker._format_history()
        self.assertNotIn("--- New task ---", result)

    def test_format_history_disabled(self):
        """history_size=0 returns empty string."""
        self.worker.history_size = 0
        self.assertEqual(self.worker._format_history(), "")

    # --- Integration tests (for Task 2B) ---

    def test_action_records_history(self):
        """action() adds an entry to history."""
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        self.planner._graph = Mock()
        self.worker._graph = Mock()

        def mock_invoke(messages, **kwargs):
            self.worker._tool_state.set_action(Direction.NORTH)

        self.worker._graph.invoke = mock_invoke

        env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        env.reset()
        state = env.state

        self.worker.action(state)
        self.assertEqual(len(self.worker._history), 1)
        entry = self.worker._history[0]
        self.assertEqual(entry["timestep"], state.timestep)
        self.assertEqual(entry["action"], "↑")

    def test_action_injects_history_into_prompt(self):
        """action() includes history text in the prompt sent to graph."""
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        self.planner._graph = Mock()

        # Seed history
        self.worker._add_to_history(
            timestep=0, position=(1, 1), action=Direction.NORTH,
            held="nothing", task_description="Pick up onion",
        )

        captured_messages = {}

        def mock_invoke(messages, **kwargs):
            captured_messages["msgs"] = messages
            self.worker._tool_state.set_action(Direction.NORTH)

        self.worker._graph = Mock()
        self.worker._graph.invoke = mock_invoke

        env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        env.reset()
        state = env.state

        self.worker.action(state)

        # Check the HumanMessage contains history
        human_msg = captured_messages["msgs"]["messages"][1]
        self.assertIn("RECENT HISTORY:", human_msg.content)

    def test_reset_clears_history(self):
        """reset() clears the history list."""
        self.worker._add_to_history(
            timestep=0, position=(0, 0), action=Action.STAY,
            held="nothing", task_description="t",
        )
        self.assertEqual(len(self.worker._history), 1)
        self.worker.reset()
        self.assertEqual(len(self.worker._history), 0)
```

**Step 2: Run tests to verify they all fail**

Run: `uv run python -m pytest testing/test_worker_agent_unit.py::TestWorkerMemory -v`
Expected: FAIL — `history_size` attribute and memory methods don't exist.

**Step 3: Commit tests only**

```bash
git add testing/test_worker_agent_unit.py
git commit -m "test(worker): add failing tests for worker memory system"
```

---

### Task 2A: Implement _add_to_history and _format_history methods

**Parallelism:** Can be written in PARALLEL with Task 2B. Must be **merged first** since 2B calls these methods.

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/worker_agent.py` (add `history_size`/`_history` to `__init__`, add two new methods)

**Step 1: Add `history_size` param to `__init__` and `_history` list**

In `worker_agent.py`, update `__init__` signature and body:

```python
def __init__(
    self,
    planner,
    worker_id: str,
    model_name: str = "gpt-4o",
    debug: bool = False,
    horizon: Optional[int] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    observability=None,
    invoke_config: Optional[dict] = None,
    history_size: int = 5,
):
    # ... existing assignments ...
    self.history_size = history_size
    self._history = []
    # ... rest of __init__ (super().__init__()) ...
```

**Step 2: Add `_add_to_history` method**

```python
def _add_to_history(self, timestep, position, action, held, task_description):
    """Record a compact history entry after each action."""
    if self.history_size <= 0:
        return
    action_name = Action.ACTION_TO_CHAR.get(action, str(action))
    self._history.append({
        "timestep": timestep,
        "position": position,
        "action": action_name,
        "held": held,
        "task": task_description,
    })
    if len(self._history) > self.history_size:
        self._history = self._history[-self.history_size:]
```

**Step 3: Add `_format_history` method**

```python
def _format_history(self):
    """Format history entries for injection into the worker prompt.

    Uses compact format with task-boundary markers:
        RECENT HISTORY:
        - Step 12: at (2,1) → ↑ [Task: Pick up onion]
        --- New task ---
        - Step 14: at (2,0) holding onion → → [Task: Deliver to pot]
    """
    if not self._history or self.history_size <= 0:
        return ""

    lines = ["RECENT HISTORY:"]
    prev_task = None
    for entry in self._history:
        if prev_task is not None and entry["task"] != prev_task:
            lines.append("--- New task ---")
        held_str = f" holding {entry['held']}" if entry["held"] != "nothing" else ""
        lines.append(
            f"- Step {entry['timestep']}: at {entry['position']}{held_str} "
            f"→ {entry['action']} [Task: {entry['task']}]"
        )
        prev_task = entry["task"]
    return "\n".join(lines)
```

**Step 4: Run storage + formatting tests**

Run: `uv run python -m pytest testing/test_worker_agent_unit.py::TestWorkerMemory::test_default_history_size testing/test_worker_agent_unit.py::TestWorkerMemory::test_custom_history_size testing/test_worker_agent_unit.py::TestWorkerMemory::test_add_to_history testing/test_worker_agent_unit.py::TestWorkerMemory::test_history_trimmed_to_size testing/test_worker_agent_unit.py::TestWorkerMemory::test_history_disabled_when_size_zero testing/test_worker_agent_unit.py::TestWorkerMemory::test_format_history_empty testing/test_worker_agent_unit.py::TestWorkerMemory::test_format_history_single_entry testing/test_worker_agent_unit.py::TestWorkerMemory::test_format_history_with_held_item testing/test_worker_agent_unit.py::TestWorkerMemory::test_format_history_no_held_item_omitted testing/test_worker_agent_unit.py::TestWorkerMemory::test_format_history_task_boundary testing/test_worker_agent_unit.py::TestWorkerMemory::test_format_history_same_task_no_boundary testing/test_worker_agent_unit.py::TestWorkerMemory::test_format_history_disabled -v`
Expected: ALL PASS

**Step 5: Verify existing tests still pass**

Run: `uv run python -m pytest testing/test_worker_agent_unit.py::TestWorkerAgentUnit -v`
Expected: ALL PASS (no regressions)

**Step 6: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/worker_agent.py
git commit -m "feat(worker): add _add_to_history and _format_history methods"
```

---

### Task 2B: Wire history into action() and reset()

**Parallelism:** Can be written in PARALLEL with Task 2A. Must be **merged after 2A** since it calls `_format_history()` and `_add_to_history()`.

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/worker_agent.py` (`action()` and `reset()` methods only)

**Step 1: Modify `action()` — inject history into prompt**

Replace the existing prompt-building block (lines ~149-153) with:

```python
# Build history text
history_text = self._format_history()

if history_text:
    prompt = (
        f"{history_text}\n\n"
        f"Your current task: {task_text}\n\n"
        f"Current game state:\n{state_text}\n\n"
        f"Choose one action to execute your task."
    )
else:
    prompt = (
        f"Your current task: {task_text}\n\n"
        f"Current game state:\n{state_text}\n\n"
        f"Choose one action to execute your task."
    )
```

**Step 2: Modify `action()` — record history after action is chosen**

After the "Step 4: Get action" block (after `chosen = ...` and the `if chosen is None` guard), add:

```python
# Record history
player = state.players[self.agent_index]
held = player.held_object.name if player.held_object else "nothing"
self._add_to_history(
    timestep=state.timestep,
    position=player.position,
    action=chosen,
    held=held,
    task_description=task_text,
)
```

**Step 3: Modify `reset()` — clear history**

Add `self._history = []` to the `reset()` method body.

**Step 4: Run integration tests**

Run: `uv run python -m pytest testing/test_worker_agent_unit.py::TestWorkerMemory::test_action_records_history testing/test_worker_agent_unit.py::TestWorkerMemory::test_action_injects_history_into_prompt testing/test_worker_agent_unit.py::TestWorkerMemory::test_reset_clears_history -v`
Expected: ALL PASS

**Step 5: Verify ALL tests pass**

Run: `uv run python -m pytest testing/test_worker_agent_unit.py -v`
Expected: ALL PASS (existing + new)

**Step 6: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/worker_agent.py
git commit -m "feat(worker): wire memory into action() and reset()"
```

---

### Task 3: Full regression test suite

**Parallelism:** SEQUENTIAL — depends on Tasks 2A + 2B both complete.

**Files:** None (verification only)

**Step 1: Run worker unit tests**

Run: `uv run python -m pytest testing/test_worker_agent_unit.py -v`
Expected: ALL PASS

**Step 2: Run all planner-worker tests**

Run: `uv run python -m unittest testing.test_planner testing.test_worker_agent_unit testing.test_planner_tools testing.test_worker_tools`
Expected: ALL PASS

**Step 3: Run quick verification test**

Run: `uv run python testing/overcooked_test.py`
Expected: PASS
