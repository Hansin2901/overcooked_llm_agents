# Planner Intelligence Enhancement Design

**Date:** 2026-03-05
**Goal:** Improve planner-worker task success rate by enhancing planner's state awareness, task decomposition, and validation capabilities.
**Context:** Analysis of run `6513850150d34889bf969d0856f47fa5` (50 steps, 0 reward) revealed poor coordination: planner made only 4 task assignments with repetitive, vague tasks that didn't adapt to game state.

---

## Problem Statement

**Current Issues:**
1. **Zero task completion:** 50-step run achieved 0 reward (no soups delivered)
2. **Vague task assignments:** Tasks like "Go to onion dispenser, pick up onion, deliver to pot" don't account for worker inventory or pot state
3. **No state validation:** Planner assigns tasks without checking if they're feasible (e.g., asking worker holding dish to pick up onion)
4. **Poor adaptation:** Only 4 task assignments in 50 steps, tasks remained repetitive despite lack of progress

**Root Cause:**
Planner lacks observation tools to understand game state, leading to blind task assignment and poor coordination.

---

## Design Overview

### Architecture Changes

**Current Architecture:**
- Planner invokes LLM every N steps (configurable via `--replan-interval`, default: 5)
- Planner has ONE tool: `assign_tasks(assignments: dict[str, str])`
- Tasks are free-form text strings
- No visibility into worker inventory, pot states, or task progress

**Proposed Architecture:**
- **Add 4 new observation tools for planner:**
  1. `get_worker_status(worker_id)` → Returns position, held item, current task
  2. `get_pot_details(position)` → Returns pot contents, cooking status, ingredients needed
  3. `get_nearby_interactables(worker_id)` → Returns what worker can interact with (within distance 1)
  4. `validate_task_feasibility(worker_id, task_description)` → Checks if task is achievable given current state

- **Keep existing architecture:**
  - FileRunLogger + ObservabilityHub for event tracking
  - `Planner.assign_tasks()` API unchanged (backward compatible)
  - Worker tools unchanged

- **Add task validation layer:**
  - Planner system prompt requires calling observation tools before `assign_tasks`
  - Planner reasoning must reference specific game state

---

## Parallel Implementation Strategy

This design is structured into **3 independent workstreams** that can be implemented in parallel, with a final integration phase.

### Workstream 1: Planner Observation Tools (4 parallel tasks)

**Dependencies:** None - all tools are independent

**Task 1.1: Implement `get_worker_status` tool**
- File: `src/overcooked_ai_py/agents/llm/planner_tools.py`
- Test file: `testing/test_planner_tools.py`
- Acceptance: Tool returns worker position, inventory, current task in human-readable format

**Task 1.2: Implement `get_pot_details` tool**
- File: `src/overcooked_ai_py/agents/llm/planner_tools.py`
- Test file: `testing/test_planner_tools.py`
- Acceptance: Tool returns pot ingredients, cooking status, readiness for interaction

**Task 1.3: Implement `get_nearby_interactables` tool**
- File: `src/overcooked_ai_py/agents/llm/planner_tools.py`
- Test file: `testing/test_planner_tools.py`
- Acceptance: Tool returns adjacent objects (distance=1) worker can interact with

**Task 1.4: Implement `validate_task_feasibility` tool**
- File: `src/overcooked_ai_py/agents/llm/planner_tools.py`
- Test file: `testing/test_planner_tools.py`
- Acceptance: Tool validates task feasibility and provides suggestions for infeasible tasks

**Parallelization:** All 4 tools can be implemented simultaneously by different agents/developers.

---

### Workstream 2: Planner Prompt Enhancement (2 parallel tasks)

**Dependencies:** None - prompt changes are independent of tool implementation

**Task 2.1: Update planner system prompt with task decomposition guidance**
- File: `src/overcooked_ai_py/agents/llm/state_serializer.py`
- Changes:
  - Add "Assign ONE clear objective per worker"
  - Add "Check worker inventory before assigning pickup tasks"
  - Add "Check pot contents before assigning delivery tasks"
  - Add "Use validate_task_feasibility before assigning tasks"
- Acceptance: Prompt includes all guidance points

**Task 2.2: Add planning history context to planner**
- File: `src/overcooked_ai_py/agents/llm/planner.py`
- File: `src/overcooked_ai_py/agents/llm/state_serializer.py`
- Changes:
  - Add planning history to ToolState (last 2-3 assignments)
  - Update prompt template to show: "Last assignment was at step X, current step is Y"
  - Show previous task assignments for context
- Acceptance: Planner prompt includes planning history

**Parallelization:** Both tasks can happen simultaneously and don't depend on Workstream 1.

---

### Workstream 3: Testing Infrastructure (3 parallel tasks)

**Dependencies:** None - test setup can happen before implementation

**Task 3.1: Create test fixtures for common game states**
- File: `testing/fixtures/planner_test_fixtures.py` (new file)
- Create fixtures:
  - Worker at dispenser with empty hands
  - Worker holding onion near pot
  - Pot with 1, 2, 3 onions
  - Pot cooking
  - Pot ready with soup
- Acceptance: Fixtures can be imported and used in tests

**Task 3.2: Create mock utilities for planner tool tests**
- File: `testing/test_planner_tools.py`
- Create helper functions:
  - `create_mock_state(workers=..., pots=..., objects=...)`
  - `assert_tool_output_format(output, expected_pattern)`
- Acceptance: Mock utilities simplify test writing

**Task 3.3: Set up integration test harness**
- File: `testing/test_planner_integration.py` (new file)
- Create test harness for end-to-end planner runs
- Mock LLM responses to simulate planner decisions
- Acceptance: Can run simulated planner episode in tests

**Parallelization:** All 3 testing tasks are independent.

---

### Integration Phase (sequential, after all workstreams complete)

**Task 4.1: Wire planner tools into planner graph**
- File: `src/overcooked_ai_py/agents/llm/planner_tools.py`
- Update `create_planner_tools()` factory to include new tools
- Acceptance: Planner can call all 4 new observation tools

**Task 4.2: Integration tests with all components**
- File: `testing/test_planner.py`
- Tests:
  - `test_planner_uses_observation_tools_before_assignment()`
  - `test_planner_adapts_tasks_based_on_state()`
  - `test_planner_validates_tasks_before_assignment()`
- Acceptance: Planner uses new tools in ReAct loop

**Task 4.3: End-to-end validation run**
- Run: `uv run python scripts/run_llm_agent.py --agent-type planner-worker --layout cramped_room --horizon 50 --debug`
- Acceptance:
  - Reward ≥ 20 (at least 1 soup delivered)
  - Log shows planner calling observation tools
  - No regression in existing tests (103 tests pass)

---

### Dependency Graph

```
Workstream 1 (Tools)     Workstream 2 (Prompts)    Workstream 3 (Testing)
    ├─ Task 1.1              ├─ Task 2.1                ├─ Task 3.1
    ├─ Task 1.2              └─ Task 2.2                ├─ Task 3.2
    ├─ Task 1.3                                         └─ Task 3.3
    └─ Task 1.4
         │                        │                          │
         └────────────────────────┴──────────────────────────┘
                                  │
                          Integration Phase
                                  │
                            ├─ Task 4.1
                            ├─ Task 4.2
                            └─ Task 4.3
```

**Total parallelizable tasks:** 9 (can run simultaneously)
**Sequential tasks:** 3 (must run after parallel work completes)

---

## Component Design

### 1. New Planner Observation Tools

**Implementation location:** `src/overcooked_ai_py/agents/llm/planner_tools.py`

#### Tool 1: `get_worker_status`
```python
@tool
def get_worker_status(worker_id: str) -> str:
    """Get current status of a worker including position, inventory, and task.

    Args:
        worker_id: "worker_0" or "worker_1"

    Returns:
        Human-readable status:
        "Worker worker_0 is at position (1, 2), holding: onion,
         current task: 'Go to pot at (2,0), deliver onion'"
    """
```

#### Tool 2: `get_pot_details`
```python
@tool
def get_pot_details(position: tuple[int, int]) -> str:
    """Get detailed pot status at a specific location.

    Args:
        position: (x, y) coordinates of pot

    Returns:
        "Pot at (2,0): contains 1 onion, needs 2 more onions,
         not cooking yet, ready for interaction"
        OR "Pot at (2,0): empty, needs 3 onions to start cooking"
        OR "Pot at (2,0): cooking (5 ticks remaining)"
        OR "No pot at position (2,0)"
    """
```

#### Tool 3: `get_nearby_interactables`
```python
@tool
def get_nearby_interactables(worker_id: str) -> str:
    """Check what objects the worker can currently interact with (distance = 1).

    Args:
        worker_id: "worker_0" or "worker_1"

    Returns:
        "Worker worker_0 can interact with: onion_dispenser (north),
         counter (south), pot at (2,0) (east)"
        OR "Worker worker_1 cannot interact with anything (not adjacent to any objects)"
    """
```

#### Tool 4: `validate_task_feasibility`
```python
@tool
def validate_task_feasibility(worker_id: str, task_description: str) -> str:
    """Validate if a proposed task is achievable for the worker.

    Args:
        worker_id: "worker_0" or "worker_1"
        task_description: The task you're considering assigning

    Returns:
        "FEASIBLE: Worker can execute this task"
        OR "INFEASIBLE: Worker is already holding dish, cannot pick up onion.
            Suggest: deliver dish first"
        OR "INFEASIBLE: Pot at (2,0) already has 3 onions, cannot add more.
            Suggest: wait for cooking to complete"
    """
```

---

### 2. Task Decomposition Strategy

**Problem:** Tasks are too high-level and vague.

**Solution:** Atomic Task Format - single-objective, verifiable actions.

**Good task examples:**
- "Move to onion dispenser at (0,1) and pick up onion" (worker has empty hands)
- "Move to pot at (2,0) and deliver your onion" (worker is holding onion)
- "Move to dish dispenser at (1,3) and pick up dish" (pot is cooking, prepare for delivery)
- "Move to pot at (2,0), pick up soup with your dish, then deliver to serving area at (3,3)" (pot is ready)
- "Wait near pot at (2,0) until soup finishes cooking" (worker is ready but soup not done)

**Task Assignment Flow:**
1. Planner calls `get_worker_status` for both workers
2. Planner calls `get_pot_details` for each pot location
3. Planner reasons about current game state and what needs to happen next
4. For each worker, planner formulates ONE atomic task
5. Planner calls `validate_task_feasibility` for each proposed task
6. If validation fails, planner revises task based on feedback
7. Planner calls `assign_tasks` with validated tasks

**Prompt Enhancement:**
Update planner system prompt in `state_serializer.py` to emphasize:
- "Assign ONE clear objective per worker"
- "Check worker inventory before assigning pickup tasks"
- "Check pot contents before assigning delivery tasks"
- "Use validate_task_feasibility before assigning tasks"

---

### 3. Adaptive Replanning Logic

**Current Behavior:**
- Planner replans every N steps (configurable via `--replan-interval`, default: 5)
- No awareness of whether tasks succeeded or failed
- Each replanning cycle starts fresh without learning from previous cycles

**Proposed Enhancement:**

**Focus:** Make each replanning cycle MORE EFFECTIVE, not more frequent.

**Changes:**
1. **Add planning history context:**
   - Include in planner prompt: "Last assignment was at step X, current step is Y"
   - Show previous task assignments: "Previously assigned: worker_0: 'pick up onion', worker_1: 'pick up onion'"
   - This helps planner see if workers are making progress or stuck

2. **Require observation before assignment:**
   - Planner system prompt enforces: "Before calling assign_tasks, you MUST call get_worker_status for both workers"
   - Prevents blind task assignment without checking current state

3. **No change to replan_interval default:**
   - Keep default at 5 (user can tune with `--replan-interval 3` if desired)
   - Focus on tool quality, not replanning frequency

**Implementation:**
- Update planner prompt in `state_serializer.py`
- Add planning history to planner's ToolState
- No changes to replanning interval logic

---

## Testing Strategy

### Unit Tests (TDD approach)

**1. Test new planner tools** (`testing/test_planner_tools.py`):
- `test_get_worker_status_returns_position_and_inventory()`
- `test_get_pot_details_shows_ingredients_and_cooking_status()`
- `test_get_nearby_interactables_checks_adjacency()`
- `test_validate_task_feasibility_catches_invalid_tasks()`
- Mock OvercookedState, verify tool output format

**2. Test planner with new tools** (`testing/test_planner.py`):
- `test_planner_can_call_observation_tools()`
- `test_planner_uses_worker_status_before_assignment()`
- `test_planner_adapts_tasks_based_on_pot_status()`
- Verify planner invokes observation tools in ReAct loop

**3. Test planning history context** (`testing/test_planner.py`):
- `test_planner_sees_previous_assignments_in_prompt()`
- `test_planner_knows_time_elapsed_since_last_plan()`

### Integration Tests

**4. End-to-end run validation** (`scripts/run_llm_agent.py`):
- Run 50-step episode with new planner tools
- Verify: reward > 0 (at least 1 soup delivered)
- Verify: planner observation tool calls appear in logs
- Compare token usage: should be similar or slightly higher (more observation calls)

### Success Metrics
- ✅ All unit tests pass
- ✅ Integration test achieves **reward ≥ 20** (1 soup) in 50 steps on cramped_room
- ✅ Log shows planner calling observation tools before each assign_tasks
- ✅ No regression in existing planner/worker tests (103 tests)

---

## Implementation Files

**New files:**
1. `testing/fixtures/planner_test_fixtures.py` - Test fixtures for common game states
2. `testing/test_planner_integration.py` - Integration test harness for end-to-end planner runs

**Modified files:**
1. `src/overcooked_ai_py/agents/llm/planner_tools.py` - Add 4 new observation tools
2. `src/overcooked_ai_py/agents/llm/state_serializer.py` - Update planner system prompt
3. `src/overcooked_ai_py/agents/llm/planner.py` - Add planning history to ToolState
4. `testing/test_planner_tools.py` - Add tests for new tools and mock utilities
5. `testing/test_planner.py` - Add integration tests for planner with new tools

**Unchanged:**
- Worker tools and prompts
- Runner script (CLI arguments remain same)
- Observability/logging infrastructure

---

## Future Work (Approach 3)

After Approach 1 is complete and validated, implement **Approach 3: Explicit Coordination Protocol**:
- Worker status reporting ("task complete", "task blocked", "waiting for resource")
- Explicit blocking (prevent two workers targeting same pot)
- Subtask checkpoints (confirm each subtask completion)
- Failure recovery (planner reassigns tasks when workers report blocks)

This will require:
- Event-driven replanning (not fixed interval)
- Worker-to-planner communication channel
- Blocking/conflict detection logic

---

## Risks and Mitigations

**Risk 1: Increased token usage**
- Mitigation: Observation tools add ~500-1000 tokens per replanning cycle, but should improve success rate to offset cost

**Risk 2: Planner doesn't use tools correctly**
- Mitigation: System prompt enforces tool usage pattern, unit tests verify tool behavior

**Risk 3: Task validation is too strict**
- Mitigation: validate_task_feasibility provides suggestions, not just rejection

**Risk 4: Planning history grows too large**
- Mitigation: Only include last 2-3 replanning cycles in context (sliding window)

---

## Done Criteria

- ✅ 4 new planner observation tools implemented and tested
- ✅ Planner system prompt updated with task decomposition guidance
- ✅ Planning history context added to planner prompts
- ✅ All unit tests pass (new + existing 103 tests)
- ✅ Integration test: 50-step run achieves reward ≥ 20 on cramped_room layout
- ✅ Design document committed to `docs/plans/`
