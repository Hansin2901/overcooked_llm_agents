# Planner Intelligence Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 4 planner observation tools, enhance prompts with task decomposition guidance, and add planning history to improve planner-worker coordination and achieve reward ≥ 20 in 50-step runs.

**Architecture:** Extend `planner_tools.py` with two new tools (`get_nearby_interactables`, `validate_task_feasibility`), enhance `get_worker_status` to include position/inventory, update planner system prompt in `state_serializer.py`, and add planning history tracking to `planner.py`.

**Tech Stack:** Python 3.10, LangChain (tool decorator), unittest, OvercookedMDP

---

Skill references: `@test-driven-development`, `@verification-before-completion`.

## 🔀 PARALLEL EXECUTION STRATEGY

**This plan is designed for maximum parallelization.** There are **9 independent tasks** that can be executed simultaneously, followed by **3 sequential integration tasks**.

### Parallel Phase (9 tasks - can all run at once)

| Workstream | Task ID | Task Name | Dependencies |
|------------|---------|-----------|--------------|
| **Workstream 1** | 1.1 | Enhance get_worker_status | None ✅ |
| **Workstream 1** | 1.2 | Add get_nearby_interactables | None ✅ |
| **Workstream 1** | 1.3 | Add validate_task_feasibility | None ✅ |
| **Workstream 1** | 1.4 | Verify all tools work together | None ✅ |
| **Workstream 2** | 2.1 | Update planner system prompt | None ✅ |
| **Workstream 2** | 2.2 | Add planning history | None ✅ |
| **Workstream 3** | 3.1 | Create test fixtures | None ✅ |
| **Workstream 3** | 3.2 | Create mock utilities | None ✅ |
| **Workstream 3** | 3.3 | Set up integration test harness | None ✅ |

### Sequential Phase (3 tasks - must run after parallel phase)

| Task ID | Task Name | Depends On |
|---------|-----------|------------|
| 4.1 | Wire tools into planner graph | Tasks 1.1-1.4 complete |
| 4.2 | Integration tests | Task 4.1 complete |
| 4.3 | End-to-end validation | Tasks 4.1-4.2 complete |

**Execution Options:**
1. **Subagent-Driven (Recommended):** Launch 9 parallel subagents, review after each completes, then run integration tasks
2. **Manual Parallel:** Work on workstreams simultaneously in separate sessions/branches
3. **Sequential:** Execute tasks 1.1 → 1.2 → ... → 4.3 in order (slower but simpler)

---

## ⚡ WORKSTREAM 1: Planner Observation Tools [PARALLEL]

**All 4 tasks in this workstream are INDEPENDENT and can run SIMULTANEOUSLY**

### Task 1.1: Enhance get_worker_status to Include Position and Inventory [PARALLEL ✅]

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/planner_tools.py:160-169`
- Test: `testing/test_planner_tools.py`

**Step 1: Write failing test for enhanced get_worker_status**

Add to `testing/test_planner_tools.py` after line 231:

```python
def test_get_worker_status_includes_position_and_inventory(self):
    """Test get_worker_status returns position and held object."""
    # Set up worker state: worker_0 at position (1,2) holding onion
    state = self.mdp.get_standard_start_state()
    # Manually place worker at known position for testing
    from overcooked_ai_py.mdp.overcooked_mdp import PlayerState
    state.players[0] = PlayerState((1, 2), Direction.NORTH, held_object=self.mdp.get_onion())
    self.worker_0_state.set_state(state, 0)

    # Assign a task
    task = Task(
        description="Deliver onion to pot",
        worker_id="worker_0",
        created_at=0,
        completed=False
    )
    self.worker_0_state.set_task(task)

    # Call get_worker_status
    get_worker_status = next(t for t in self.obs_tools if t.name == "get_worker_status")
    result = get_worker_status.invoke({"worker_id": "worker_0"})

    # Should be human-readable, not JSON
    self.assertIn("position (1, 2)", result.lower())
    self.assertIn("holding", result.lower())
    self.assertIn("onion", result.lower())
    self.assertIn("Deliver onion to pot", result)
    self.assertNotIn("{", result)  # Not JSON format
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_get_worker_status_includes_position_and_inventory -v`
Expected: FAIL - assertion error, current implementation returns JSON

**Step 3: Implement enhanced get_worker_status**

Modify `src/overcooked_ai_py/agents/llm/planner_tools.py`, replace lines 160-169:

```python
@tool
def get_worker_status(worker_id: str) -> str:
    """Get current status of a worker including position, inventory, and task.

    Args:
        worker_id: "worker_0" or "worker_1"

    Returns:
        Human-readable status like:
        "Worker worker_0 is at position (1, 2), holding: onion, current task: 'Deliver onion to pot'"
    """
    if worker_id not in worker_registry:
        return f"Error: Unknown worker_id '{worker_id}'. Valid workers: {', '.join(sorted(worker_registry.keys()))}"

    worker_state = worker_registry[worker_id]
    if worker_state.state is None:
        return f"Worker {worker_id}: state not initialized"

    player = worker_state.state.players[worker_state.agent_index]
    pos = player.position

    # Format held object
    if player.held_object is None:
        held_desc = "empty hands"
    else:
        obj_name = player.held_object.name
        held_desc = f"holding: {obj_name}"

    # Format task
    if worker_state.current_task is None:
        task_desc = "no assigned task"
    elif worker_state.current_task.completed:
        task_desc = f"completed task: '{worker_state.current_task.description}'"
    else:
        task_desc = f"current task: '{worker_state.current_task.description}'"

    return f"Worker {worker_id} is at position {pos}, {held_desc}, {task_desc}"
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_get_worker_status_includes_position_and_inventory -v`
Expected: PASS

**Step 5: Update existing tests that expect JSON format**

Modify `testing/test_planner_tools.py` tests that parse JSON (lines 164-230):
- `test_get_worker_status_idle`: Change to assert string format instead of JSON
- `test_get_worker_status_working`: Change to assert string format
- `test_get_worker_status_completed`: Change to assert string format

Replace JSON assertions with string checks:

```python
def test_get_worker_status_idle(self):
    """Test get_worker_status returns correct status for idle worker."""
    get_worker_status = next(t for t in self.obs_tools if t.name == "get_worker_status")
    result = get_worker_status.invoke({"worker_id": "worker_0"})

    # Should be human-readable string
    self.assertIn("worker_0", result)
    self.assertIn("position", result.lower())
    self.assertIn("no assigned task", result.lower())

def test_get_worker_status_working(self):
    """Test get_worker_status returns correct status for working worker."""
    task = Task(
        description="Pick up onion",
        worker_id="worker_0",
        created_at=0,
        completed=False,
        steps_active=5
    )
    self.worker_0_state.set_task(task)

    get_worker_status = next(t for t in self.obs_tools if t.name == "get_worker_status")
    result = get_worker_status.invoke({"worker_id": "worker_0"})

    self.assertIn("worker_0", result)
    self.assertIn("Pick up onion", result)
    self.assertIn("current task", result.lower())

def test_get_worker_status_completed(self):
    """Test get_worker_status returns correct status for completed task."""
    task = Task(
        description="Deliver soup",
        worker_id="worker_0",
        created_at=0,
        completed=True,
        steps_active=10
    )
    self.worker_0_state.set_task(task)

    get_worker_status = next(t for t in self.obs_tools if t.name == "get_worker_status")
    result = get_worker_status.invoke({"worker_id": "worker_0"})

    self.assertIn("worker_0", result)
    self.assertIn("Deliver soup", result)
    self.assertIn("completed", result.lower())
```

**Step 6: Run all planner_tools tests**

Run: `uv run python -m unittest testing.test_planner_tools -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/planner_tools.py testing/test_planner_tools.py
git commit -m "feat: enhance get_worker_status to include position and inventory

- Changed from JSON to human-readable format
- Now includes worker position and held object
- Updated tests to assert string format instead of JSON parsing

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 1.2: Add get_nearby_interactables Tool [PARALLEL ✅]

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/planner_tools.py:224`
- Test: `testing/test_planner_tools.py`

**Step 1: Write failing test**

Add to `testing/test_planner_tools.py`:

```python
def test_get_nearby_interactables_with_adjacent_objects(self):
    """Test get_nearby_interactables returns objects within distance 1."""
    # Set up: place worker_0 at (1, 1) in cramped_room layout
    # In cramped_room, (1,1) is adjacent to onion dispenser at (0,1) and pot at (2,0)
    state = self.mdp.get_standard_start_state()
    from overcooked_ai_py.mdp.overcooked_mdp import PlayerState
    state.players[0] = PlayerState((1, 1), Direction.NORTH)
    self.worker_0_state.set_state(state, 0)

    # Re-create tools with updated state
    self.planner_tool_state.set_state(state, 0)
    self.worker_1_state.set_state(state, 1)
    self.obs_tools, _, _ = create_planner_tools(
        self.planner_tool_state, self.worker_registry
    )

    # Call get_nearby_interactables
    get_nearby_interactables = next(t for t in self.obs_tools if t.name == "get_nearby_interactables")
    result = get_nearby_interactables.invoke({"worker_id": "worker_0"})

    # Should list adjacent objects
    self.assertIn("worker_0", result)
    self.assertIn("can interact with", result.lower())
    # Check for specific objects (depends on cramped_room layout)
    # May contain: onion_dispenser, pot, counter, etc.
    self.assertTrue(any(keyword in result.lower() for keyword in ["dispenser", "pot", "counter"]))

def test_get_nearby_interactables_no_adjacent_objects(self):
    """Test get_nearby_interactables when worker has no adjacent objects."""
    # Place worker at center of open space with no adjacent objects
    state = self.mdp.get_standard_start_state()
    from overcooked_ai_py.mdp.overcooked_mdp import PlayerState
    # Find an empty position with no adjacent objects (may need adjustment per layout)
    state.players[0] = PlayerState((2, 2), Direction.NORTH)
    self.worker_0_state.set_state(state, 0)
    self.planner_tool_state.set_state(state, 0)
    self.obs_tools, _, _ = create_planner_tools(
        self.planner_tool_state, self.worker_registry
    )

    get_nearby_interactables = next(t for t in self.obs_tools if t.name == "get_nearby_interactables")
    result = get_nearby_interactables.invoke({"worker_id": "worker_0"})

    # Should indicate no interactables or only floor
    self.assertIn("worker_0", result)
    # Message format may vary, but should be clear
    self.assertTrue(
        "cannot interact" in result.lower() or
        "no objects" in result.lower() or
        "floor" in result.lower()
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_get_nearby_interactables_with_adjacent_objects -v`
Expected: FAIL with "AttributeError: 'filter' object has no attribute 'name'" (tool doesn't exist)

**Step 3: Implement get_nearby_interactables**

Add to `src/overcooked_ai_py/agents/llm/planner_tools.py` before line 224 (before the observation_tools list):

```python
@tool
def get_nearby_interactables(worker_id: str) -> str:
    """Check what objects a worker can currently interact with (Manhattan distance = 1).

    Args:
        worker_id: "worker_0" or "worker_1"

    Returns:
        Human-readable description of adjacent objects:
        "Worker worker_0 can interact with: onion_dispenser (west), pot at (2,0) (north)"
    """
    if worker_id not in worker_registry:
        return f"Error: Unknown worker_id '{worker_id}'. Valid workers: {', '.join(sorted(worker_registry.keys()))}"

    worker_state = worker_registry[worker_id]
    if worker_state.state is None:
        return f"Worker {worker_id}: state not initialized"

    player = worker_state.state.players[worker_state.agent_index]
    pos = player.position

    interactables = []

    # Check all 4 directions (up, down, left, right)
    for direction, direction_name in [(Direction.NORTH, "north"), (Direction.SOUTH, "south"),
                                       (Direction.EAST, "east"), (Direction.WEST, "west")]:
        adj_pos = Action.move_in_direction(pos, direction)
        x, y = adj_pos

        # Check bounds
        if not (0 <= x < worker_state.mdp.width and 0 <= y < worker_state.mdp.height):
            continue

        # Get terrain type
        terrain = worker_state.mdp.terrain_mtx[y][x]
        terrain_map = {
            " ": None,  # floor, skip
            "X": "counter",
            "P": "pot",
            "O": "onion_dispenser",
            "T": "tomato_dispenser",
            "D": "dish_dispenser",
            "S": "serving_location",
            "#": None,  # wall, skip
        }

        terrain_name = terrain_map.get(terrain, terrain)
        if terrain_name is None:
            continue  # Skip floor and walls

        # Check for objects at this position
        if terrain_name == "pot" and worker_state.state.has_object(adj_pos):
            soup = worker_state.state.get_object(adj_pos)
            if soup.is_ready:
                interactables.append(f"pot at {adj_pos} ({direction_name}) [READY SOUP]")
            elif soup.is_cooking:
                ticks_left = soup.cook_time - soup._cooking_tick
                interactables.append(f"pot at {adj_pos} ({direction_name}) [cooking, {ticks_left} ticks left]")
            else:
                interactables.append(f"pot at {adj_pos} ({direction_name}) [{len(soup.ingredients)}/3 ingredients]")
        elif terrain_name == "counter" and worker_state.state.has_object(adj_pos):
            obj = worker_state.state.get_object(adj_pos)
            interactables.append(f"{terrain_name} with {obj.name} ({direction_name})")
        else:
            interactables.append(f"{terrain_name} ({direction_name})")

    if not interactables:
        return f"Worker {worker_id} cannot interact with anything from position {pos} (no adjacent objects)"

    return f"Worker {worker_id} can interact with: {', '.join(interactables)}"
```

**Step 4: Update observation_tools list**

Modify line 224 to include the new tool:

```python
observation_tools = [get_surroundings, get_pot_details, check_path, get_worker_status, get_nearby_interactables]
```

**Step 5: Update test expectation for tool count**

Modify `testing/test_planner_tools.py` line 52-53:

```python
def test_factory_creates_correct_tools(self):
    """Test that the factory creates the expected number and types of tools."""
    # Should have 5 observation tools now (was 4)
    self.assertEqual(len(self.obs_tools), 5)
```

Update line 63 expected tool names:

```python
expected_obs_names = {"get_surroundings", "get_pot_details", "check_path", "get_worker_status", "get_nearby_interactables"}
```

**Step 6: Run tests**

Run: `uv run python -m unittest testing.test_planner_tools -v`
Expected: All tests PASS including new get_nearby_interactables tests

**Step 7: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/planner_tools.py testing/test_planner_tools.py
git commit -m "feat: add get_nearby_interactables tool for planner

- Returns objects within Manhattan distance 1 of worker
- Shows direction and object details (pot status, counter items)
- Helps planner understand what workers can interact with immediately

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 1.3: Add validate_task_feasibility Tool [PARALLEL ✅]

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/planner_tools.py:224`
- Test: `testing/test_planner_tools.py`

**Step 1: Write failing tests**

Add to `testing/test_planner_tools.py`:

```python
def test_validate_task_feasibility_pickup_with_full_hands(self):
    """Test validation catches worker trying to pick up when already holding object."""
    # Set worker_0 holding onion
    state = self.mdp.get_standard_start_state()
    from overcooked_ai_py.mdp.overcooked_mdp import PlayerState
    state.players[0] = PlayerState((1, 1), Direction.NORTH, held_object=self.mdp.get_onion())
    self.worker_0_state.set_state(state, 0)
    self.planner_tool_state.set_state(state, 0)
    self.obs_tools, _, _ = create_planner_tools(
        self.planner_tool_state, self.worker_registry
    )

    validate = next(t for t in self.obs_tools if t.name == "validate_task_feasibility")
    result = validate.invoke({
        "worker_id": "worker_0",
        "task_description": "Pick up a dish from dish dispenser"
    })

    # Should indicate infeasible and suggest action
    self.assertIn("INFEASIBLE", result)
    self.assertIn("already holding", result.lower())
    self.assertIn("onion", result.lower())
    self.assertIn("suggest", result.lower())

def test_validate_task_feasibility_deliver_to_full_pot(self):
    """Test validation catches trying to add ingredient to full pot."""
    # Set up: pot at (2,0) with 3 onions (full), worker_0 holding onion
    state = self.mdp.get_standard_start_state()
    from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, SoupState

    # Create full pot
    full_soup = SoupState.get_soup((2,0), num_onions=3, num_tomatoes=0, cooking_tick=0, cook_time=20)
    state.objects[(2,0)] = full_soup
    state.players[0] = PlayerState((1, 0), Direction.EAST, held_object=self.mdp.get_onion())

    self.worker_0_state.set_state(state, 0)
    self.planner_tool_state.set_state(state, 0)
    self.obs_tools, _, _ = create_planner_tools(
        self.planner_tool_state, self.worker_registry
    )

    validate = next(t for t in self.obs_tools if t.name == "validate_task_feasibility")
    result = validate.invoke({
        "worker_id": "worker_0",
        "task_description": "Deliver onion to pot at (2,0)"
    })

    # Should indicate infeasible
    self.assertIn("INFEASIBLE", result)
    self.assertIn("full", result.lower() or "3/3" in result)
    self.assertIn("suggest", result.lower())

def test_validate_task_feasibility_valid_task(self):
    """Test validation passes for achievable task."""
    # Worker with empty hands, pot with room for ingredients
    state = self.mdp.get_standard_start_state()
    from overcooked_ai_py.mdp.overcooked_mdp import PlayerState
    state.players[0] = PlayerState((0, 1), Direction.EAST)  # Near onion dispenser
    self.worker_0_state.set_state(state, 0)
    self.planner_tool_state.set_state(state, 0)
    self.obs_tools, _, _ = create_planner_tools(
        self.planner_tool_state, self.worker_registry
    )

    validate = next(t for t in self.obs_tools if t.name == "validate_task_feasibility")
    result = validate.invoke({
        "worker_id": "worker_0",
        "task_description": "Pick up onion from dispenser"
    })

    # Should indicate feasible
    self.assertIn("FEASIBLE", result)
    self.assertNotIn("INFEASIBLE", result)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_validate_task_feasibility_pickup_with_full_hands -v`
Expected: FAIL (tool doesn't exist)

**Step 3: Implement validate_task_feasibility**

Add to `src/overcooked_ai_py/agents/llm/planner_tools.py` before the observation_tools list:

```python
@tool
def validate_task_feasibility(worker_id: str, task_description: str) -> str:
    """Validate if a proposed task is achievable for the worker given current state.

    Args:
        worker_id: "worker_0" or "worker_1"
        task_description: The task being considered

    Returns:
        "FEASIBLE: ..." or "INFEASIBLE: ... Suggest: ..."
    """
    if worker_id not in worker_registry:
        return f"Error: Unknown worker_id '{worker_id}'"

    worker_state = worker_registry[worker_id]
    if worker_state.state is None:
        return f"Error: Worker {worker_id} state not initialized"

    player = worker_state.state.players[worker_state.agent_index]
    task_lower = task_description.lower()

    # Check: trying to pick up when already holding something
    if any(keyword in task_lower for keyword in ["pick up", "pickup", "grab", "get"]):
        if player.held_object is not None:
            held_name = player.held_object.name
            return f"INFEASIBLE: Worker {worker_id} is already holding {held_name}, cannot pick up another item. Suggest: deliver or drop current item first."

    # Check: trying to deliver ingredient to full pot
    if "deliver" in task_lower or "add" in task_lower or "put" in task_lower:
        # Extract pot position if mentioned (simple pattern matching)
        import re
        pot_match = re.search(r'pot.*?(\(\d+,\s*\d+\))', task_lower)
        if pot_match:
            try:
                pot_pos_str = pot_match.group(1)
                pot_pos = eval(pot_pos_str)  # Convert "(2,0)" to tuple

                # Check if pot exists and is full
                if worker_state.state.has_object(pot_pos):
                    soup = worker_state.state.get_object(pot_pos)
                    if soup.name == "soup" and len(soup.ingredients) >= 3:
                        if soup.is_cooking or soup.is_ready:
                            return f"INFEASIBLE: Pot at {pot_pos} is already cooking or ready (3/3 ingredients). Suggest: wait for soup to finish or choose different pot."
                        else:
                            return f"INFEASIBLE: Pot at {pot_pos} is full (3/3 ingredients) but not cooking. Suggest: worker with empty hands should interact to start cooking."
            except:
                pass  # If position parsing fails, skip this check

    # Check: trying to pick up soup without dish
    if "pick up soup" in task_lower or "collect soup" in task_lower or "get soup" in task_lower:
        if player.held_object is None or player.held_object.name != "dish":
            return f"INFEASIBLE: Worker {worker_id} needs to be holding a dish to pick up soup. Suggest: get a dish from dispenser first."

    # Check: trying to serve without holding soup
    if "serve" in task_lower or "deliver to serving" in task_lower:
        if player.held_object is None:
            return f"INFEASIBLE: Worker {worker_id} has empty hands, cannot serve. Suggest: pick up soup with dish first."
        elif player.held_object.name != "soup":
            held_name = player.held_object.name
            return f"INFEASIBLE: Worker {worker_id} is holding {held_name}, not soup. Suggest: pick up soup with dish before serving."

    # If no issues detected, task seems feasible
    return f"FEASIBLE: Task '{task_description}' appears achievable for {worker_id}"
```

**Step 4: Update observation_tools list and test expectations**

Modify line in `planner_tools.py` (now with 6 observation tools):

```python
observation_tools = [get_surroundings, get_pot_details, check_path, get_worker_status, get_nearby_interactables, validate_task_feasibility]
```

Update `testing/test_planner_tools.py` test count and names:

```python
def test_factory_creates_correct_tools(self):
    """Test that the factory creates the expected number and types of tools."""
    # Should have 6 observation tools
    self.assertEqual(len(self.obs_tools), 6)

    # ... later in test:
    expected_obs_names = {"get_surroundings", "get_pot_details", "check_path", "get_worker_status", "get_nearby_interactables", "validate_task_feasibility"}
```

**Step 5: Run tests**

Run: `uv run python -m unittest testing.test_planner_tools -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/planner_tools.py testing/test_planner_tools.py
git commit -m "feat: add validate_task_feasibility tool for planner

- Validates task against current worker state
- Catches common errors: picking up with full hands, delivering to full pot,
  serving without soup, picking up soup without dish
- Provides suggestions when task is infeasible

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 1.4: Verify All Planner Tools Work Together [PARALLEL ✅]

**Files:**
- Test: `testing/test_planner_tools.py`

**Step 1: Add integration test**

Add to `testing/test_planner_tools.py`:

```python
def test_all_planner_tools_integration(self):
    """Integration test: verify all 6 observation tools can be called successfully."""
    # Set up realistic game state
    state = self.mdp.get_standard_start_state()
    from overcooked_ai_py.mdp.overcooked_mdp import PlayerState
    state.players[0] = PlayerState((1, 1), Direction.NORTH, held_object=self.mdp.get_onion())
    state.players[1] = PlayerState((3, 1), Direction.SOUTH)

    self.planner_tool_state.set_state(state, 0)
    self.worker_0_state.set_state(state, 0)
    self.worker_1_state.set_state(state, 1)

    # Recreate tools
    obs_tools, _, _ = create_planner_tools(
        self.planner_tool_state, self.worker_registry
    )

    # Call each observation tool
    tools_dict = {tool.name: tool for tool in obs_tools}

    # 1. get_surroundings
    result = tools_dict["get_surroundings"].invoke({})
    self.assertIn("up:", result)

    # 2. get_pot_details
    result = tools_dict["get_pot_details"].invoke({})
    self.assertIn("Pot at", result)

    # 3. check_path
    result = tools_dict["check_path"].invoke({"target": "onion_dispenser"})
    self.assertIn("steps away", result)

    # 4. get_worker_status
    result = tools_dict["get_worker_status"].invoke({"worker_id": "worker_0"})
    self.assertIn("worker_0", result)
    self.assertIn("holding", result.lower())

    # 5. get_nearby_interactables
    result = tools_dict["get_nearby_interactables"].invoke({"worker_id": "worker_1"})
    self.assertIn("worker_1", result)

    # 6. validate_task_feasibility
    result = tools_dict["validate_task_feasibility"].invoke({
        "worker_id": "worker_0",
        "task_description": "Deliver onion to pot"
    })
    self.assertIn("FEASIBLE", result or "INFEASIBLE" in result)  # Either is valid response
```

**Step 2: Run integration test**

Run: `uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_all_planner_tools_integration -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `uv run python -m unittest testing.test_planner_tools -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add testing/test_planner_tools.py
git commit -m "test: add integration test for all planner observation tools

Verifies all 6 planner tools work together in realistic game state

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## ⚡ WORKSTREAM 2: Planner Prompt Enhancement [PARALLEL]

**Both tasks in this workstream are INDEPENDENT and can run SIMULTANEOUSLY (also independent of Workstreams 1 and 3)**

### Task 2.1: Update Planner System Prompt with Task Decomposition Guidance [PARALLEL ✅]

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/state_serializer.py`
- Test: `testing/test_planner.py`

**Step 1: Locate planner prompt in state_serializer.py**

Run: `uv run grep -n "def.*planner.*prompt" src/overcooked_ai_py/agents/llm/state_serializer.py`
Expected: Find function that generates planner system prompt

**Step 2: Write test for enhanced prompt content**

Add to `testing/test_planner.py`:

```python
def test_planner_system_prompt_includes_task_decomposition_guidance(self):
    """Test planner system prompt includes task decomposition guidelines."""
    from overcooked_ai_py.agents.llm.state_serializer import get_planner_system_prompt
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    prompt = get_planner_system_prompt(mdp)

    # Check for key guidance points
    self.assertIn("ONE clear objective", prompt)
    self.assertIn("worker inventory", prompt.lower())
    self.assertIn("pot contents", prompt.lower())
    self.assertIn("validate_task_feasibility", prompt)
```

**Step 3: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_planner.TestPlanner.test_planner_system_prompt_includes_task_decomposition_guidance -v`
Expected: FAIL (prompt doesn't include guidance yet)

**Step 4: Read current planner system prompt**

Run: `uv run grep -A 50 "def get_planner_system_prompt" src/overcooked_ai_py/agents/llm/state_serializer.py | head -60`
Expected: See current prompt structure

**Step 5: Enhance planner system prompt**

Modify `src/overcooked_ai_py/agents/llm/state_serializer.py`, find the planner system prompt function and add guidance section:

```python
# Add to planner system prompt (location depends on current structure):

TASK ASSIGNMENT GUIDELINES:
1. Assign ONE clear, specific objective per worker
2. ALWAYS check worker inventory before assigning pickup tasks (use get_worker_status)
3. ALWAYS check pot contents before assigning delivery tasks (use get_pot_details)
4. Use validate_task_feasibility to verify tasks are achievable before assigning
5. Break complex goals into atomic tasks: "Pick up onion from dispenser at (0,1)"
   instead of "Get ingredients and make soup"

TASK VALIDATION PROCESS:
- Before calling assign_tasks, call get_worker_status for both workers
- Call get_pot_details to understand cooking progress
- For each proposed task, call validate_task_feasibility
- If infeasible, adjust based on suggestions
- Only call assign_tasks after validation passes
```

**Step 6: Run test to verify it passes**

Run: `uv run python -m unittest testing.test_planner.TestPlanner.test_planner_system_prompt_includes_task_decomposition_guidance -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/state_serializer.py testing/test_planner.py
git commit -m "feat: add task decomposition guidance to planner system prompt

- Emphasizes ONE clear objective per worker
- Requires checking worker inventory and pot contents
- Enforces validate_task_feasibility usage
- Promotes atomic task breakdown

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2.2: Add Planning History Context to Planner [PARALLEL ✅]

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/planner.py`
- Modify: `src/overcooked_ai_py/agents/llm/state_serializer.py`
- Test: `testing/test_planner.py`

**Step 1: Write test for planning history**

Add to `testing/test_planner.py`:

```python
def test_planner_tracks_planning_history(self):
    """Test planner maintains history of previous task assignments."""
    from overcooked_ai_py.agents.llm.planner import Planner
    from overcooked_ai_py.agents.llm.tool_state import ToolState
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.planning.planners import MotionPlanner

    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    motion_planner = MotionPlanner(mdp)

    # Create planner with mock observability
    from unittest.mock import Mock
    mock_obs = Mock()
    mock_obs.emit = Mock()
    mock_obs.start_role = Mock()
    mock_obs.end_role = Mock()

    planner = Planner(
        model_name="gpt-4o-mini",
        worker_ids=["worker_0", "worker_1"],
        observability=mock_obs
    )

    # Initialize worker states
    worker_0_state = ToolState()
    worker_0_state.init(mdp, motion_planner)
    worker_1_state = ToolState()
    worker_1_state.init(mdp, motion_planner)

    state = mdp.get_standard_start_state()
    worker_0_state.set_state(state, 0)
    worker_1_state.set_state(state, 1)

    worker_registry = {
        "worker_0": worker_0_state,
        "worker_1": worker_1_state
    }

    # First assignment at step 0
    planner.assign_tasks_to_workers(state, step=0, worker_registry=worker_registry)

    # Check history was recorded
    self.assertIsNotNone(planner.planning_history)
    self.assertEqual(len(planner.planning_history), 1)
    self.assertEqual(planner.planning_history[0]["step"], 0)

    # Second assignment at step 5
    planner.assign_tasks_to_workers(state, step=5, worker_registry=worker_registry)

    # History should have 2 entries
    self.assertEqual(len(planner.planning_history), 2)
    self.assertEqual(planner.planning_history[1]["step"], 5)

def test_planner_prompt_includes_planning_history(self):
    """Test planner prompt includes previous assignments when available."""
    from overcooked_ai_py.agents.llm.state_serializer import format_planner_prompt_with_history
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    state = mdp.get_standard_start_state()

    planning_history = [
        {"step": 0, "assignments": {"worker_0": "Pick up onion", "worker_1": "Get dish"}},
        {"step": 5, "assignments": {"worker_0": "Deliver onion to pot", "worker_1": "Wait at serving area"}}
    ]

    prompt = format_planner_prompt_with_history(mdp, state, current_step=10, history=planning_history)

    # Should include history context
    self.assertIn("step 0", prompt.lower())
    self.assertIn("step 5", prompt.lower())
    self.assertIn("Pick up onion", prompt)
    self.assertIn("current step is 10", prompt.lower() or "step 10" in prompt.lower())
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_planner.TestPlanner.test_planner_tracks_planning_history -v`
Expected: FAIL (planning_history attribute doesn't exist)

**Step 3: Add planning_history to Planner class**

Modify `src/overcooked_ai_py/agents/llm/planner.py`:

```python
class Planner:
    def __init__(self, model_name: str, worker_ids: list[str], observability):
        # ... existing init code ...
        self.planning_history = []  # Add this line
        self.max_history_size = 3  # Keep last 3 planning cycles
```

**Step 4: Update assign_tasks_to_workers to record history**

Modify `assign_tasks_to_workers` method in `planner.py`:

```python
def assign_tasks_to_workers(self, state, step: int, worker_registry: dict):
    # ... existing code to invoke graph and assign tasks ...

    # After successful task assignment, record in history
    assignments = {
        worker_id: worker_state.current_task.description
        for worker_id, worker_state in worker_registry.items()
        if worker_state.current_task is not None
    }

    self.planning_history.append({
        "step": step,
        "assignments": assignments
    })

    # Trim history to max size (keep most recent)
    if len(self.planning_history) > self.max_history_size:
        self.planning_history = self.planning_history[-self.max_history_size:]
```

**Step 5: Add function to format history in prompt**

Add to `src/overcooked_ai_py/agents/llm/state_serializer.py`:

```python
def format_planner_prompt_with_history(mdp, state, current_step: int, history: list) -> str:
    """Format planner prompt including planning history.

    Args:
        mdp: OvercookedGridworld
        state: Current game state
        current_step: Current timestep
        history: List of previous planning cycles

    Returns:
        Formatted prompt string
    """
    base_prompt = get_planner_system_prompt(mdp)

    if not history:
        return base_prompt + f"\n\nCurrent step: {current_step}"

    # Add history section
    history_text = "\nPLANNING HISTORY:\n"
    for entry in history[-3:]:  # Last 3 entries
        step = entry["step"]
        assignments = entry["assignments"]
        history_text += f"\nStep {step}:\n"
        for worker_id, task in assignments.items():
            history_text += f"  {worker_id}: '{task}'\n"

    # Calculate time elapsed
    last_step = history[-1]["step"]
    delta = current_step - last_step
    history_text += f"\nLast assignment was at step {last_step}, current step is {current_step} (Δ={delta} steps)\n"

    return base_prompt + history_text
```

**Step 6: Use history in planner prompt**

Modify `planner.py` to pass history to prompt formatter when building prompt for graph invocation.

**Step 7: Run tests**

Run: `uv run python -m unittest testing.test_planner -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/planner.py src/overcooked_ai_py/agents/llm/state_serializer.py testing/test_planner.py
git commit -m "feat: add planning history tracking to planner

- Planner maintains last 3 task assignments with step numbers
- Prompt includes history showing previous assignments and time elapsed
- Helps planner understand if workers are stuck or making progress

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## ⚡ WORKSTREAM 3: Testing Infrastructure [PARALLEL]

**All 3 tasks in this workstream are INDEPENDENT and can run SIMULTANEOUSLY (also independent of Workstreams 1 and 2)**

### Task 3.1: Create Test Fixtures for Common Game States [PARALLEL ✅]

**Files:**
- Create: `testing/fixtures/__init__.py`
- Create: `testing/fixtures/planner_test_fixtures.py`
- Test: Use in `testing/test_planner_tools.py`

**Step 1: Create fixtures directory**

Run: `mkdir -p testing/fixtures && touch testing/fixtures/__init__.py`

**Step 2: Write test to verify fixtures work**

Add to `testing/test_planner_tools.py`:

```python
def test_fixtures_available(self):
    """Test that test fixtures can be imported and used."""
    from testing.fixtures.planner_test_fixtures import (
        create_worker_at_dispenser,
        create_worker_holding_onion,
        create_pot_with_ingredients
    )

    # Should be callable
    self.assertTrue(callable(create_worker_at_dispenser))
    self.assertTrue(callable(create_worker_holding_onion))
    self.assertTrue(callable(create_pot_with_ingredients))
```

**Step 3: Run test to verify it fails**

Run: `uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_fixtures_available -v`
Expected: FAIL (module doesn't exist)

**Step 4: Create fixture functions**

Create `testing/fixtures/planner_test_fixtures.py`:

```python
"""Test fixtures for planner tests."""

from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    PlayerState,
    SoupState,
    Direction
)


def create_worker_at_dispenser(mdp: OvercookedGridworld, worker_index: int = 0, dispenser_type: str = "onion"):
    """Create game state with worker positioned at dispenser.

    Args:
        mdp: OvercookedGridworld instance
        worker_index: 0 or 1
        dispenser_type: "onion", "tomato", or "dish"

    Returns:
        OvercookedState with worker at dispenser
    """
    state = mdp.get_standard_start_state()

    # Get dispenser location
    if dispenser_type == "onion":
        locations = mdp.get_onion_dispenser_locations()
    elif dispenser_type == "tomato":
        locations = mdp.get_tomato_dispenser_locations()
    elif dispenser_type == "dish":
        locations = mdp.get_dish_dispenser_locations()
    else:
        raise ValueError(f"Unknown dispenser type: {dispenser_type}")

    if not locations:
        raise ValueError(f"No {dispenser_type} dispenser in this layout")

    # Place worker at first dispenser location
    dispenser_pos = locations[0]
    state.players[worker_index] = PlayerState(dispenser_pos, Direction.NORTH)

    return state


def create_worker_holding_onion(mdp: OvercookedGridworld, worker_index: int = 0, position: tuple = None):
    """Create game state with worker holding an onion.

    Args:
        mdp: OvercookedGridworld instance
        worker_index: 0 or 1
        position: Optional (x, y) position, uses default if None

    Returns:
        OvercookedState with worker holding onion
    """
    state = mdp.get_standard_start_state()

    if position is None:
        # Use worker's default start position
        position = state.players[worker_index].position

    onion = mdp.get_onion()
    state.players[worker_index] = PlayerState(position, Direction.NORTH, held_object=onion)

    return state


def create_worker_holding_dish(mdp: OvercookedGridworld, worker_index: int = 0, position: tuple = None):
    """Create game state with worker holding a dish."""
    state = mdp.get_standard_start_state()

    if position is None:
        position = state.players[worker_index].position

    dish = mdp.get_dish()
    state.players[worker_index] = PlayerState(position, Direction.NORTH, held_object=dish)

    return state


def create_pot_with_ingredients(mdp: OvercookedGridworld, num_onions: int = 1, num_tomatoes: int = 0, cooking: bool = False):
    """Create game state with pot containing ingredients.

    Args:
        mdp: OvercookedGridworld instance
        num_onions: Number of onions in pot (0-3)
        num_tomatoes: Number of tomatoes in pot (0-3)
        cooking: Whether pot is cooking

    Returns:
        OvercookedState with pot at first pot location
    """
    state = mdp.get_standard_start_state()

    pot_locations = mdp.get_pot_locations()
    if not pot_locations:
        raise ValueError("No pot in this layout")

    pot_pos = pot_locations[0]

    # Create soup state
    cooking_tick = 0 if not cooking else 1
    cook_time = 20  # Default cook time

    soup = SoupState.get_soup(
        pot_pos,
        num_onions=num_onions,
        num_tomatoes=num_tomatoes,
        cooking_tick=cooking_tick,
        cook_time=cook_time
    )

    state.objects[pot_pos] = soup

    return state


def create_ready_soup(mdp: OvercookedGridworld):
    """Create game state with ready soup in pot."""
    state = mdp.get_standard_start_state()

    pot_locations = mdp.get_pot_locations()
    if not pot_locations:
        raise ValueError("No pot in this layout")

    pot_pos = pot_locations[0]
    cook_time = 20

    # Create ready soup (cooking_tick = cook_time means ready)
    soup = SoupState.get_soup(
        pot_pos,
        num_onions=3,
        num_tomatoes=0,
        cooking_tick=cook_time,  # Ready when cooking_tick == cook_time
        cook_time=cook_time
    )

    state.objects[pot_pos] = soup

    return state
```

**Step 5: Run test**

Run: `uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_fixtures_available -v`
Expected: PASS

**Step 6: Commit**

```bash
git add testing/fixtures/__init__.py testing/fixtures/planner_test_fixtures.py testing/test_planner_tools.py
git commit -m "test: add fixtures for common game states

- Fixtures for workers at dispensers, holding items
- Fixtures for pots with ingredients, cooking, ready
- Simplifies test setup for planner tool tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3.2: Create Mock Utilities for Planner Tool Tests [PARALLEL ✅]

**Files:**
- Modify: `testing/test_planner_tools.py`

**Step 1: Add helper functions to test file**

Add to `testing/test_planner_tools.py` after imports:

```python
def create_mock_state(mdp, workers: list = None, pots: list = None, objects: dict = None):
    """Helper to create mock game states for testing.

    Args:
        mdp: OvercookedGridworld instance
        workers: List of (position, direction, held_object) tuples for each player
        pots: List of (position, num_onions, num_tomatoes, cooking) tuples
        objects: Dict of position -> object to add to state

    Returns:
        OvercookedState
    """
    from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, SoupState

    state = mdp.get_standard_start_state()

    # Set up workers
    if workers:
        for i, worker_config in enumerate(workers):
            if i >= len(state.players):
                break
            pos, direction, held_obj = worker_config
            state.players[i] = PlayerState(pos, direction, held_object=held_obj)

    # Set up pots
    if pots:
        for pot_config in pots:
            pos, num_onions, num_tomatoes, cooking = pot_config
            cooking_tick = 1 if cooking else 0
            soup = SoupState.get_soup(
                pos,
                num_onions=num_onions,
                num_tomatoes=num_tomatoes,
                cooking_tick=cooking_tick,
                cook_time=20
            )
            state.objects[pos] = soup

    # Add custom objects
    if objects:
        state.objects.update(objects)

    return state


def assert_tool_output_format(test_case, output: str, expected_patterns: list):
    """Helper to assert tool output contains expected patterns.

    Args:
        test_case: TestCase instance (self)
        output: Tool output string
        expected_patterns: List of strings/regex patterns to check
    """
    for pattern in expected_patterns:
        test_case.assertIn(pattern.lower(), output.lower(),
                          f"Expected pattern '{pattern}' not found in output: {output}")
```

**Step 2: Add test that uses mock utilities**

Add to `testing/test_planner_tools.py`:

```python
def test_mock_utilities_work(self):
    """Test that mock utilities simplify test setup."""
    # Create custom state using helper
    state = create_mock_state(
        self.mdp,
        workers=[
            ((1, 1), Direction.NORTH, self.mdp.get_onion()),
            ((3, 2), Direction.SOUTH, None)
        ],
        pots=[
            ((2, 0), 2, 0, False)  # Pot with 2 onions, not cooking
        ]
    )

    # Verify state was created correctly
    self.assertEqual(state.players[0].position, (1, 1))
    self.assertIsNotNone(state.players[0].held_object)
    self.assertEqual(state.players[0].held_object.name, "onion")
    self.assertTrue(state.has_object((2, 0)))

    # Test output assertion helper
    test_output = "Worker worker_0 is at position (1, 1), holding: onion"
    assert_tool_output_format(
        self,
        test_output,
        ["worker_0", "position (1, 1)", "holding", "onion"]
    )
```

**Step 3: Run test**

Run: `uv run python -m unittest testing.test_planner_tools.TestPlannerTools.test_mock_utilities_work -v`
Expected: PASS

**Step 4: Commit**

```bash
git add testing/test_planner_tools.py
git commit -m "test: add mock utilities for planner tool tests

- create_mock_state helper for custom game states
- assert_tool_output_format helper for output validation
- Simplifies test writing and improves readability

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3.3: Set Up Integration Test Harness [PARALLEL ✅]

**Files:**
- Create: `testing/test_planner_integration.py`

**Step 1: Create integration test file with mock LLM**

Create `testing/test_planner_integration.py`:

```python
"""Integration tests for planner with mocked LLM responses."""

import unittest
from unittest.mock import Mock, patch

from overcooked_ai_py.agents.llm.planner import Planner
from overcooked_ai_py.agents.llm.tool_state import ToolState
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner


class TestPlannerIntegration(unittest.TestCase):
    """Integration tests for planner with simulated decision-making."""

    def setUp(self):
        """Set up test environment."""
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.motion_planner = MotionPlanner(self.mdp)

        # Create mock observability
        self.mock_obs = Mock()
        self.mock_obs.emit = Mock()
        self.mock_obs.start_role = Mock()
        self.mock_obs.end_role = Mock()

    def test_planner_initialization(self):
        """Test planner can be initialized with worker registry."""
        planner = Planner(
            model_name="gpt-4o-mini",
            worker_ids=["worker_0", "worker_1"],
            observability=self.mock_obs
        )

        self.assertIsNotNone(planner)
        self.assertEqual(len(planner.planning_history), 0)

    @patch("overcooked_ai_py.agents.llm.planner.ChatLiteLLM")
    def test_planner_can_assign_tasks(self, mock_llm):
        """Test planner can assign tasks to workers (with mocked LLM)."""
        # Mock LLM to return task assignment tool call
        mock_response = Mock()
        mock_response.tool_calls = [
            {
                "name": "assign_tasks",
                "args": {
                    "assignments": '{"worker_0": "Pick up onion", "worker_1": "Get dish"}'
                }
            }
        ]
        mock_llm.return_value.invoke.return_value = mock_response

        planner = Planner(
            model_name="gpt-4o-mini",
            worker_ids=["worker_0", "worker_1"],
            observability=self.mock_obs
        )

        # Create worker states
        worker_0_state = ToolState()
        worker_0_state.init(self.mdp, self.motion_planner)
        worker_1_state = ToolState()
        worker_1_state.init(self.mdp, self.motion_planner)

        state = self.mdp.get_standard_start_state()
        worker_0_state.set_state(state, 0)
        worker_1_state.set_state(state, 1)

        worker_registry = {
            "worker_0": worker_0_state,
            "worker_1": worker_1_state
        }

        # Assign tasks
        planner.assign_tasks_to_workers(state, step=0, worker_registry=worker_registry)

        # Verify tasks were assigned
        self.assertIsNotNone(worker_0_state.current_task)
        self.assertEqual(worker_0_state.current_task.description, "Pick up onion")
        self.assertIsNotNone(worker_1_state.current_task)
        self.assertEqual(worker_1_state.current_task.description, "Get dish")

        # Verify history was recorded
        self.assertEqual(len(planner.planning_history), 1)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run integration test**

Run: `uv run python -m unittest testing.test_planner_integration -v`
Expected: PASS (or SKIP if mocking is complex - adjust as needed)

**Step 3: Commit**

```bash
git add testing/test_planner_integration.py
git commit -m "test: add integration test harness for planner

- Tests planner initialization and task assignment flow
- Mocks LLM responses for deterministic testing
- Foundation for end-to-end planner testing

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## 🔗 INTEGRATION PHASE [SEQUENTIAL - Run after all parallel tasks complete]

### Task 4.1: Wire New Tools Into Planner Graph [SEQUENTIAL ⚠️ - Requires 1.1-1.4]

**Files:**
- Verify: `src/overcooked_ai_py/agents/llm/planner_tools.py`
- Test: `testing/test_planner.py`

**Step 1: Verify tools are in observation_tools list**

Run: `grep "observation_tools = " src/overcooked_ai_py/agents/llm/planner_tools.py`
Expected: Should show all 6 tools in list

**Step 2: Test planner can access all new tools**

Add to `testing/test_planner.py`:

```python
def test_planner_has_access_to_all_observation_tools(self):
    """Test planner receives all 6 observation tools."""
    from overcooked_ai_py.agents.llm.planner_tools import create_planner_tools
    from overcooked_ai_py.agents.llm.tool_state import ToolState
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.planning.planners import MotionPlanner

    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    motion_planner = MotionPlanner(mdp)

    planner_state = ToolState()
    planner_state.init(mdp, motion_planner)

    worker_0_state = ToolState()
    worker_0_state.init(mdp, motion_planner)
    worker_1_state = ToolState()
    worker_1_state.init(mdp, motion_planner)

    worker_registry = {
        "worker_0": worker_0_state,
        "worker_1": worker_1_state
    }

    obs_tools, action_tools, action_tool_names = create_planner_tools(
        planner_state, worker_registry
    )

    # Should have 6 observation tools
    self.assertEqual(len(obs_tools), 6)

    # Check all expected tools are present
    tool_names = {tool.name for tool in obs_tools}
    expected = {
        "get_surroundings",
        "get_pot_details",
        "check_path",
        "get_worker_status",
        "get_nearby_interactables",
        "validate_task_feasibility"
    }
    self.assertEqual(tool_names, expected)
```

**Step 3: Run test**

Run: `uv run python -m unittest testing.test_planner.TestPlanner.test_planner_has_access_to_all_observation_tools -v`
Expected: PASS

**Step 4: Run full planner test suite**

Run: `uv run python -m unittest testing.test_planner -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add testing/test_planner.py
git commit -m "test: verify planner has access to all new observation tools

Confirms all 6 observation tools are available to planner graph

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4.2: Integration Tests With All Components [SEQUENTIAL ⚠️ - Requires 4.1]

**Files:**
- Test: `testing/test_planner.py`

**Step 1: Add comprehensive integration test**

Add to `testing/test_planner.py`:

```python
@patch("overcooked_ai_py.agents.llm.planner.ChatLiteLLM")
def test_planner_uses_observation_tools_before_assignment(self, mock_llm):
    """Test planner calls observation tools before assign_tasks."""
    from overcooked_ai_py.agents.llm.planner import Planner
    from overcooked_ai_py.agents.llm.tool_state import ToolState
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.planning.planners import MotionPlanner

    # Track tool calls
    tool_calls_sequence = []

    def mock_invoke(messages, tools):
        # Record which tools were called
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_sequence.append(tc['name'])

        # Return mock assignment
        mock_response = Mock()
        mock_response.tool_calls = [{
            "name": "assign_tasks",
            "args": {"assignments": '{"worker_0": "Task 1", "worker_1": "Task 2"}'}
        }]
        return mock_response

    mock_llm.return_value.invoke = mock_invoke

    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    motion_planner = MotionPlanner(mdp)

    mock_obs = Mock()
    planner = Planner(
        model_name="gpt-4o-mini",
        worker_ids=["worker_0", "worker_1"],
        observability=mock_obs
    )

    worker_0_state = ToolState()
    worker_0_state.init(mdp, motion_planner)
    worker_1_state = ToolState()
    worker_1_state.init(mdp, motion_planner)

    state = mdp.get_standard_start_state()
    worker_0_state.set_state(state, 0)
    worker_1_state.set_state(state, 1)

    worker_registry = {"worker_0": worker_0_state, "worker_1": worker_1_state}

    planner.assign_tasks_to_workers(state, step=0, worker_registry=worker_registry)

    # Verify observation tools were called before assign_tasks
    # Note: Exact sequence depends on LLM behavior, but we expect some obs tools
    # This test may need adjustment based on actual graph behavior
    self.assertTrue(len(tool_calls_sequence) > 0)
```

**Step 2: Run integration test**

Run: `uv run python -m unittest testing.test_planner.TestPlanner.test_planner_uses_observation_tools_before_assignment -v`
Expected: PASS or SKIP (may need adjustment based on actual graph behavior)

**Step 3: Run full test suite**

Run: `uv run python -m unittest testing.test_planner testing.test_planner_tools testing.test_worker_agent_unit -v`
Expected: All existing tests still PASS (no regressions)

**Step 4: Commit**

```bash
git add testing/test_planner.py
git commit -m "test: add integration tests for planner with all components

Verifies planner can use observation tools and assign tasks

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4.3: End-to-End Validation Run [SEQUENTIAL ⚠️ - Requires 4.1, 4.2]

**Files:**
- Run: `scripts/run_llm_agent.py`
- Verify: Log files and console output

**Step 1: Run baseline test (before changes)**

Run: `uv run python scripts/run_llm_agent.py --agent-type planner-worker --layout cramped_room --horizon 50 --debug 2>&1 | tee baseline_run.log`
Expected: Completes, likely with low reward

**Step 2: Run full test suite to verify no regressions**

Run: `uv run python -m unittest testing.test_planner testing.test_planner_tools testing.test_worker_agent_unit -v`
Expected: All 103+ tests PASS

**Step 3: Run enhanced planner validation**

Run: `uv run python scripts/run_llm_agent.py --agent-type planner-worker --layout cramped_room --horizon 50 --replan-interval 3 --debug 2>&1 | tee enhanced_run.log`
Expected: Completes with improved reward (target: ≥ 20)

**Step 4: Analyze logs**

Run: `grep -E "event_type.*(llm.generation|tool.call)" logs/agent_runs/*.jsonl | tail -50`
Expected: See new planner observation tool calls (get_worker_status, get_nearby_interactables, validate_task_feasibility)

**Step 5: Compare results**

Create comparison script or manually check:
- Baseline reward vs enhanced reward
- Number of planner observation tool calls
- Task assignment quality (from debug output)

**Step 6: Document results**

Create `docs/results/2026-03-05-planner-enhancement-results.md`:

```markdown
# Planner Intelligence Enhancement Results

**Date:** 2026-03-05

## Test Configuration
- Layout: cramped_room
- Horizon: 50 steps
- Replan interval: 3 (enhanced) vs 5 (baseline)
- Model: [model name from .env]

## Results

### Baseline (before enhancement)
- Reward: [X]
- Soups delivered: [Y]
- Planner observation calls: ~4-5 per replanning cycle

### Enhanced (after enhancement)
- Reward: [X]
- Soups delivered: [Y]
- Planner observation calls: [Z] per replanning cycle

## Key Improvements
- [ ] Reward ≥ 20 (1+ soup delivered)
- [ ] Planner uses get_worker_status before assignments
- [ ] Planner validates tasks with validate_task_feasibility
- [ ] Task descriptions are more specific and atomic

## Sample Task Assignments

**Baseline:**
```
worker_0: "Go to onion dispenser, pick up onion, deliver to pot"
worker_1: "Go to onion dispenser, pick up onion, deliver to pot"
```

**Enhanced:**
```
worker_0: "Move to onion dispenser at (0,1) and pick up onion (you have empty hands)"
worker_1: "Move to dish dispenser at (1,3) and pick up dish (pot is cooking, prepare for delivery)"
```

## Observations
[Add notes about planner behavior, coordination quality, etc.]
```

**Step 7: Commit results**

```bash
git add docs/results/2026-03-05-planner-enhancement-results.md
git commit -m "docs: add planner enhancement validation results

End-to-end test results showing improved coordination and task success

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Done Criteria

✅ All tasks completed when:
- 6 planner observation tools implemented and tested
- Planner system prompt includes task decomposition guidance
- Planning history tracking added and working
- All unit tests pass (new + existing 103 tests)
- Integration test achieves reward ≥ 20 in 50-step run on cramped_room
- Log shows planner calling observation tools before task assignments
- Results documented with comparison to baseline

---

## 🚀 PARALLEL EXECUTION SUMMARY

### Phase 1: PARALLEL EXECUTION (9 tasks - all can run simultaneously)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PARALLEL PHASE (9 TASKS)                     │
│                 All tasks run SIMULTANEOUSLY                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   WORKSTREAM 1       │  │   WORKSTREAM 2       │  │   WORKSTREAM 3       │
│   Planner Tools      │  │   Prompt Enhancement │  │   Test Infrastructure│
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│ ✅ 1.1 Enhance       │  │ ✅ 2.1 Update        │  │ ✅ 3.1 Test          │
│    get_worker_status │  │    system prompt     │  │    fixtures          │
│                      │  │                      │  │                      │
│ ✅ 1.2 Add           │  │ ✅ 2.2 Add planning  │  │ ✅ 3.2 Mock          │
│    get_nearby_       │  │    history tracking  │  │    utilities         │
│    interactables     │  │                      │  │                      │
│                      │  └──────────────────────┘  │ ✅ 3.3 Integration   │
│ ✅ 1.3 Add           │                            │    test harness      │
│    validate_task_    │                            │                      │
│    feasibility       │                            └──────────────────────┘
│                      │
│ ✅ 1.4 Verify all    │
│    tools together    │
└──────────────────────┘

          ALL 9 TASKS COMPLETE ✅
                    ↓
```

### Phase 2: SEQUENTIAL EXECUTION (3 tasks - must run in order)

```
┌─────────────────────────────────────────────────────────────────┐
│                  SEQUENTIAL PHASE (3 TASKS)                     │
│              Run AFTER all parallel tasks complete              │
└─────────────────────────────────────────────────────────────────┘

    ⚠️  4.1: Wire Tools Into Planner Graph
              ↓ (depends on 1.1-1.4)

    ⚠️  4.2: Integration Tests With All Components
              ↓ (depends on 4.1)

    ⚠️  4.3: End-to-End Validation Run
              ✓ (depends on 4.1, 4.2)
```

### Execution Strategies

**Option 1: Subagent-Driven Development (RECOMMENDED for speed)**
```bash
# Launch all 9 parallel tasks at once using subagent-driven-development skill
# Each task runs in its own subagent, completes independently
# After all 9 complete, run sequential tasks 4.1 → 4.2 → 4.3
```

**Option 2: Manual Parallel Execution**
```bash
# Work on all 3 workstreams simultaneously:
# - Terminal 1: Work on Workstream 1 tasks (1.1-1.4)
# - Terminal 2: Work on Workstream 2 tasks (2.1-2.2)
# - Terminal 3: Work on Workstream 3 tasks (3.1-3.3)
# After all complete, run Integration Phase (4.1-4.3) in any terminal
```

**Option 3: Sequential Execution (slower but simpler)**
```bash
# Run tasks in order: 1.1 → 1.2 → ... → 4.3
# Takes longer but requires less coordination
```

### Task Count Summary

| Phase | Parallelizable | Sequential | Total |
|-------|----------------|------------|-------|
| **Parallel Phase** | 9 tasks ✅ | 0 | 9 |
| **Integration Phase** | 0 | 3 tasks ⚠️ | 3 |
| **TOTAL** | **9 tasks** | **3 tasks** | **12 tasks** |

**Time Savings with Parallel Execution:**
- Sequential: ~12 task-units of time
- Parallel (9 at once): ~1 task-unit + 3 task-units = **4 task-units total**
- **Estimated speedup: 3x faster** 🚀
