# Planner-Worker Agent Architecture: Basic Framework

## Goal

Create a two-tier agent hierarchy for Overcooked:
- **Planner**: A single LLM that sees the full game state and assigns high-level tasks to each chef. It runs every N steps (not every step). It does NOT produce actions directly.
- **Workers**: One per chef/player on the floor. Each worker receives its task from the planner and produces one action per timestep. Workers **cannot communicate with each other** — they only receive instructions from the planner.

This is the **minimal viable framework** — designed so future branches can add worker specialization, smarter replanning, inter-worker messaging, etc.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Shared Planner                              │
│                         (runs every N steps)                        │
│                                                                     │
│  Sees: full game state, all player positions, all pot statuses      │
│  Tools:                                                             │
│   - assign_task(worker_id, description)                             │
│   - get_worker_status(worker_id)                                    │
│   - observation tools (read-only)                                   │
│                                                                     │
│  Outputs: task assignments cached until next replan                 │
└──────────┬──────────────────────────────┬───────────────────────────┘
           │ task for chef 0              │ task for chef 1
           ▼                              ▼
┌─────────────────────┐       ┌─────────────────────┐
│  WorkerAgent 0      │       │  WorkerAgent 1      │
│  (controls Chef 0)  │       │  (controls Chef 1)  │
│                     │       │                     │
│  Runs: EVERY step   │       │  Runs: EVERY step   │
│  Sees: its task +   │       │  Sees: its task +   │
│        game state   │       │        game state   │
│                     │       │                     │
│  Tools:             │       │  Tools:             │
│   - observe         │       │   - observe         │
│   - action (move,   │       │   - action (move,   │
│     interact, wait) │       │     interact, wait) │
│                     │       │                     │
│  ⛔ Cannot see      │       │  ⛔ Cannot see      │
│  Worker 1's task    │       │  Worker 0's task    │
│  or communicate     │       │  or communicate     │
│  with Worker 1      │       │  with Worker 0      │
└─────────────────────┘       └─────────────────────┘
           │                              │
           ▼                              ▼
      action for                     action for
      Player 0                       Player 1
           │                              │
           └──────────┬───────────────────┘
                      ▼
              env.step((action_0, action_1))
```

### How It Fits the Existing Interface

The environment expects `AgentPair(agent_0, agent_1)` where each agent independently calls `action(state)` every timestep. We keep this interface:

```python
planner = Planner(model_name="gpt-4o", ...)

worker_0 = WorkerAgent(planner, worker_id="worker_0", ...)
worker_1 = WorkerAgent(planner, worker_id="worker_1", ...)

agent_pair = AgentPair(worker_0, worker_1)  # Standard interface, no changes needed
```

- The `Planner` is a shared object (not an `Agent`). It holds the planning LLM graph and task cache.
- Each `WorkerAgent` extends `Agent`. It holds a reference to the shared planner but can only read its own task.
- When `agent_pair.joint_action(state)` is called, it calls `worker_0.action(state)` then `worker_1.action(state)`.
- The first worker to call `action()` triggers the planner (if replan is needed). The planner runs once, assigns tasks to all workers, and caches. The second worker just reads its cached task.

### Execution Timeline

```
Step 0 (replan triggered):
  worker_0.action(state)
    → planner hasn't run this step → planner runs
    → planner assigns: worker_0="get onion", worker_1="get dish"
    → worker_0 reads its task "get onion"
    → worker_0 LLM → move_left
  worker_1.action(state)
    → planner already ran this step → skip
    → worker_1 reads its task "get dish"
    → worker_1 LLM → move_right
  env.step((move_left, move_right))

Step 1 (no replan):
  worker_0.action(state)
    → planner not due → skip
    → worker_0 reads cached task "get onion"
    → worker_0 LLM → move_up
  worker_1.action(state)
    → worker_1 reads cached task "get dish"
    → worker_1 LLM → interact
  env.step((move_up, interact))

...

Step 5 (replan triggered by interval):
  worker_0.action(state)
    → planner due → planner runs again
    → planner reassigns tasks based on new state
    → ...
```

### Communication Rules

1. **Planner → Worker**: Via task cache. Planner writes tasks keyed by `worker_id`. Each worker reads only its own.
2. **Worker → Planner**: Via worker status (task steps taken, completion flag). Planner queries this when replanning.
3. **Worker ↔ Worker**: **NO communication.** Workers share a planner reference but can only access their own task. The `WorkerAgent` class has no method to read another worker's task or state.
4. **Coordination**: The planner is solely responsible. It sees all players and assigns complementary tasks (e.g., "you get onion, you get dish").

### Key Design Decisions

1. **Planner is not an Agent** — It's a shared service object. It doesn't produce actions. It runs on-demand when a worker triggers replanning.
2. **Step-based caching** — The planner tracks which timestep it last planned for. If a worker calls it on the same timestep, it returns cached results. This prevents double-planning when AgentPair calls two workers sequentially.
3. **Workers are standard Agents** — They extend `Agent`, implement `action(state)`, and work with `AgentPair` unchanged.
4. **Worker isolation enforced in code** — `WorkerAgent` only has access to `planner.get_task(self.worker_id)`. No method exists to get another worker's task.
5. **Instance-level tool state** — Each worker has its own `ToolState`. The planner has its own `ToolState`. No module-level globals.
6. **Replan interval** — Configurable. Planner runs every N steps or when a worker has no task.

## Scope

### In Scope (This Plan)
- `Planner` class (shared, not an Agent) with LangGraph + planning tools
- `WorkerAgent` class (extends Agent) with LangGraph + action tools
- `ToolState` class for instance-level tool state
- `Task` dataclass for planner→worker communication
- Planner tools: `assign_task(worker_id, desc)`, `get_worker_status(worker_id)`, observation tools
- Worker tools: observation tools + action tools (factory pattern, bound to ToolState)
- Shared `build_react_graph()` graph builder
- Planner and worker system prompts
- Step-based caching to avoid double-planning
- Integration with existing `AgentPair` and run script
- Tests

### Out of Scope (Future Branches)
- Worker-to-worker communication
- Task queues / priority systems
- Worker specialization
- Learned replanning triggers
- Inter-planner communication (two planners for two teams)

---

## Implementation Plan

### Task 1: Create Task data structure

**File:** `src/overcooked_ai_py/agents/llm/task.py` (new)

```python
from dataclasses import dataclass

@dataclass
class Task:
    """A high-level task assigned by the planner to a specific worker."""
    description: str           # Natural language task description
    worker_id: str             # Which worker this is assigned to
    created_at: int            # Timestep when created
    completed: bool = False    # Whether worker signals completion
    steps_active: int = 0      # Timesteps this task has been active
```

**Verification:** Import works, fields work as expected.

---

### Task 2: Create ToolState class

**File:** `src/overcooked_ai_py/agents/llm/tool_state.py` (new)

Each worker and the planner each get their own instance:

```python
from typing import Optional
from overcooked_ai_py.agents.llm.task import Task

class ToolState:
    """Encapsulates tool context for one agent (worker or planner).

    Each worker has its own ToolState. Workers cannot access each other's.
    """
    def __init__(self):
        self.mdp = None
        self.state = None
        self.agent_index = None
        self.motion_planner = None
        self.chosen_action = None
        self.current_task: Optional[Task] = None

    def init(self, mdp, motion_planner):
        self.mdp = mdp
        self.motion_planner = motion_planner

    def set_state(self, state, agent_index):
        self.state = state
        self.agent_index = agent_index
        self.chosen_action = None

    def set_action(self, action):
        self.chosen_action = action

    def set_task(self, task: Task):
        self.current_task = task

    def get_status(self) -> dict:
        """Return status for planner to query."""
        if self.current_task is None:
            return {"status": "idle", "task": None}
        return {
            "status": "completed" if self.current_task.completed else "working",
            "task": self.current_task.description,
            "steps_active": self.current_task.steps_active,
        }

    def reset(self):
        self.state = None
        self.agent_index = None
        self.chosen_action = None
        self.current_task = None
```

**Verification:** Unit tests for all methods.

---

### Task 3: Create worker tools (factory pattern)

**File:** `src/overcooked_ai_py/agents/llm/worker_tools.py` (new)

Each worker gets its own tool set bound to its own ToolState:

```python
def create_worker_tools(tool_state: ToolState) -> tuple:
    """Create worker tools bound to a specific ToolState.

    Returns: (observation_tools, action_tools, action_tool_names)
    """
    @tool
    def get_surroundings() -> str:
        """Check adjacent cells in all 4 directions."""
        # Same logic as current tools.py but reads from tool_state
        ...

    @tool
    def get_pot_details() -> str: ...

    @tool
    def check_path(target: str) -> str: ...

    @tool
    def move_up() -> str:
        tool_state.set_action(Direction.NORTH)
        return "Moving up"
    # ... move_down, move_left, move_right, wait, interact

    observation_tools = [get_surroundings, get_pot_details, check_path]
    action_tools = [move_up, move_down, move_left, move_right, wait, interact]
    action_tool_names = {t.name for t in action_tools}
    return observation_tools, action_tools, action_tool_names
```

**Verification:** Two workers with separate ToolStates don't share state.

---

### Task 4: Create planner tools

**File:** `src/overcooked_ai_py/agents/llm/planner_tools.py` (new)

The planner gets the worker registry (dict of worker_id → ToolState). Workers never see this registry:

```python
def create_planner_tools(
    planner_tool_state: ToolState,
    worker_registry: dict[str, ToolState],
) -> tuple:
    """Create planner tools.

    Args:
        planner_tool_state: Planner's own ToolState (for observation tools)
        worker_registry: Maps worker_id → worker's ToolState.
            Workers never see this dict — only the planner does.

    Returns: (observation_tools, action_tools, action_tool_names)
    """

    @tool
    def assign_task(worker_id: str, description: str) -> str:
        """Assign a task to a worker (chef on the floor).

        Args:
            worker_id: e.g. "worker_0", "worker_1"
            description: Clear, actionable task.
        """
        if worker_id not in worker_registry:
            available = list(worker_registry.keys())
            return f"Error: Unknown worker '{worker_id}'. Available: {available}"
        task = Task(
            description=description,
            worker_id=worker_id,
            created_at=planner_tool_state.state.timestep if planner_tool_state.state else 0,
        )
        worker_registry[worker_id].set_task(task)
        return f"Task assigned to {worker_id}: {description}"

    @tool
    def get_worker_status(worker_id: str) -> str:
        """Check a worker's current status and task progress."""
        if worker_id not in worker_registry:
            return f"Error: Unknown worker '{worker_id}'."
        return str(worker_registry[worker_id].get_status())

    # Planner observation tools (bound to planner's own ToolState)
    @tool
    def get_surroundings() -> str: ...
    @tool
    def get_pot_details() -> str: ...
    @tool
    def check_path(target: str) -> str: ...

    observation_tools = [get_surroundings, get_pot_details, check_path, get_worker_status]
    action_tools = [assign_task]
    action_tool_names = {"assign_task"}
    return observation_tools, action_tools, action_tool_names
```

**Note:** `assign_task` is the planner's "action" — it terminates the planner's ReAct loop. But the planner may need to assign tasks to multiple workers in one planning step. Options:
- Call `assign_task` multiple times (once per worker) before terminating — requires the graph to only terminate after all workers have tasks.
- Or have a single `assign_tasks` tool that takes a dict of {worker_id: description}.

For simplicity in the basic version, use `assign_tasks` (plural) that assigns to all workers at once:

```python
    @tool
    def assign_tasks(assignments: str) -> str:
        """Assign tasks to all workers at once.

        Args:
            assignments: JSON string mapping worker_id to task description.
                Example: {"worker_0": "Pick up onion and put in pot",
                          "worker_1": "Get a dish from the dispenser"}
        """
        import json
        try:
            task_map = json.loads(assignments)
        except json.JSONDecodeError:
            return "Error: Invalid JSON. Provide {\"worker_0\": \"task\", ...}"

        results = []
        for wid, desc in task_map.items():
            if wid not in worker_registry:
                results.append(f"Error: Unknown worker '{wid}'")
                continue
            task = Task(
                description=desc,
                worker_id=wid,
                created_at=planner_tool_state.state.timestep if planner_tool_state.state else 0,
            )
            worker_registry[wid].set_task(task)
            results.append(f"{wid}: {desc}")
        return "Tasks assigned:\n" + "\n".join(results)
```

**Verification:** `assign_tasks` creates Task objects in the correct workers' ToolStates.

---

### Task 5: Create shared graph builder

**File:** `src/overcooked_ai_py/agents/llm/graph_builder.py` (new)

Generic ReAct graph used by both planner and workers:

```python
def build_react_graph(
    model_name: str,
    system_prompt: str,
    observation_tools: list,
    action_tools: list,
    action_tool_names: set[str],
    get_chosen_fn: callable,
    debug: bool = False,
    debug_prefix: str = "[LLM]",
    api_base: str = None,
    api_key: str = None,
):
    """Build a ReAct LangGraph.

    Same structure as current graph.py:
    START → llm → route → {obs_tools → llm (loop), action_tools → END, end → END}

    Args:
        get_chosen_fn: Returns truthy value when the graph should terminate.
            For workers: lambda: tool_state.chosen_action
            For planner: lambda: <check if tasks were assigned>
    """
    ...
```

This replaces the need for separate planner_graph.py and worker_graph.py.

**Verification:** Compiles with both planner and worker tool sets.

---

### Task 6: Create planner and worker system prompts

**File:** `src/overcooked_ai_py/agents/llm/state_serializer.py` (modify existing)

```python
def build_planner_system_prompt(mdp, worker_ids, horizon=None) -> str:
    """System prompt for the planner.

    Includes:
    - Full game rules and layout
    - All player positions and key locations
    - List of available workers and their IDs
    - Instructions to assign complementary tasks
    - Emphasis: workers CANNOT talk to each other, so tasks must be self-contained
    - Strategy guidance for coordination
    """
    ...

def build_worker_system_prompt(mdp, agent_index, worker_id, horizon=None) -> str:
    """System prompt for a worker.

    Includes:
    - Game rules focused on movement and interaction
    - Layout info
    - Instructions to execute assigned task
    - NO mention of other workers (worker doesn't know they exist)
    - Navigation guidance
    """
    ...
```

**Verification:** Prompts generate valid text for cramped_room layout.

---

### Task 7: Create Planner class

**File:** `src/overcooked_ai_py/agents/llm/planner.py` (new)

The planner is a shared service, NOT an Agent:

```python
class Planner:
    """Central planner that assigns tasks to worker agents.

    NOT an Agent — it doesn't produce actions.
    Shared by all WorkerAgents. Runs once per replan interval.

    Args:
        model_name: LiteLLM model string
        replan_interval: Steps between replanning (default: 5)
        debug: Print planner reasoning
    """

    def __init__(self, model_name="gpt-4o", replan_interval=5,
                 debug=False, horizon=None, api_base=None, api_key=None):
        self.model_name = model_name
        self.replan_interval = replan_interval
        self.debug = debug
        self.horizon = horizon
        self.api_base = api_base
        self.api_key = api_key

        self._tool_state = ToolState()  # Planner's own ToolState
        self._graph = None
        self._system_prompt = None
        self._worker_registry: dict[str, ToolState] = {}
        self._last_plan_step = -1  # Timestep of last planning

    def register_worker(self, worker_id: str, worker_tool_state: ToolState):
        """Register a worker. Called during setup."""
        self._worker_registry[worker_id] = worker_tool_state

    def init(self, mdp, motion_planner):
        """Initialize planner after all workers are registered."""
        self._tool_state.init(mdp, motion_planner)

        worker_ids = list(self._worker_registry.keys())
        self._system_prompt = build_planner_system_prompt(
            mdp, worker_ids, self.horizon
        )
        obs, act, act_names = create_planner_tools(
            self._tool_state, self._worker_registry
        )
        self._graph = build_react_graph(
            self.model_name, self._system_prompt,
            obs, act, act_names,
            get_chosen_fn=lambda: self._tasks_assigned(),
            debug=self.debug, debug_prefix="[Planner]",
            api_base=self.api_base, api_key=self.api_key,
        )

    def _tasks_assigned(self):
        """Check if planner has assigned tasks (termination condition)."""
        for wid, ts in self._worker_registry.items():
            if ts.current_task and ts.current_task.steps_active == 0:
                return True
        return None

    def should_replan(self, state) -> bool:
        """Check if replanning is needed."""
        if self._last_plan_step == state.timestep:
            return False  # Already planned this step
        # First step or interval elapsed
        if self._last_plan_step < 0:
            return True
        steps_since = state.timestep - self._last_plan_step
        if steps_since >= self.replan_interval:
            return True
        # Any worker without a task
        for wid, ts in self._worker_registry.items():
            if ts.current_task is None or ts.current_task.completed:
                return True
        return False

    def maybe_replan(self, state):
        """Run planner if needed. Called by first worker each step."""
        if not self.should_replan(state):
            return

        state_text = serialize_state(
            self._tool_state.mdp, state, 0, self.horizon  # agent_index=0 for full view
        )
        self._tool_state.set_state(state, 0)

        # Include worker statuses
        statuses = []
        for wid, ts in self._worker_registry.items():
            statuses.append(f"  {wid} (Player {wid[-1]}): {ts.get_status()}")
        status_text = "\n".join(statuses)

        prompt = (
            f"Worker statuses:\n{status_text}\n\n"
            f"Current game state:\n{state_text}\n\n"
            f"Assign tasks to your workers."
        )

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt),
        ]
        self._graph.invoke({"messages": messages})
        self._last_plan_step = state.timestep

        if self.debug:
            for wid, ts in self._worker_registry.items():
                if ts.current_task:
                    print(f"  [Planner] → {wid}: {ts.current_task.description}")

    def get_task(self, worker_id: str) -> Task:
        """Get a worker's current task. Workers call this to read their own task only."""
        ts = self._worker_registry.get(worker_id)
        if ts is None:
            return None
        return ts.current_task

    def reset(self):
        self._tool_state.reset()
        self._graph = None
        self._last_plan_step = -1
        self._worker_registry.clear()
```

**Verification:** Planner initializes, registers workers, runs once per interval.

---

### Task 8: Create WorkerAgent class

**File:** `src/overcooked_ai_py/agents/llm/worker_agent.py` (new)

Standard `Agent` that queries the shared planner for its task:

```python
class WorkerAgent(Agent):
    """Worker agent that controls one chef on the floor.

    Receives tasks from a shared Planner. Produces one action per timestep.
    Cannot see other workers' tasks or communicate with them.

    Args:
        planner: Shared Planner instance
        worker_id: This worker's ID (e.g., "worker_0")
        model_name: LiteLLM model for this worker's LLM
        debug: Print reasoning
    """

    def __init__(self, planner, worker_id, model_name="gpt-4o",
                 debug=False, horizon=None, api_base=None, api_key=None):
        self.planner = planner
        self.worker_id = worker_id
        self.model_name = model_name
        self.debug = debug
        self.horizon = horizon
        self.api_base = api_base
        self.api_key = api_key

        self._tool_state = ToolState()
        self._graph = None
        self._system_prompt = None
        super().__init__()

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        from overcooked_ai_py.planning.planners import MotionPlanner
        mp = MotionPlanner(mdp)

        self._tool_state.init(mdp, mp)

        # Register with planner
        self.planner.register_worker(self.worker_id, self._tool_state)

        # Build worker graph
        self._system_prompt = build_worker_system_prompt(
            mdp, self.agent_index, self.worker_id, self.horizon
        )
        obs, act, act_names = create_worker_tools(self._tool_state)
        self._graph = build_react_graph(
            self.model_name, self._system_prompt,
            obs, act, act_names,
            get_chosen_fn=lambda: self._tool_state.chosen_action,
            debug=self.debug, debug_prefix=f"[{self.worker_id}]",
            api_base=self.api_base, api_key=self.api_key,
        )

    def action(self, state):
        # Step 1: Trigger planner if needed (first worker to call triggers it)
        self.planner.maybe_replan(state)

        # Step 2: Read this worker's task
        task = self.planner.get_task(self.worker_id)
        task_text = task.description if task else "No task assigned. Wait for instructions."

        # Step 3: Run worker LLM
        state_text = serialize_state(self.mdp, state, self.agent_index, self.horizon)
        self._tool_state.set_state(state, self.agent_index)

        prompt = (
            f"Your current task: {task_text}\n\n"
            f"Current game state:\n{state_text}\n\n"
            f"Choose one action to execute your task."
        )

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt),
        ]
        self._graph.invoke({"messages": messages})

        # Step 4: Get action
        chosen = self._tool_state.chosen_action
        if chosen is None:
            from overcooked_ai_py.mdp.actions import Action
            chosen = Action.STAY

        # Step 5: Track task progress
        if task:
            task.steps_active += 1

        if self.debug:
            from overcooked_ai_py.mdp.actions import Action
            action_name = Action.ACTION_TO_CHAR.get(chosen, str(chosen))
            print(f"  [Step {state.timestep}] {self.worker_id} (Player {self.agent_index}) → {action_name}")

        action_probs = self.a_probs_from_action(chosen)
        return chosen, {"action_probs": action_probs}

    def reset(self):
        super().reset()
        self._tool_state.reset()
        self._graph = None
```

**Key:** `action()` calls `self.planner.maybe_replan(state)` first. The planner checks `_last_plan_step == state.timestep` to avoid running twice when the second worker calls it on the same step.

**Verification:** Worker produces valid actions every step.

---

### Task 9: Update __init__.py and run script

**Files:**
- `src/overcooked_ai_py/agents/llm/__init__.py` (modify)
- `scripts/run_llm_agent.py` (modify)

Update exports:
```python
from overcooked_ai_py.agents.llm.llm_agent import LLMAgent
from overcooked_ai_py.agents.llm.worker_agent import WorkerAgent
from overcooked_ai_py.agents.llm.planner import Planner
```

Update run script:
```python
parser.add_argument("--agent-type", choices=["llm", "planner-worker"], default="llm")
parser.add_argument("--replan-interval", type=int, default=5)

if args.agent_type == "planner-worker":
    planner = Planner(
        model_name=model, replan_interval=args.replan_interval,
        debug=args.debug, horizon=args.horizon,
        api_base=api_base, api_key=api_key,
    )
    worker_0 = WorkerAgent(planner, "worker_0", model_name=model,
                           debug=args.debug, horizon=args.horizon,
                           api_base=api_base, api_key=api_key)
    worker_1 = WorkerAgent(planner, "worker_1", model_name=model,
                           debug=args.debug, horizon=args.horizon,
                           api_base=api_base, api_key=api_key)
    agent_pair = AgentPair(worker_0, worker_1)

    # Initialize planner after both workers registered
    env.reset()
    agent_pair.reset()
    agent_pair.set_mdp(mdp)
    planner.init(mdp, MotionPlanner(mdp))
else:
    llm_agent = LLMAgent(...)
    partner = make_greedy_partner(mdp)
    agent_pair = AgentPair(llm_agent, partner)
    env.reset()
    agent_pair.reset()
    agent_pair.set_mdp(mdp)
```

**Verification:** Script runs with `--agent-type planner-worker`.

---

### Task 10: Write tests

**File:** `testing/planner_worker_test.py` (new)

1. **Task dataclass** — creation, defaults, worker_id
2. **ToolState** — all methods, get_status
3. **Worker tools** — factory creates isolated tools per worker
4. **Planner tools** — assign_tasks writes to correct worker ToolStates
5. **Worker isolation** — Worker 0 tools cannot access Worker 1 ToolState
6. **Planner.should_replan** — interval logic, step caching, idle workers
7. **Planner.maybe_replan** — runs once per step even when called twice
8. **WorkerAgent** — instantiation, set_mdp registers with planner
9. **Integration** — mock LLMs, verify planner assigns tasks → workers produce actions

**Verification:** `python -m pytest testing/planner_worker_test.py`

---

## File Summary

| File | Status | Purpose |
|------|--------|---------|
| `agents/llm/task.py` | New | Task dataclass |
| `agents/llm/tool_state.py` | New | Instance-level tool state |
| `agents/llm/worker_tools.py` | New | Worker tools (factory) |
| `agents/llm/planner_tools.py` | New | Planner tools (assign_tasks, get_worker_status) |
| `agents/llm/graph_builder.py` | New | Shared ReAct graph builder |
| `agents/llm/planner.py` | New | Planner class (shared service) |
| `agents/llm/worker_agent.py` | New | WorkerAgent class (extends Agent) |
| `agents/llm/state_serializer.py` | Modify | Add planner/worker system prompts |
| `agents/llm/__init__.py` | Modify | Export new classes |
| `scripts/run_llm_agent.py` | Modify | Add --agent-type planner-worker |
| `testing/planner_worker_test.py` | New | Tests |

## Existing Files NOT Modified

`llm_agent.py`, `graph.py`, `tools.py`, `agent.py`, `overcooked_env.py` — all untouched. Zero regression risk.

## Execution Order

```
Task 1 (Task) ──────────────────────────────────────┐
Task 2 (ToolState) ──┬──────────────────────────────┤
                     ├── Task 3 (worker tools) ──┐  │
                     └── Task 4 (planner tools) ─┤  │
                                                  │  │
Task 5 (graph builder) ──────────────────────────┤  │
Task 6 (prompts) ────────────────────────────────┤  │
                                                  │  │
                     Task 7 (Planner) ────────────┤  │
                     Task 8 (WorkerAgent) ────────┘  │
                                                     │
                     Task 9 (script integration) ────┘
                     Task 10 (tests) ─── throughout
```
