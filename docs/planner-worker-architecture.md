
## Overview

The planner-worker architecture is a **hierarchical multi-agent system** where one centralized Planner LLM coordinates two independent Worker LLMs to cooperatively play Overcooked. This document provides a complete end-to-end explanation of how the system works.

**Key Insight**: This is a **3-LLM system** (1 Planner + 2 Workers) that operates with strict isolation between workers while maintaining centralized coordination.

---

## Part 1: System Components

### 1.1 Core Components Overview

```
┌─────────────────────────────────────────────────────────┐
│                    GAME ENVIRONMENT                      │
│                 (OvercookedGridworld)                    │
└─────────────────────────────────────────────────────────┘
                           │
                           │ State updates
                           ↓
┌─────────────────────────────────────────────────────────┐
│                      PLANNER (Shared)                    │
│  - Centralized decision maker                           │
│  - Has access to ALL worker states                      │
│  - Assigns tasks every N steps (replan_interval)        │
│  - Does NOT execute actions directly                    │
└─────────────────────────────────────────────────────────┘
                     │                  │
         Task assignment         Task assignment
                     ↓                  ↓
        ┌──────────────────┐  ┌──────────────────┐
        │   WORKER_0       │  │   WORKER_1       │
        │  (Agent class)   │  │  (Agent class)   │
        │  - Player 0      │  │  - Player 1      │
        │  - Isolated      │  │  - Isolated      │
        │  - Executes      │  │  - Executes      │
        │    actions       │  │    actions       │
        └──────────────────┘  └──────────────────┘
                     │                  │
              Action outputs      Action outputs
                     └────────┬─────────┘
                              ↓
                    Joint actions to environment
```

### 1.2 Detailed Component Breakdown

#### **Planner** (`src/overcooked_ai_py/agents/llm/planner.py`)

**Purpose**: Central coordinator that assigns complementary tasks to workers.

**Key Attributes**:
- `model_name`: LiteLLM model string (e.g., "gpt-4o")
- `replan_interval`: Number of steps between replanning (default: 5)
- `_tool_state`: Planner's own `ToolState` instance (for observation tools)
- `_worker_registry`: Dict mapping `worker_id` → worker's `ToolState`
- `_graph`: LangGraph StateGraph for ReAct loop
- `_system_prompt`: One-time prompt with game rules and coordination strategy
- `_last_plan_step`: Timestep of last planning (prevents duplicate planning)

**Key Methods**:
- `register_worker(worker_id, worker_tool_state)`: Called during setup to register workers
- `init(mdp, motion_planner)`: Initialize after all workers registered
- `should_replan(state)`: Returns `True` if replanning needed (interval elapsed or worker idle)
- `maybe_replan(state)`: Triggers planner LLM if `should_replan()` returns `True`
- `get_task(worker_id)`: Workers call this to read their assigned task
- `reset()`: Reset for new episode (preserves worker registry)

**Critical Constraints**:
- **NOT an Agent subclass** - doesn't produce actions directly
- Runs at most once per timestep (first worker to call `maybe_replan` triggers it)
- Has 20-step recursion limit (more than workers to allow complex reasoning)

---

#### **WorkerAgent** (`src/overcooked_ai_py/agents/llm/worker_agent.py`)

**Purpose**: Individual chef that executes assigned tasks and produces game actions.

**Key Attributes**:
- `planner`: Reference to shared `Planner` instance
- `worker_id`: Unique identifier (e.g., "worker_0", "worker_1")
- `model_name`: LiteLLM model string
- `_tool_state`: Worker's own isolated `ToolState` instance
- `_graph`: Worker's own LangGraph StateGraph for ReAct loop
- `_system_prompt`: Worker-specific prompt with task execution guidance
- `agent_index`: Player index (0 or 1) inherited from `Agent` class

**Key Methods**:
- `set_mdp(mdp)`: Initialize with MDP, create motion planner, register with planner
- `action(state)`: Main decision loop - returns `(action, info_dict)`
- `reset()`: Reset for new episode

**Workflow in `action()` method**:
```python
def action(self, state):
    # Step 1: Trigger planner if needed (first worker triggers)
    self.planner.maybe_replan(state)

    # Step 2: Read assigned task from planner
    task = self.planner.get_task(self.worker_id)
    task_text = task.description if task else "No task assigned."

    # Step 3: Serialize game state to text
    state_text = serialize_state(mdp, state, agent_index, horizon)
    self._tool_state.set_state(state, self.agent_index)

    # Step 4: Build prompt with task + state
    prompt = f"Your current task: {task_text}\n\nCurrent game state:\n{state_text}"

    # Step 5: Invoke worker's ReAct graph (15-step recursion limit)
    self._graph.invoke({"messages": [SystemMessage, HumanMessage]})

    # Step 6: Extract chosen action from tool_state
    chosen = self._tool_state.chosen_action or Action.STAY

    # Step 7: Increment task progress counter
    if task:
        task.steps_active += 1

    return chosen, {"action_probs": ...}
```

**Critical Constraints**:
- Can hold **ONLY ONE ITEM** at a time (Overcooked game rule)
- Must be **adjacent** (Manhattan distance = 1) to interact with objects
- Has 15-step recursion limit (prevents infinite observation loops)
- **Cannot see other workers' tasks** - complete isolation except through planner

---

#### **ToolState** (`src/overcooked_ai_py/agents/llm/tool_state.py`)

**Purpose**: Encapsulates the runtime context for one agent (planner or worker).

**Attributes**:
```python
class ToolState:
    mdp: OvercookedGridworld           # Game rules and layout
    state: OvercookedState             # Current game state (updated each step)
    agent_index: int                   # Which player (0 or 1) - None for planner
    motion_planner: MotionPlanner      # A* pathfinding for distance calculations
    chosen_action: Action              # Action selected by worker (None for planner)
    current_task: Optional[Task]       # Task assigned to this worker
```

**Key Methods**:
- `init(mdp, motion_planner)`: One-time setup with game context
- `set_state(state, agent_index)`: Update before each LLM invocation (resets `chosen_action`)
- `set_action(action)`: Called by action tools to commit to an action
- `set_task(task)`: Called by planner's `assign_tasks` tool to update worker task
- `get_status()`: Returns worker status dict for planner to query
- `reset()`: Clear state for new episode (preserves `mdp` and `motion_planner`)

**Critical Design**:
- **Each worker has its own `ToolState`** - workers cannot access each other's state
- Planner has its own `ToolState` + references to all worker `ToolState` instances via `_worker_registry`
- Tools are bound to specific `ToolState` instances via Python closures

---

#### **Task** (`src/overcooked_ai_py/agents/llm/task.py`)

**Purpose**: Data structure for communication between planner and workers.

```python
@dataclass
class Task:
    description: str           # Natural language task (e.g., "Pick up 3 onions")
    worker_id: str             # Which worker this is for (e.g., "worker_0")
    created_at: int            # Timestep when assigned
    completed: bool = False    # Whether worker marked task as done
    steps_active: int = 0      # How long worker has been working on this task
```

**Lifecycle**:
1. Planner creates `Task` via `assign_tasks` tool
2. `Task` stored in worker's `ToolState.current_task`
3. Worker reads task via `planner.get_task(worker_id)`
4. Worker increments `steps_active` each timestep
5. Worker can mark `completed = True` (optional, not currently used)
6. Planner can query task status via `ToolState.get_status()`

---

#### **ReAct Graph Builder** (`src/overcooked_ai_py/agents/llm/graph_builder.py`)

**Purpose**: Shared factory function that creates LangGraph StateGraph for both planner and workers.

**Graph Structure**:
```
START
  ↓
llm_node (calls LLM with tools)
  ↓
route_after_llm (decides next step based on tool calls)
  ├─→ obs_tools (observation tools like get_surroundings)
  │     ↓
  │   loop back to llm_node
  ├─→ action_tools (action tools like move_up or assign_tasks)
  │     ↓
  │   END (terminates graph)
  └─→ end (no tool calls)
        ↓
      END
```

**Key Function Signature**:
```python
def build_react_graph(
    model_name: str,              # LiteLLM model
    system_prompt: str,           # One-time game rules prompt
    observation_tools: list,      # Non-terminating tools
    action_tools: list,           # Terminating tools
    action_tool_names: set[str],  # Names of action tools (for routing)
    get_chosen_fn: Callable,      # Check if action chosen (termination)
    debug: bool = False,
    debug_prefix: str = "[LLM]",
    api_base: str = None,
    api_key: str = None,
    llm_timeout_seconds: float = 35.0,
) -> StateGraph
```

**Termination Logic**:
- **Workers**: Graph terminates when `tool_state.chosen_action` is not None (any action tool sets this)
- **Planner**: Graph terminates when any worker has `current_task.steps_active == 0` (fresh task assigned)

**Recursion Limits** (prevents infinite loops):
- Workers: 15 steps (allows ~5 observation calls before forcing action)
- Planner: 20 steps (allows more complex reasoning)

**Timeout**: Each LLM call has 35-second hard timeout to prevent hangs

---

### 1.3 State Serialization (`src/overcooked_ai_py/agents/llm/state_serializer.py`)

**Purpose**: Convert game state into text for LLM consumption.

**Key Functions**:

1. **`serialize_state(mdp, state, agent_index, horizon)`**
   - Converts `OvercookedState` to text description
   - Includes: timestep, grid with player positions, held objects, pot status, counter objects, orders
   - Example output:
     ```
     Timestep: 42 / 200

     GRID:
       XXPXX
       X   X
       O Y S

     YOU: pos=(2,2), facing=up, holding=onion
     PARTNER: pos=(4,2), facing=right, holding=nothing

     POTS:
       Pot at (2,0): FULL (3/3) but NOT COOKING (onion, onion, onion). INTERACT with empty hands to start cooking.

     COUNTER OBJECTS:
       dish: [(1,1), (3,1)]

     ORDERS:
       any-onion_soup
     ```

2. **`build_planner_system_prompt(mdp, worker_ids, horizon)`**
   - One-time prompt for planner with:
     - Role: "You are the PLANNER in a cooperative cooking game"
     - Game rules (soup pipeline, interact mechanics, one-item-at-a-time constraint)
     - Layout grid and key locations
     - Worker list
     - Coordination strategy tips
     - Task assignment guidelines (be specific, self-contained, include locations)

3. **`build_worker_system_prompt(mdp, agent_index, worker_id, horizon)`**
   - One-time prompt for worker with:
     - Role: "You are worker_0, a chef in Overcooked. You are Player 0."
     - Game rules (same as planner)
     - Layout grid and key locations
     - **ACTION GUIDE - ADJACENCY RULES** (critical section teaching Manhattan distance)
     - Concrete examples showing adjacency calculations
     - Navigation tips (use `check_path()`, avoid obstacles)
     - Workflow for each turn (read state → observe → plan → execute ONE action)

---

## Part 2: Tools Available to Agents

### 2.1 Planner Tools (`src/overcooked_ai_py/agents/llm/planner_tools.py`)

**Factory Function**: `create_planner_tools(planner_tool_state, worker_registry)`

#### **Observation Tools** (non-terminating):

1. **`get_surroundings()`**
   - Returns terrain type and objects adjacent to planner in 4 directions
   - Shows: terrain name, object details (soup status, ingredients), partner position
   - Example: `"up: pot [FULL 3/3, NOT COOKING - INTERACT to start]"`

2. **`get_pot_details()`**
   - Returns detailed status of ALL pots in kitchen
   - Shows: ingredients list, cooking timer, ready flag, actionable instructions
   - Example: `"Pot at (2,1): COOKING, 15 ticks remaining. Ingredients: onion, onion, tomato."`

3. **`check_path(target)`**
   - Calculates shortest path distance to nearest target
   - Targets: 'onion_dispenser', 'tomato_dispenser', 'dish_dispenser', 'pot', 'serving', 'dish' (counter), 'counter'
   - Uses motion planner's A* search
   - Example: `"Nearest pot is at (2,1), 8 steps away."`

4. **`get_worker_status(worker_id)`**
   - Returns current status of specified worker
   - Returns JSON: `{"status": "working", "task": "Pick up onion", "steps_active": 3}`
   - Status values: "idle", "working", "completed"

#### **Action Tools** (terminating):

1. **`assign_tasks(assignments: str)`**
   - Assigns tasks to workers and ENDS planner's turn
   - Input: JSON string `'{"worker_0": "task description", "worker_1": "..."}'`
   - Creates `Task` objects and stores them in worker `ToolState` instances
   - Returns confirmation: `"Assigned: worker_0: Pick up onion; worker_1: Get a dish"`
   - **This is the ONLY action tool for the planner**

---

### 2.2 Worker Tools (`src/overcooked_ai_py/agents/llm/worker_tools.py`)

**Factory Function**: `create_worker_tools(tool_state)`

#### **Observation Tools** (non-terminating):

1. **`get_surroundings()`**
   - Same as planner version, but scoped to worker's own position
   - Returns terrain and objects in 4 adjacent cells
   - Shows partner position as `[PARTNER HERE]`

2. **`get_pot_details()`**
   - Same as planner version
   - Returns status of all pots with cooking instructions

3. **`check_path(target)`**
   - Same as planner version
   - Calculates distance from worker's current position to target

#### **Action Tools** (terminating):

1. **`move_up()` / `move_down()` / `move_left()` / `move_right()`**
   - Moves worker in specified direction (NORTH/SOUTH/WEST/EAST)
   - Also updates facing direction
   - Sets `tool_state.chosen_action` to the direction
   - Returns: `"Moving up."`

2. **`wait()`**
   - Worker stays in place (Action.STAY)
   - Sets `tool_state.chosen_action` to Action.STAY
   - Returns: `"Waiting."`

3. **`interact()`**
   - Interacts with object/terrain worker is facing
   - Uses include: pick up item, place in pot, collect soup, deliver soup, start cooking
   - Sets `tool_state.chosen_action` to Action.INTERACT
   - Returns: `"Interacting."`

**Critical Difference from Planner**:
- Workers have 6 action tools (movement + interact + wait)
- Planner has 1 action tool (assign_tasks)
- Workers execute game actions; planner only coordinates

---

## Part 3: Workflow and State Changes

### 3.1 Episode Initialization

```python
# In run_llm_agent.py
mdp = OvercookedGridworld.from_layout_name("cramped_room")
env = OvercookedEnv.from_mdp(mdp, horizon=200)

# Create planner
planner = Planner(model_name="gpt-4o", replan_interval=5, debug=True)

# Create workers
worker_0 = WorkerAgent(planner, "worker_0", model_name="gpt-4o")
worker_1 = WorkerAgent(planner, "worker_1", model_name="gpt-4o")

# Create agent pair
agent_pair = AgentPair(worker_0, worker_1)

# Initialize
env.reset()
agent_pair.reset()
agent_pair.set_mdp(mdp)  # This triggers worker registration with planner
planner.init(mdp, MotionPlanner(mdp))  # Finalize planner setup
```

**What Happens in `agent_pair.set_mdp(mdp)`**:
1. Calls `worker_0.set_mdp(mdp)`:
   - Creates `MotionPlanner` for worker_0
   - Initializes `worker_0._tool_state` with mdp and motion planner
   - Calls `planner.register_worker("worker_0", worker_0._tool_state)`
   - Builds worker's system prompt
   - Creates worker's observation and action tools
   - Builds worker's ReAct graph

2. Calls `worker_1.set_mdp(mdp)`:
   - Same process as worker_0
   - Now planner has both workers in `_worker_registry`

3. Calls `planner.init(mdp, MotionPlanner(mdp))`:
   - Initializes planner's own tool state
   - Builds planner's system prompt (includes worker_ids list)
   - Creates planner's tools (passes `_worker_registry` to tool factory)
   - Builds planner's ReAct graph

**State After Initialization**:
- Planner has references to both workers' `ToolState` instances
- Each worker has its own isolated `ToolState`
- All agents have compiled LangGraph graphs ready to invoke
- `_last_plan_step = -1` (forces planning on first step)

---

### 3.2 Per-Timestep Execution Flow

**Main Game Loop** (in `run_llm_agent.py`):
```python
while not done:
    state = env.state

    # Get actions from both workers
    joint_action_and_infos = agent_pair.joint_action(state)

    # Execute actions in environment
    next_state, rewards, done, info = env.step(actions, infos)
```

**Detailed Step-by-Step Flow**:

#### **Step 1: First Worker Called (`worker_0.action(state)`)**

```
┌─────────────────────────────────────────────────────────┐
│ worker_0.action(state) - Timestep 0                     │
└─────────────────────────────────────────────────────────┘
  │
  ├─→ self.planner.maybe_replan(state)
  │    │
  │    ├─→ should_replan(state)?
  │    │     - First step: YES (_last_plan_step = -1)
  │    │     - Sets _last_plan_step = 0 (prevents duplicate)
  │    │
  │    ├─→ serialize_state(mdp, state, agent_index=0, horizon)
  │    │     Returns: "Timestep: 0 / 200\nGRID:\n..."
  │    │
  │    ├─→ planner._tool_state.set_state(state, agent_index=0)
  │    │
  │    ├─→ Build worker status text:
  │    │     "worker_0 (Player 0): idle"
  │    │     "worker_1 (Player 1): idle"
  │    │
  │    ├─→ Build planner prompt:
  │    │     "Worker statuses:\n...\n\nCurrent game state:\n...\n\nAssign tasks."
  │    │
  │    ├─→ Invoke planner._graph with messages
  │    │     [SystemMessage(planner system prompt),
  │    │      HumanMessage(worker statuses + game state)]
  │    │
  │    │   ┌───────────────────────────────────────────┐
  │    │   │ PLANNER REACT LOOP (max 20 steps)        │
  │    │   └───────────────────────────────────────────┘
  │    │     │
  │    │     ├─→ llm_node: Call LLM with tools
  │    │     │     LLM reasoning: "Worker 0 should gather onions, Worker 1 should get a dish"
  │    │     │
  │    │     ├─→ route_after_llm: Check tool calls
  │    │     │     - LLM calls get_pot_details() → obs_tools
  │    │     │
  │    │     ├─→ obs_tools: Execute get_pot_details()
  │    │     │     Returns: "Pot at (2,1): empty (0/3 ingredients)"
  │    │     │     → Back to llm_node
  │    │     │
  │    │     ├─→ llm_node: Call LLM again with observation results
  │    │     │     LLM reasoning: "Pot is empty, workers should gather ingredients"
  │    │     │
  │    │     ├─→ route_after_llm: Check tool calls
  │    │     │     - LLM calls assign_tasks({"worker_0": "Pick up 3 onions...", "worker_1": "Get a dish..."})
  │    │     │     → action_tools
  │    │     │
  │    │     └─→ action_tools: Execute assign_tasks()
  │    │           - Creates Task(description="Pick up 3 onions...", worker_id="worker_0", created_at=0)
  │    │           - Calls worker_registry["worker_0"].set_task(task)
  │    │           - Creates Task for worker_1
  │    │           - Calls worker_registry["worker_1"].set_task(task)
  │    │           - Returns: "Assigned: worker_0: Pick up 3 onions; worker_1: Get a dish"
  │    │           → END (graph terminates)
  │    │
  │    └─→ Planner graph completed
  │
  ├─→ task = planner.get_task("worker_0")
  │     Returns: Task(description="Pick up 3 onions...", worker_id="worker_0", created_at=0, steps_active=0)
  │
  ├─→ task_text = "Pick up 3 onions from the onion dispenser and place them in the pot at (2,1)"
  │
  ├─→ state_text = serialize_state(mdp, state, agent_index=0, horizon)
  │
  ├─→ worker_0._tool_state.set_state(state, agent_index=0)
  │     - Sets tool_state.state = state
  │     - Sets tool_state.agent_index = 0
  │     - Resets tool_state.chosen_action = None
  │
  ├─→ prompt = "Your current task: {task_text}\n\nCurrent game state:\n{state_text}\n\nChoose one action."
  │
  ├─→ Invoke worker_0._graph
  │
  │   ┌───────────────────────────────────────────┐
  │   │ WORKER_0 REACT LOOP (max 15 steps)       │
  │   └───────────────────────────────────────────┘
  │     │
  │     ├─→ llm_node: Call LLM with tools
  │     │     LLM reasoning: "I need to move to the onion dispenser. Let me check my surroundings first."
  │     │
  │     ├─→ route_after_llm: Check tool calls
  │     │     - LLM calls get_surroundings() → obs_tools
  │     │
  │     ├─→ obs_tools: Execute get_surroundings()
  │     │     Returns: "up: wall\ndown: floor\nleft: onion_dispenser\nright: counter"
  │     │     → Back to llm_node
  │     │
  │     ├─→ llm_node: Call LLM again
  │     │     LLM reasoning: "Onion dispenser is to my left. I'll move left."
  │     │
  │     ├─→ route_after_llm: Check tool calls
  │     │     - LLM calls move_left() → action_tools
  │     │
  │     └─→ action_tools: Execute move_left()
  │           - Calls tool_state.set_action(Direction.WEST)
  │           - Sets tool_state.chosen_action = Direction.WEST
  │           → END (graph terminates)
  │
  ├─→ chosen = tool_state.chosen_action  # Direction.WEST
  │
  ├─→ task.steps_active += 1  # Now 1
  │
  └─→ return (Direction.WEST, {"action_probs": ...})
```

#### **Step 2: Second Worker Called (`worker_1.action(state)`)**

```
┌─────────────────────────────────────────────────────────┐
│ worker_1.action(state) - Timestep 0                     │
└─────────────────────────────────────────────────────────┘
  │
  ├─→ self.planner.maybe_replan(state)
  │    │
  │    ├─→ should_replan(state)?
  │    │     - _last_plan_step = 0 (same as current timestep)
  │    │     - Returns FALSE (already planned this timestep)
  │    │
  │    └─→ Skip replanning
  │
  ├─→ task = planner.get_task("worker_1")
  │     Returns: Task(description="Get a dish from the dish dispenser", worker_id="worker_1", created_at=0, steps_active=0)
  │
  ├─→ task_text = "Get a dish from the dish dispenser"
  │
  ├─→ state_text = serialize_state(mdp, state, agent_index=1, horizon)
  │
  ├─→ worker_1._tool_state.set_state(state, agent_index=1)
  │
  ├─→ Invoke worker_1._graph
  │
  │   ┌───────────────────────────────────────────┐
  │   │ WORKER_1 REACT LOOP                       │
  │   └───────────────────────────────────────────┘
  │     (Similar to worker_0, but for agent_index=1)
  │
  └─→ return (Direction.NORTH, {"action_probs": ...})
```

#### **Step 3: Actions Executed in Environment**

```
joint_actions = (Direction.WEST, Direction.NORTH)  # From worker_0 and worker_1

next_state, rewards, done, info = env.step(joint_actions, infos)
  - Player 0 moves west
  - Player 1 moves north
  - State updated
  - Timestep incremented to 1
```

---

### 3.3 Replanning Trigger Conditions

**When `should_replan(state)` returns `True`**:

1. **First step of episode**: `_last_plan_step = -1`
2. **Interval elapsed**: `state.timestep - _last_plan_step >= replan_interval`
   - Example: If `replan_interval = 5`, replans at timesteps 0, 5, 10, 15, ...
3. **Worker without task**: Any worker has `current_task = None`
4. **Worker completed task**: Any worker has `current_task.completed = True`

**Prevents duplicate planning**:
- `_last_plan_step` updated immediately after planning
- Second worker at same timestep sees `_last_plan_step == state.timestep` and skips

---

### 3.4 Context Available to Each Agent

#### **Planner's Context** (when invoking graph):

```python
# System prompt (one-time, built at init):
- Game rules (soup pipeline, interact mechanics, one-item constraint)
- Layout grid with legend
- Key locations (pots, dispensers, serving)
- Worker list (["worker_0", "worker_1"])
- Coordination strategy
- Task assignment guidelines

# Per-invocation prompt:
- Worker statuses:
  - worker_0 (Player 0): working - "Pick up 3 onions..." (active for 8 steps)
  - worker_1 (Player 1): idle
- Current game state:
  - Timestep: 10 / 200
  - Grid with player positions (Y = worker_0, @ = worker_1)
  - Player positions, facing, held objects
  - Pot status (ingredients, cooking, ready)
  - Counter objects
  - Orders

# Tools available:
- Observation: get_surroundings, get_pot_details, check_path, get_worker_status
- Action: assign_tasks

# Data accessible via tools:
- Full game state (via planner_tool_state.state)
- All worker statuses (via worker_registry)
- Motion planning (via planner_tool_state.motion_planner)
```

#### **Worker's Context** (when invoking graph):

```python
# System prompt (one-time, built at init):
- Role: "You are worker_0, a chef in Overcooked. You are Player 0."
- Game rules (same as planner)
- Layout grid with legend
- Key locations
- ACTION GUIDE with adjacency rules and concrete examples
- Navigation tips
- Workflow instructions

# Per-invocation prompt:
- Your current task: "Pick up 3 onions from the onion dispenser and place them in the pot at (2,1)"
- Current game state:
  - Timestep: 10 / 200
  - Grid (Y = you, @ = partner)
  - YOU: pos=(1,2), facing=left, holding=nothing
  - PARTNER: pos=(4,3), facing=up, holding=dish
  - Pot status
  - Counter objects
  - Orders

# Tools available:
- Observation: get_surroundings, get_pot_details, check_path
- Action: move_up, move_down, move_left, move_right, wait, interact

# Data accessible via tools:
- Own position and state (via tool_state.state.players[agent_index])
- Partner position (via tool_state.state.players[1 - agent_index])
- Full game state (pots, counters, etc.)
- Motion planning (via tool_state.motion_planner)

# NOT accessible:
- Other workers' tasks (isolated via ToolState)
- Other workers' tool states
- Planner's reasoning
```

---

### 3.5 Communication Flow

```
┌─────────────────────────────────────────────────────────┐
│                    GAME ENVIRONMENT                      │
│                         │                                │
│                         ↓ State update                   │
└─────────────────────────────────────────────────────────┘
                          │
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ↓                                   ↓
 ┌────────────┐                      ┌────────────┐
 │  WORKER_0  │                      │  WORKER_1  │
 └────────────┘                      └────────────┘
        │                                   │
        │ maybe_replan() [FIRST]            │ maybe_replan() [SECOND]
        ↓                                   ↓
 ┌─────────────────────────────────────────────────────────┐
 │                    PLANNER                               │
 │  should_replan()?                                        │
 │    - First call: YES → triggers planning                │
 │    - Second call: NO → skip (already planned)           │
 └─────────────────────────────────────────────────────────┘
        ↓ (if YES)
 ┌─────────────────────────────────────────────────────────┐
 │  PLANNER REACT LOOP                                      │
 │  - Observes full game state                             │
 │  - Queries worker statuses via get_worker_status()      │
 │  - Reasons about task allocation                        │
 │  - Calls assign_tasks({"worker_0": "...", "worker_1": "..."}) │
 └─────────────────────────────────────────────────────────┘
        │
        ↓ assign_tasks() writes to worker_registry
 ┌──────────────────────────────────────────────────────────┐
 │  worker_registry["worker_0"].set_task(Task(...))         │
 │  worker_registry["worker_1"].set_task(Task(...))         │
 └──────────────────────────────────────────────────────────┘
        │                                   │
        ↓ get_task("worker_0")              ↓ get_task("worker_1")
 ┌────────────┐                      ┌────────────┐
 │  WORKER_0  │                      │  WORKER_1  │
 │  Reads task│                      │  Reads task│
 └────────────┘                      └────────────┘
        │                                   │
        ↓ REACT LOOP                        ↓ REACT LOOP
 ┌────────────┐                      ┌────────────┐
 │  Executes  │                      │  Executes  │
 │  task via  │                      │  task via  │
 │  tools     │                      │  tools     │
 └────────────┘                      └────────────┘
        │                                   │
        └─────────────┬──────────────┬──────┘
                      ↓              ↓
               (action_0, info_0)  (action_1, info_1)
                      │              │
                      └──────┬───────┘
                             ↓
                    Joint actions to env
```

**Key Communication Paths**:

1. **Worker → Planner** (read-only):
   - Workers trigger `maybe_replan()` each timestep
   - Planner reads worker status via `worker_registry[worker_id].get_status()`

2. **Planner → Worker** (write via task assignment):
   - Planner writes tasks to `worker_registry[worker_id].set_task(task)`
   - Workers read tasks via `planner.get_task(worker_id)`

3. **Worker ↔ Worker** (NO DIRECT COMMUNICATION):
   - Workers CANNOT see each other's tasks
   - Workers CANNOT access each other's ToolState
   - Workers only know partner position via game state serialization
   - Coordination happens ONLY through planner task assignment

---

### 3.6 State Transitions

**Worker ToolState Lifecycle**:

```
[Episode Start]
  ↓
set_mdp(mdp)
  - Creates ToolState
  - Initializes mdp, motion_planner
  - current_task = None
  - chosen_action = None
  ↓
[Each Timestep]
  ↓
set_state(state, agent_index)
  - Updates state
  - Updates agent_index
  - Resets chosen_action = None
  ↓
Planner assigns task (maybe)
  - Planner calls worker_registry[wid].set_task(new_task)
  - current_task = Task(description="...", steps_active=0)
  ↓
Worker invokes graph
  - Observes via tools (uses state, agent_index, mdp, motion_planner)
  - Calls action tool (sets chosen_action)
  ↓
Worker increments task progress
  - current_task.steps_active += 1
  ↓
[Next Timestep]
  - Loop back to set_state()
```

**Planner ToolState Lifecycle**:

```
[Episode Start]
  ↓
init(mdp, motion_planner)
  - Creates ToolState
  - Initializes mdp, motion_planner
  - Receives worker_registry from workers
  ↓
[Replanning Triggered]
  ↓
set_state(state, agent_index=0)
  - Updates state
  - Sets agent_index = 0 (planner views as player 0 for full visibility)
  ↓
Invoke planner graph
  - Observes via tools (uses state, mdp, motion_planner, worker_registry)
  - Calls assign_tasks() to write to worker_registry
  ↓
_last_plan_step = state.timestep
  - Prevents replanning until interval elapsed
```

---

## Part 4: Critical Design Constraints

### 4.1 Isolation Guarantees

**Workers Cannot**:
- Access other workers' `ToolState` instances
- See other workers' assigned tasks
- Communicate directly with each other
- Share memory or state

**How Isolation is Enforced**:
1. Each worker's tools are created via `create_worker_tools(tool_state)`, binding them to that worker's `ToolState` via Python closures
2. Workers only have reference to shared `Planner`, not to each other
3. Planner API (`get_task(worker_id)`) only returns the calling worker's task

### 4.2 Game Rule Constraints

**One Item at a Time**:
- Workers can hold max 1 item (enforced by game engine)
- Must place item before picking up another
- System prompts explicitly teach this rule

**Adjacency Requirement**:
- Workers must be adjacent (Manhattan distance = 1) to interact
- System prompts include concrete examples:
  - `You at (1,1), target at (0,1): |1-0| + |1-1| = 1 → ADJACENT!`
- Workers often forget this and try to move onto targets instead of interacting from adjacent cell

**Cooking Mechanics** (depends on layout):
- **New dynamics** (default): Pot with 3/3 ingredients does NOT cook automatically - worker must interact with empty hands to start
- **Old dynamics**: Pot starts cooking automatically when 3rd ingredient added

### 4.3 Recursion Limits

**Purpose**: Prevent infinite observation loops where LLM keeps calling observation tools without choosing an action.

**Limits**:
- **Workers**: 15 steps (allows ~5 observation calls before forcing termination)
- **Planner**: 20 steps (allows more complex reasoning)

**Behavior on Limit Reached**:
- LangGraph raises `RecursionError`
- Workers catch error and default to `Action.STAY`
- Planner catches error and tasks remain unchanged

### 4.4 Timeout Protection

**Hard Timeout**: 35 seconds per LLM turn (includes tool calls and LLM responses)

**Implementation**:
- Uses Unix `signal.SIGALRM` for hard wall-clock timeout
- Only works on main thread (disabled in workers/subthreads)
- Raises `TimeoutError` if exceeded

**Fallback**:
- LiteLLM has 30-second timeout on API calls
- If timeout occurs, graph catches exception and returns error AIMessage

---

## Part 5: Example Execution Trace

**Setup**: Cramped room layout, 2 workers, replan interval = 5

```
┌──────────────────────────────────────────────────────────────────┐
│ TIMESTEP 0                                                       │
└──────────────────────────────────────────────────────────────────┘

[worker_0.action(state)]
  → maybe_replan(): YES (first step)

  [PLANNER REACT]
    System: "You are the PLANNER..."
    User: "Worker statuses: worker_0 (Player 0): idle; worker_1 (Player 1): idle
           Current game state: Timestep 0/200, Grid: ..., Pots: empty
           Assign tasks to your workers."

    LLM: "I need to check the pot status first."
    → Calls get_pot_details()
    Tool: "Pot at (2,1): empty (0/3 ingredients)"

    LLM: "The pot is empty. Worker 0 should gather 3 onions. Worker 1 should get a dish and wait."
    → Calls assign_tasks('{"worker_0": "Pick up 3 onions from dispenser at (0,2) and place in pot at (2,1)", "worker_1": "Get a dish from dispenser at (4,0) and wait"}')
    Tool: "Assigned: worker_0: Pick up 3 onions...; worker_1: Get a dish..."

  [END PLANNER REACT]

  → task = "Pick up 3 onions from dispenser at (0,2) and place in pot at (2,1)"
  → state_text = "Timestep: 0/200\nGRID:\n...\nYOU: pos=(1,1), facing=up, holding=nothing..."

  [WORKER_0 REACT]
    System: "You are worker_0, a chef in Overcooked. You are Player 0..."
    User: "Your current task: Pick up 3 onions from dispenser at (0,2) and place in pot at (2,1)
           Current game state: ...
           Choose one action to execute your task."

    LLM: "I'm at (1,1) and need to get to onion dispenser at (0,2). Let me check the path."
    → Calls check_path('onion_dispenser')
    Tool: "Nearest onion_dispenser is at (0,2), 2 steps away."

    LLM: "I'm 2 steps away. I'll move left toward the dispenser."
    → Calls move_left()
    Tool: "Moving left."

  [END WORKER_0 REACT]

  → task.steps_active = 1
  → return (Direction.WEST, info)

[worker_1.action(state)]
  → maybe_replan(): NO (_last_plan_step = 0)

  → task = "Get a dish from dispenser at (4,0) and wait"

  [WORKER_1 REACT]
    LLM: "I need to move to dish dispenser at (4,0)."
    → Calls check_path('dish_dispenser')
    → Calls move_right()
  [END WORKER_1 REACT]

  → return (Direction.EAST, info)

[env.step((WEST, EAST), infos)]
  → Player 0 moves from (1,1) to (0,1)
  → Player 1 moves from (3,1) to (4,1)
  → Timestep → 1

┌──────────────────────────────────────────────────────────────────┐
│ TIMESTEP 1-4 (No replanning)                                     │
└──────────────────────────────────────────────────────────────────┘

[Each timestep]
  → maybe_replan(): NO (interval not elapsed)
  → Workers execute existing tasks
  → Worker_0 continues gathering onions
  → Worker_1 continues getting dish

┌──────────────────────────────────────────────────────────────────┐
│ TIMESTEP 5 (Replanning interval elapsed)                         │
└──────────────────────────────────────────────────────────────────┘

[worker_0.action(state)]
  → maybe_replan(): YES (5 - 0 >= 5)

  [PLANNER REACT]
    User: "Worker statuses:
           worker_0 (Player 0): working - Pick up 3 onions... (active for 5 steps)
           worker_1 (Player 1): working - Get a dish... (active for 5 steps)
           Current game state:
           YOU: pos=(0,2), holding=onion
           PARTNER: pos=(4,0), holding=dish
           Pots: Pot at (2,1): 1/3 ingredients (onion)
           ..."

    LLM: "Worker 0 has placed 1 onion and is holding another. They should continue. Worker 1 has a dish and should wait near the pot."
    → Calls assign_tasks('{"worker_0": "Continue adding 2 more onions to pot at (2,1)", "worker_1": "Wait near pot at (2,1) with your dish"}')

  [END PLANNER REACT]

  → New tasks assigned

  [Continue workers executing new tasks...]
```

---

## Summary

The planner-worker architecture is a **3-LLM hierarchical system** where:

1. **One Planner LLM** coordinates strategy by assigning complementary tasks every N steps
2. **Two Worker LLMs** execute assigned tasks independently, choosing actions each timestep
3. **Strict isolation** between workers prevents direct communication
4. **Task-based communication** flows from Planner → Workers via `Task` objects
5. **ReAct pattern** allows both planner and workers to observe before acting
6. **Recursion limits** and **timeouts** prevent infinite loops and hangs

**Key Innovation**: Centralized planning with decentralized execution, enabling coordination without direct worker-to-worker communication.
