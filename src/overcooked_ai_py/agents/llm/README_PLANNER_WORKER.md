# Planner-Worker Architecture

Hierarchical multi-agent system where one Planner LLM coordinates multiple Worker LLM agents in cooperative cooking tasks.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                     PLANNER                         │
│  (Shared service, not an Agent)                     │
│  - Analyzes full game state                         │
│  - Assigns tasks to workers every N steps           │
│  - Uses: assign_tasks tool                          │
│  - Does NOT execute actions                         │
└─────────────┬───────────────────────┬───────────────┘
              │                       │
              ▼                       ▼
     ┌────────────────┐      ┌────────────────┐
     │   WORKER_0     │      │   WORKER_1     │
     │  (Agent)       │      │  (Agent)       │
     │  - Player 0    │      │  - Player 1    │
     │  - Executes    │      │  - Executes    │
     │    assigned    │      │    assigned    │
     │    task        │      │    task        │
     │  - Produces    │      │  - Produces    │
     │    actions     │      │    actions     │
     └────────────────┘      └────────────────┘
```

## Components

### Core Files

- **`planner.py`** - Planner class (shared service)
- **`worker_agent.py`** - WorkerAgent class (extends Agent)
- **`task.py`** - Task dataclass for planner→worker communication
- **`tool_state.py`** - Per-agent state management (replaces module globals)
- **`worker_tools.py`** - Worker tool factory (observation + action tools)
- **`planner_tools.py`** - Planner tool factory (assign_tasks, get_worker_status)
- **`graph_builder.py`** - Shared ReAct graph builder for both planner and workers
- **`state_serializer.py`** - State serialization + role-specific system prompts

### Legacy Files (Single Agent Mode)

- **`llm_agent.py`** - Original LLMAgent (single agent + greedy partner)
- **`graph.py`** - DEPRECATED: Use graph_builder.py instead
- **`tools.py`** - DEPRECATED: Use worker_tools.py instead

## Key Concepts

### 1. Planner (Coordination)

**Role**: Strategic task assignment and coordination

**Capabilities**:
- Observes full game state
- Knows all worker positions and statuses
- Assigns complementary tasks to workers
- Replans every N steps (configurable)

**Limitations**:
- Does NOT execute actions directly
- Cannot control workers' low-level decisions
- Workers cannot ask questions (tasks must be self-contained)

**Tools**:
- `assign_tasks(assignments: str)` - Assigns tasks via JSON: `{"worker_0": "task", "worker_1": "task"}`
- `get_worker_status()` - Check worker positions, held items, task progress
- Shared observation tools: `get_surroundings()`, `get_pot_details()`, `check_path()`

### 2. WorkerAgent (Execution)

**Role**: Execute assigned tasks by choosing actions each timestep

**Capabilities**:
- Reads assigned task from Planner
- Chooses actions to accomplish task
- Uses observation tools to understand environment
- Executes one action per timestep

**Limitations**:
- Cannot see other workers' tasks
- Cannot communicate with other workers
- Can hold ONLY ONE ITEM at a time
- Must be adjacent (Manhattan distance = 1) to interact

**Tools**:
- Observation: `get_surroundings()`, `get_pot_details()`, `check_path(target)`
- Actions: `move_up()`, `move_down()`, `move_left()`, `move_right()`, `interact()`, `wait()`

### 3. Task Communication

```python
@dataclass
class Task:
    description: str        # Natural language task ("Pick up onion and place in pot at (2,0)")
    worker_id: str          # "worker_0" or "worker_1"
    created_at: int         # Timestep when assigned
    completed: bool         # Worker signals completion
    steps_active: int       # How long worker has been working on this task
```

**Task Assignment Flow**:
1. Planner calls `assign_tasks({"worker_0": "task description"})`
2. Task object created and stored in worker's ToolState
3. Worker reads task via `planner.get_task(worker_id)`
4. Worker executes task by choosing actions each step
5. Task.steps_active increments each timestep

### 4. State Isolation

Each agent has its own **ToolState** instance:

```python
class ToolState:
    mdp: OvercookedGridworld
    state: OvercookedState
    agent_index: int                    # 0 or 1 (which player)
    motion_planner: MotionPlanner
    chosen_action: Action               # Action selected this turn
    current_task: Optional[Task]        # Assigned task (workers only)
```

**Isolation enforcement**:
- Tools are created via factory functions that capture `ToolState` via closure
- Worker_0 tools can ONLY access worker_0's ToolState
- Worker_1 tools can ONLY access worker_1's ToolState
- Planner tools have access to all worker ToolStates via `worker_registry`

## Usage Example

```python
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.llm import Planner, WorkerAgent
from overcooked_ai_py.planning.planners import MotionPlanner

# Setup environment
mdp = OvercookedGridworld.from_layout_name("cramped_room")
env = OvercookedEnv.from_mdp(mdp, horizon=200)

# Create planner (shared by both workers)
planner = Planner(
    model_name="gpt-4o",
    replan_interval=5,      # Replan every 5 steps
    debug=True,
    horizon=200
)

# Create workers
worker_0 = WorkerAgent(planner, "worker_0", model_name="gpt-4o", debug=True)
worker_1 = WorkerAgent(planner, "worker_1", model_name="gpt-4o", debug=True)
agent_pair = AgentPair(worker_0, worker_1)

# Initialize (IMPORTANT: order matters!)
env.reset()
agent_pair.reset()
agent_pair.set_mdp(mdp)             # Workers register with planner here
planner.init(mdp, MotionPlanner(mdp))  # Planner builds graph after workers registered

# Run episode
done = False
while not done:
    state = env.state
    joint_action_and_infos = agent_pair.joint_action(state)
    actions = tuple(a for a, _ in joint_action_and_infos)
    infos = tuple(info for _, info in joint_action_and_infos)
    next_state, rewards, done, info = env.step(actions, infos)
```

See `scripts/run_llm_agent.py` for complete implementation with visualization.

## Observability and Benchmark Metadata

Both `llm` and `planner-worker` modes emit per-run JSONL logs with consistent
event schema. Local JSONL remains the source of truth.

- Log directory: `logs/agent_runs/`
- Event types: `run.start`, `run.end`, `llm.generation`, `tool.call`, `planner.assignment`, `action.commit`, `error`
- Required tags are always enforced:
  - `mode:llm` or `mode:planner-worker`
  - `layout:<layout_name>`

### LangFuse Trace Structure

When LangFuse is enabled, traces are emitted via the custom hierarchy reporter
(not graph callbacks). The expected structure is:

`run -> step_<n> -> role(planner/worker_0/worker_1/llm_agent) -> llm.generation/tool spans`

This avoids router/node-name clutter (`dispatch`, `observe`, `act`) and keeps
traces aligned to environment steps.

Trace metadata behavior:
- Trace name: `--run-name` (or autogenerated default)
- Trace tags (Tags column): user `--tags` + required mode/layout tags
- Trace metadata includes: mode, layout, model, experiment, variant, and run_id
- Run-end score: `episode_reward` is written as a LangFuse trace score

### CLI Flags

`scripts/run_llm_agent.py` includes run metadata flags for experiment tracking:

- `--run-name` (default autogenerated)
- `--run-title` (default empty)
- `--tags` (comma-separated user tags)
- `--experiment` (default `default-exp`)
- `--variant` (default `baseline`)
- `--notes` (default empty)

### Smoke Command Examples

```bash
uv run python scripts/run_llm_agent.py \
  --agent-type llm \
  --layout cramped_room \
  --horizon 20 \
  --run-name smoke-llm \
  --tags bench,smoke

uv run python scripts/run_llm_agent.py \
  --agent-type planner-worker \
  --layout cramped_room \
  --horizon 20 \
  --run-name smoke-pw \
  --tags bench,smoke
```

### Optional LangFuse Environment

If LangFuse keys are provided, run/step/role spans and manual LLM/tool
observations are emitted. If unavailable or misconfigured, execution continues
with local logs only.

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## System Prompts

### Planner Prompt Emphasizes:
- **Coordination**: Assign complementary tasks to workers
- **Worker isolation**: Workers CANNOT communicate with each other
- **Task clarity**: Tasks must be specific and self-contained
- **Strategic thinking**: Divide labor, avoid conflicts, balance workload

### Worker Prompt Emphasizes:
- **Task execution**: Focus on completing assigned task
- **Adjacency rules**: Manhattan distance = 1 means adjacent → INTERACT
- **One item limit**: Can hold ONLY ONE ITEM at a time
- **Navigation**: Use `check_path()` to plan routes around obstacles
- **NO mention of other workers**: Worker doesn't know about multi-agent system

## Critical Constraints

### 1. One Item At A Time
Players can hold **ONLY ONE ITEM**. To pick up something new, must place current item on counter or in pot.

**Bad**: "Pick up 3 onions and place them in pot"
**Good**: "Pick up onion, place in pot, repeat 3 times"

### 2. Adjacency = Manhattan Distance of 1
Worker must be **exactly 1 square away** (orthogonally) to interact.

**Formula**: `distance = |worker_x - target_x| + |worker_y - target_y|`
**Adjacent if**: `distance == 1`

**Examples**:
- Worker at (1,1), target at (0,1): distance = 1 → ADJACENT ✓
- Worker at (3,1), target at (4,1): distance = 1 → ADJACENT ✓
- Worker at (1,1), target at (3,3): distance = 4 → NOT adjacent ✗

**Action**: When adjacent AND facing target → INTERACT (don't move!)

### 3. Recursion Limits
Prevents infinite observation loops where LLM keeps calling tools without choosing an action.

- **Workers**: 15 steps max (~5 observation calls)
- **Planner**: 20 steps max (more reasoning allowed)

If limit reached → Worker defaults to `Action.STAY`

## Design Decisions

### Why Planner is NOT an Agent
- Planner doesn't control a player or produce actions
- Planner is a **shared service** accessed by both workers
- Avoids confusion about "which player is the planner?"

### Why Workers Can't Communicate
- Simulates realistic distributed task execution
- Forces planner to create self-contained task descriptions
- Makes task assignment more challenging and interesting

### Why Replanning is Periodic
- Constant replanning is expensive (LLM calls)
- Workers need stability to execute multi-step tasks
- Default: replan every 5 steps (configurable)

### Why Factory Pattern for Tools
- Each worker needs isolated tools bound to their ToolState
- Closures capture the specific ToolState instance
- Prevents workers from accessing each other's state

## Testing

**Unit tests** (103 total):
```bash
# All planner-worker tests
uv run python -m unittest \
  testing.test_task \
  testing.test_tool_state \
  testing.test_worker_tools \
  testing.test_planner_tools \
  testing.test_graph_builder \
  testing.test_state_serializer_prompts \
  testing.test_planner \
  testing.test_worker_agent_unit

# Just planner tests
uv run python -m unittest testing.test_planner -v

# Just worker tests
uv run python -m unittest testing.test_worker_agent_unit -v
```

**Integration test**:
```bash
# Run a short episode with visualization
source .env && uv run python scripts/run_llm_agent.py \
  --agent-type planner-worker \
  --horizon 50 \
  --debug \
  --visualize
```

## Debugging Tips

### Workers Stuck in Loops
**Symptom**: Worker repeats same reasoning without progress
**Cause**: Worker doesn't realize they're blocked or already adjacent
**Fix**: Check debug output for position changes; workers should use `check_path()` before moving

### Program Freezes
**Symptom**: Program hangs with no output
**Cause**: LLM stuck in infinite observation loop (calling tools but never choosing action)
**Fix**: Recursion limits (15/20 steps) should prevent this; if still happens, reduce limits

### Workers Not Interacting
**Symptom**: Workers try to move onto target instead of interacting
**Cause**: LLM doesn't understand adjacency rules
**Fix**: System prompt includes explicit Manhattan distance formula and concrete examples

### Planner Assigns Invalid Tasks
**Symptom**: Tasks ignore one-item-at-a-time constraint or require impossible coordination
**Cause**: Planner prompt not emphasizing constraints
**Fix**: System prompt includes "CRITICAL" markers for key constraints

## Future Enhancements

- **Memory system**: Workers remember previous task outcomes
- **Dynamic worker allocation**: Scale to N workers (currently fixed at 2)
- **Task completion feedback**: Workers explicitly signal when task is done
- **Hierarchical planning**: Planner creates multi-step task sequences
- **Adaptive replanning**: Replan when unexpected events occur, not just on interval

## See Also

- `examples/graph_builder_usage.py` - Example of creating planner/worker graphs
- `docs/plans/planner-worker-basic.md` - Original implementation plan
- `testing/test_*.py` - Comprehensive test suite (103 tests)
