# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the video game Overcooked. The goal is to deliver soups by coordinating tasks like gathering ingredients, cooking, and delivery. The codebase includes an MDP environment, planning algorithms, agent implementations (including LLM-based agents), and a web demo interface.

**Core Paper**: [On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789) (NeurIPS 2019)

**Note**: The `human_aware_rl` directory containing DRL/BC implementations is deprecated (see [issue #162](https://github.com/HumanCompatibleAI/overcooked_ai/issues/162)).

## Development Setup

### Installation

The project uses `uv` for dependency management (recommended):

```bash
# Clone and setup
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
cd overcooked_ai

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # or `.venv/Scripts/activate` on Windows
uv sync

# Install in editable mode (required for coverage tests)
uv pip install -e .
```

Alternative installation via pip:
```bash
pip install overcooked-ai
```

### Environment Configuration

For LLM agents, create a `.env` file in the project root (see `.env.example`):

**Standard providers:**
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Custom OpenAI-compatible endpoints (e.g., TritonAI):**
```bash
# IMPORTANT: Prefix model with "openai/" for LiteLLM
LLM_MODEL=openai/api-gpt-oss-120b
LLM_API_BASE=https://tritonai-api.ucsd.edu/v1
LLM_API_KEY=your-key-here
```

## Common Commands

### Testing

```bash
# Quick verification test (run from project root)
python testing/overcooked_test.py

# Full test suite (5-10 minutes)
python -m unittest discover -s testing/ -p "*_test.py"

# Run specific test file
python testing/agent_test.py
python testing/planners_test.py

# With coverage
source .venv/bin/activate
coverage run -m unittest discover -s testing/ -p "*_test.py"
coverage report
```

### Running LLM Agents

**Basic usage** (reads from `.env` automatically):
```bash
# Run with real-time visualization (recommended)
source .env && uv run python scripts/run_llm_agent.py \
  --layout cramped_room \
  --horizon 20 \
  --visualize \
  --fps 2 \
  --debug

# Run without visualization (text only)
source .env && uv run python scripts/run_llm_agent.py \
  --layout cramped_room \
  --horizon 20 \
  --debug
```

**Configuration via `.env` file** (recommended):
```bash
# .env file should contain:
LLM_MODEL=openai/moonshotai.kimi-k2.5
LLM_API_BASE=https://tritonai-api.ucsd.edu/v1
LLM_API_KEY=your-key-here

# Or for standard OpenAI:
OPENAI_API_KEY=sk-...
```

**Command-line options:**
- `--model MODEL`: LiteLLM model name (default: from LLM_MODEL env or "gpt-4o")
- `--layout LAYOUT`: Kitchen layout (default: "cramped_room")
- `--horizon N`: Episode length in timesteps (default: 200)
- `--debug`: Print LLM reasoning for each decision
- `--visualize`: Show real-time pygame visualization window
- `--fps N`: Frames per second for visualization (default: 2)

**Available layouts:**
- `cramped_room` - Small, tight kitchen (good for testing)
- `asymmetric_advantages` - Asymmetric layout requiring coordination
- `coordination_ring` - Circular kitchen design
- `forced_coordination` - Requires tight teamwork
- `counter_circuit` - Circuit-style kitchen layout

**Custom OpenAI-compatible endpoints:**
The agent now automatically reads `LLM_API_BASE` and `LLM_API_KEY` from environment variables, making it easy to use custom endpoints like TritonAI or other OpenAI-compatible APIs.

### Interactive Tutorial

See `Overcooked Tutorial.ipynb` for a comprehensive guide on using the environment.

## Code Architecture

### Module Structure

```
src/
├── overcooked_ai_py/          # Core game engine and agents
│   ├── mdp/                   # Markov Decision Process
│   │   ├── overcooked_mdp.py     # Core game logic (Recipe, OvercookedState, OvercookedGridworld)
│   │   ├── overcooked_env.py     # Gymnasium-compatible environment wrapper
│   │   ├── actions.py            # Action definitions (Direction, Action)
│   │   ├── layout_generator.py   # Programmatic layout generation
│   │   └── overcooked_trajectory.py  # Trajectory data structures
│   ├── agents/                # Agent implementations
│   │   ├── agent.py              # Base Agent class and standard agents (RandomAgent, GreedyHumanModel, etc.)
│   │   ├── benchmarking.py       # Agent evaluation utilities
│   │   └── llm/                  # LLM-based agents (NEW)
│   │       ├── llm_agent.py         # LLMAgent using LangGraph ReAct loop
│   │       ├── graph.py             # LangGraph StateGraph construction
│   │       ├── tools.py             # LLM tools (observation & action tools)
│   │       └── state_serializer.py  # State-to-text serialization
│   ├── planning/              # Near-optimal planning algorithms
│   │   ├── planners.py           # MotionPlanner, MediumLevelActionManager
│   │   └── search.py             # A* search, shortest path
│   ├── visualization/         # Rendering and display
│   │   ├── state_visualizer.py   # State rendering to images
│   │   └── pygame_utils.py       # Pygame integration
│   └── data/                  # Static resources
│       ├── layouts/              # .layout files defining level layouts
│       ├── graphics/             # Sprites and graphics assets
│       └── fonts/                # Font files
│
├── human_aware_rl/            # DEPRECATED: DRL training code
│   ├── ppo/                      # PPO (Proximal Policy Optimization)
│   ├── imitation/                # Behavior cloning
│   └── rllib/                    # RLlib integration
│
└── overcooked_demo/           # Web-based demo (Flask + SocketIO)
    └── server/
        ├── app.py                # Flask application
        └── game.py               # Game server logic

testing/                       # Unit tests
```

### Key Abstractions

**OvercookedGridworld** (`mdp/overcooked_mdp.py`):
- Core MDP defining game rules, state transitions, and reward structure
- Initialized from `.layout` files or programmatically
- Main methods: `get_actions()`, `get_state_transition()`, `get_reward()`

**OvercookedEnv** (`mdp/overcooked_env.py`):
- Gymnasium-compatible environment wrapper around OvercookedGridworld
- Manages episode lifecycle, horizon, and trajectory collection
- Instantiate via: `OvercookedEnv.from_mdp(mdp, horizon=400)`

**Agent** (`agents/agent.py`):
- Base class for all agents
- Key method: `action(state) -> (action, action_info)`
- Built-in agents: `RandomAgent`, `GreedyHumanModel`, `StayAgent`, `FixedPlanAgent`
- Agents must call `set_mdp(mdp)` and `reset()` before use

**LLMAgent** (`agents/llm/llm_agent.py`):
- Agent that uses LLMs (via LiteLLM + LangGraph) to decide actions each timestep
- Implements a ReAct loop where the LLM can call observation tools and action tools
- Serializes game state to text for LLM processing
- Usage: `LLMAgent(model_name="gpt-4o", debug=False, horizon=200)`

**MotionPlanner** (`planning/planners.py`):
- Computes optimal motion plans using A* search
- Used by greedy agents and LLM tools for distance calculations
- Cached to disk for performance (stored in `data/planners/`)

**MediumLevelActionManager** (`planning/planners.py`):
- High-level action planner that decomposes goals into motion primitives
- Required for `GreedyHumanModel` and other planning-based agents
- Computationally expensive; results are cached

### State Representation

An `OvercookedState` contains:
- `players`: List of `PlayerState` objects (position, orientation, held object)
- `objects`: Dict mapping positions to `ObjectState` (pots, ingredients, soups, dishes, etc.)
- `timestep`: Current timestep in episode
- Game progression tracked through object states and player inventories

### Agent Patterns

**Creating and running a pair of agents**:
```python
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent

mdp = OvercookedGridworld.from_layout_name("cramped_room")
env = OvercookedEnv.from_mdp(mdp, horizon=400)

agent0 = RandomAgent()
agent1 = RandomAgent()
agent_pair = AgentPair(agent0, agent1)

env.reset()
agent_pair.set_mdp(mdp)

state = env.state
joint_action, infos = agent_pair.joint_action(state)
next_state, rewards, done, info = env.step(joint_action, infos)
```

**Using planners with agents**:
```python
from overcooked_ai_py.planning.planners import (
    MediumLevelActionManager,
    NO_COUNTERS_PARAMS
)
from overcooked_ai_py.agents.agent import GreedyHumanModel

# MediumLevelActionManager is computationally expensive; use caching
mlam = MediumLevelActionManager.from_pickle_or_compute(
    mdp, NO_COUNTERS_PARAMS, force_compute=False
)
greedy_agent = GreedyHumanModel(mlam)
```

## LLM Agent Architecture

The LLM agent (`agents/llm/`) uses a tool-calling ReAct pattern:

1. **State Serialization** (`state_serializer.py`): Converts game state to text description
2. **System Prompt** (`state_serializer.py`): One-time prompt with game rules and layout info
3. **LangGraph Loop** (`graph.py`):
   - LLM can call observation tools (check surroundings, distances, etc.)
   - LLM must call exactly one action tool to commit to a game action
   - Action tools terminate the loop
4. **Tools** (`tools.py`):
   - Observation tools: `get_surroundings()`, `distance_to()`, `partner_info()`
   - Action tools: `move_up()`, `move_down()`, `move_left()`, `move_right()`, `interact()`, `wait()`

The agent uses module-level state to pass game context to tools (set via `init_tools()` and `set_state()`).

## LLM Agent Memory System

✅ **Status: Complete and Production-Ready** (Merged to master on 2026-02-26)

The LLM agent includes a configurable memory system that tracks recent (reasoning, action) history across timesteps within an episode.

**Memory Features:**
- **Configurable History Size**: Default 10 entries, adjustable via `history_size` parameter
- **Instance-Based**: Each `LLMAgent` maintains its own independent memory
- **Automatic Trimming**: History is automatically trimmed to the configured size
- **Error Resilient**: Defensive error handling for graph failures
- **Fully Tested**: 11 unit + integration tests, all passing

**History Format:**

Each action creates a history entry:
```python
{
    "timestep": 42,
    "reasoning": "I'll move to the pot to add my onion",
    "action": "↑"  # Human-readable action character
}
```

**Usage:**

```python
# Default: 10 entries
llm_agent = LLMAgent(model_name="gpt-4o", debug=True, horizon=200)

# Custom history size
llm_agent = LLMAgent(model_name="gpt-4o", history_size=5)

# Disable memory
llm_agent = LLMAgent(model_name="gpt-4o", history_size=0)
```

**How It Works:**

1. Before each action, the agent's recent history is injected into the prompt:
   ```
   RECENT HISTORY:
   - Step 45: "I need to get to the onion dispenser" → ↑
   - Step 46: "Continuing toward the dispenser" → ↑
   - Step 47: "At the dispenser, picking up onion" → interact

   Current game state:
   Timestep: 48 / 200
   ...
   ```

2. After the graph executes, reasoning is extracted from the LLM's final AIMessage
3. The (timestep, reasoning, action) tuple is stored in history
4. History is automatically trimmed to `history_size` entries

**Future Enhancements:**

See `docs/plans/2026-02-24-llm-agent-memory-design.md` for planned improvements:
- Enhanced reasoning capture (full observation summaries)
- Explicit goal management with `update_goal()` tool
- Structured goal hierarchies with subgoal tracking

## Working with Layouts

Layouts are stored as `.layout` files in `src/overcooked_ai_py/data/layouts/`. They use a grid-based text format:

- `X`: Wall
- `O`: Counter
- `P`: Pot
- `T`: Tomato dispenser
- `D`: Dish dispenser
- `S`: Serving area
- `1`, `2`: Player start positions

Create new layouts by either:
1. Writing `.layout` files manually
2. Using `layout_generator.py` to generate random layouts programmatically

## Dependencies

Core dependencies (always required):
- `gymnasium`: Environment interface
- `numpy`: Numerical operations
- `pygame`: Visualization
- `scipy`, `dill`, `tqdm`: Utilities

LLM agent dependencies (install separately):
- `python-dotenv`: Environment variable loading
- `langchain-core`, `langchain-community`: LangChain framework
- `langgraph`: State graph for agent loops
- `litellm`: Unified LLM API

Human-aware RL dependencies (deprecated, `harl` extra):
- `ray[rllib]`: Distributed RL training
- `tensorflow`: Deep learning
- `wandb`: Experiment tracking

## Python Version

Requires Python 3.10 (specified as `>=3.10,<3.11` in `pyproject.toml`).

## Notes for Development

- Always activate the virtual environment before running code
- Use `uv` for dependency management to leverage the lockfile (`uv.lock`)
- When adding new features to LLM agents, update tools in `agents/llm/tools.py` and ensure they follow the LangChain tool decorator pattern
- Planner computations are expensive; always use `from_pickle_or_compute()` with caching
- Test changes with both the quick test (`testing/overcooked_test.py`) and full suite
- The CI pipeline runs tests on Python 3.10.16 using `uv` and GitHub Actions
