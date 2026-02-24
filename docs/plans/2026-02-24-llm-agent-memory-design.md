# LLM Agent Memory System Design

**Date:** 2026-02-24
**Status:** Approved
**Target:** Baseline implementation with future extensibility

## Overview

Add memory capabilities to the LLM agent to maintain context across timesteps within an episode. Currently, the agent is stateless between timesteps, making decisions purely based on the current game state snapshot. This design introduces a configurable history of (reasoning, action) tuples that allows the agent to remember its recent decisions and maintain implicit goal continuity.

## Requirements

### Baseline (v1)
1. **History Tracking:** Store the last N timesteps of (reasoning, action) tuples
2. **Configurable Size:** History size should be configurable (default: 10 entries)
3. **Instance-Based Memory:** Each `LLMAgent` instance maintains its own independent memory
4. **Implicit Goals:** Agent develops goals naturally through reasoning (no explicit goal prompting)
5. **Error Handling:** Defensive error handling for graph failures and missing reasoning

### Future Enhancements (Shelved for v2+)
- **Enhanced Reasoning Capture:** Full observation summary (all tool calls within timestep)
- **Explicit Goal Management:** `update_goal(new_goal)` tool for dynamic goal updating
- **Goal Persistence:** Dedicated goal tracking with structured subgoal decomposition

## Architecture

### Core Changes

All changes are isolated to `LLMAgent` class in `src/overcooked_ai_py/agents/llm/llm_agent.py`:

1. **Instance Variables:**
   - `self._history = []` - stores history entries
   - `self.history_size` - configurable max entries (default: 10)

2. **Message Construction:**
   - Inject formatted history into `HumanMessage` before current state
   - Format: `RECENT HISTORY:\n- Step X: "reasoning" → action`

3. **Post-Action Capture:**
   - Extract reasoning from LLM's final AIMessage
   - Store (timestep, reasoning, action) tuple
   - Auto-trim to `history_size`

**Unchanged Components:**
- `graph.py` - LangGraph structure unchanged
- `tools.py` - Tools unaware of history
- `state_serializer.py` - State serialization unchanged

## Data Structures

### History Entry Format

```python
{
    "timestep": int,        # Game timestep when action was taken
    "reasoning": str,       # LLM's final reasoning text before action
    "action": str,          # Human-readable action (e.g., "move_up", "interact")
}
```

### Example History State

```python
self._history = [
    {"timestep": 45, "reasoning": "I need to get to the onion dispenser", "action": "move_up"},
    {"timestep": 46, "reasoning": "Continuing toward the dispenser", "action": "move_up"},
    {"timestep": 47, "reasoning": "At the dispenser, picking up onion", "action": "interact"},
    {"timestep": 48, "reasoning": "Got the onion, heading to pot", "action": "move_right"},
    {"timestep": 49, "reasoning": "Almost at the pot", "action": "move_up"},
]
```

## Message Construction

### Prompt Assembly

The `action()` method constructs prompts with history prepended:

```python
def action(self, state):
    state_text = serialize_state(self.mdp, state, self.agent_index, self.horizon)
    history_text = self._format_history()

    if history_text:
        prompt = f"{history_text}\n\nCurrent game state:\n{state_text}\n\nDecide your action."
    else:
        prompt = f"Current game state:\n{state_text}\n\nDecide your action."

    messages = [
        SystemMessage(content=self._system_prompt),
        HumanMessage(content=prompt),
    ]

    # ... execute graph
```

### History Formatting

```python
def _format_history(self):
    """Format history entries for display to LLM."""
    if not self._history or self.history_size == 0:
        return ""

    lines = ["RECENT HISTORY:"]
    for entry in self._history:
        lines.append(
            f"- Step {entry['timestep']}: \"{entry['reasoning']}\" → {entry['action']}"
        )
    return "\n".join(lines)
```

### Example LLM Input

```
RECENT HISTORY:
- Step 45: "I need to get to the onion dispenser" → move_up
- Step 46: "Continuing toward the dispenser" → move_up
- Step 47: "At the dispenser, picking up onion" → interact

Current game state:
Timestep: 48 / 200
GRID:
  ###########
  #O  P    S#
  ...
```

## Reasoning Extraction

### Extraction Logic

Reasoning is captured from the AIMessage that called an action tool:

```python
def _extract_reasoning(self, messages):
    """Extract reasoning text from LLM's final decision.

    Returns the content of the AIMessage that called an action tool,
    or a descriptive fallback if extraction fails.
    """
    if not messages:
        return "(no messages returned)"

    try:
        # Look backwards for AIMessage with action tool call
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                # Check if this message called an action tool
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc["name"] in ACTION_TOOL_NAMES:
                            # Found action decision
                            return msg.content.strip() if msg.content else "(no reasoning provided)"
                # Fallback to any AIMessage content
                elif msg.content:
                    return msg.content.strip()

        return "(no reasoning found)"
    except Exception as e:
        if self.debug:
            print(f"  [LLMAgent] Reasoning extraction failed: {e}")
        return "(extraction failed)"
```

### Error Handling

The `action()` method includes defensive error handling:

```python
try:
    result = self._graph.invoke({"messages": messages})
    reasoning = self._extract_reasoning(result["messages"])
except Exception as e:
    if self.debug:
        print(f"  [LLMAgent] Graph execution failed: {e}")
    reasoning = "(graph execution failed)"
    result = None

chosen = get_chosen_action()
if chosen is None:
    if self.debug:
        print(f"  [LLMAgent] No action tool called, defaulting to STAY")
    chosen = Action.STAY

self._add_to_history(state.timestep, reasoning, chosen)
```

### History Storage

```python
def _add_to_history(self, timestep, reasoning, action):
    """Add entry to history and maintain size limit."""
    action_name = Action.ACTION_TO_CHAR.get(action, str(action))

    self._history.append({
        "timestep": timestep,
        "reasoning": reasoning,
        "action": action_name,
    })

    # Trim to history_size
    if len(self._history) > self.history_size:
        self._history = self._history[-self.history_size:]
```

## Configuration

### Initialization

```python
def __init__(self, model_name="gpt-4o", debug=False, horizon=None,
             api_base=None, history_size=10):
    self.model_name = model_name
    self.debug = debug
    self.horizon = horizon
    self.api_base = api_base
    self.history_size = history_size  # NEW

    self._graph = None
    self._system_prompt = None
    self._history = []  # NEW

    super().__init__()
```

### Reset Behavior

```python
def reset(self):
    """Reset agent state between episodes."""
    super().reset()
    self._graph = None
    self._system_prompt = None
    self._history = []  # Clear history for new episode
```

### Usage Examples

```python
# Default: 10 entries
llm_agent = LLMAgent(model_name="gpt-4o", debug=True, horizon=200)

# Custom history size: 5 entries
llm_agent = LLMAgent(model_name="gpt-4o", debug=True,
                     horizon=200, history_size=5)

# Disable history: 0 entries
llm_agent = LLMAgent(model_name="gpt-4o", debug=True,
                     horizon=200, history_size=0)
```

## Future Enhancements

### Enhanced Reasoning Capture (v2)

Instead of just the final decision text, capture a summary of all observations within the timestep:

```python
# Current (v1):
"I'll move to the pot to add my onion"

# Enhanced (v2):
"Checked surroundings: pot at (3,1) with 1/3 ingredients, onion dispenser at (1,1).
Partner is at pot. Decision: Move to pot to add my onion."
```

**Implementation:** Track all tool calls and responses within the LangGraph loop, generate summary before storing in history.

### Explicit Goal Management (v2+)

Add a `update_goal(new_goal)` tool that allows the LLM to explicitly set and update its high-level goal:

```python
@tool
def update_goal(new_goal: str) -> str:
    """Update your high-level goal for this episode."""
    _agent._goal = new_goal
    return f"Goal updated to: {new_goal}"
```

**Prompt Integration:**
```
CURRENT GOAL: Make three onion soups

RECENT HISTORY:
- Step 45: "Moving to onion dispenser" → move_up
...
```

**Benefits:**
- Explicit goal tracking for better long-term planning
- Agent can update goals when circumstances change
- Easier to debug and understand agent strategy

### Goal Persistence and Subgoals (v3+)

Structured goal hierarchy with completion tracking:

```python
{
    "main_goal": "Deliver 3 soups",
    "current_subgoal": "Get onion from dispenser",
    "completed_subgoals": ["Navigate to kitchen area"],
    "progress": {"soups_delivered": 1, "target": 3}
}
```

## Implementation Checklist

- [ ] Add `history_size` parameter to `LLMAgent.__init__()`
- [ ] Add `_history` instance variable initialization
- [ ] Implement `_format_history()` method
- [ ] Implement `_extract_reasoning()` method
- [ ] Implement `_add_to_history()` method
- [ ] Update `action()` method to inject history into prompt
- [ ] Update `action()` method to capture reasoning post-execution
- [ ] Add error handling with try-except around graph invocation
- [ ] Update `reset()` to clear history
- [ ] Add unit tests for history tracking
- [ ] Add unit tests for error handling edge cases
- [ ] Update CLAUDE.md with memory system documentation
