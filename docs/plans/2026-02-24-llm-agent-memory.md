# LLM Agent Memory System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable memory to LLMAgent to track (reasoning, action) history across timesteps

**Architecture:** Instance-based memory stored in LLMAgent, injected into prompts before current state, extracted from AIMessage after graph execution

**Tech Stack:** Python 3.10, LangGraph, pytest

---

## Task 1: Add History Formatting Method

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/llm_agent.py`
- Test: `testing/llm_agent_test.py` (new file)

**Step 1: Write the failing test**

Create `testing/llm_agent_test.py`:

```python
import unittest
from overcooked_ai_py.agents.llm.llm_agent import LLMAgent


class TestLLMAgentMemory(unittest.TestCase):
    """Test memory system for LLM agent."""

    def test_format_history_empty(self):
        """Empty history returns empty string."""
        agent = LLMAgent(history_size=5)
        result = agent._format_history()
        self.assertEqual(result, "")

    def test_format_history_with_entries(self):
        """History is formatted correctly."""
        agent = LLMAgent(history_size=5)
        agent._history = [
            {"timestep": 1, "reasoning": "Getting onion", "action": "^"},
            {"timestep": 2, "reasoning": "Moving to pot", "action": ">"},
        ]
        result = agent._format_history()
        expected = (
            "RECENT HISTORY:\n"
            '- Step 1: "Getting onion" → ^\n'
            '- Step 2: "Moving to pot" → >'
        )
        self.assertEqual(result, expected)

    def test_format_history_respects_zero_size(self):
        """History size of 0 returns empty string."""
        agent = LLMAgent(history_size=0)
        agent._history = [{"timestep": 1, "reasoning": "Test", "action": "^"}]
        result = agent._format_history()
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest testing/llm_agent_test.py::TestLLMAgentMemory::test_format_history_empty -v`

Expected: FAIL with "AttributeError: 'LLMAgent' object has no attribute '_format_history'"

**Step 3: Add minimal implementation**

In `src/overcooked_ai_py/agents/llm/llm_agent.py`, add after `__init__`:

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

Also add to `__init__` (after existing instance variables):

```python
self.history_size = 10  # Will be parameterized in next task
self._history = []
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest testing/llm_agent_test.py::TestLLMAgentMemory -v`

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/llm_agent.py testing/llm_agent_test.py
git commit -m "feat: add history formatting to LLMAgent

- Add _format_history() method
- Add _history and history_size instance variables
- Add tests for history formatting"
```

---

## Task 2: Add History Storage Method

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/llm_agent.py`
- Test: `testing/llm_agent_test.py`

**Step 1: Write the failing test**

Add to `testing/llm_agent_test.py` in `TestLLMAgentMemory`:

```python
def test_add_to_history_stores_entry(self):
    """Adding to history stores the entry correctly."""
    from overcooked_ai_py.mdp.actions import Direction

    agent = LLMAgent(history_size=5)
    agent._add_to_history(42, "Moving up", Direction.NORTH)

    self.assertEqual(len(agent._history), 1)
    self.assertEqual(agent._history[0]["timestep"], 42)
    self.assertEqual(agent._history[0]["reasoning"], "Moving up")
    self.assertEqual(agent._history[0]["action"], "^")

def test_add_to_history_trims_to_size(self):
    """History is trimmed to history_size."""
    from overcooked_ai_py.mdp.actions import Direction

    agent = LLMAgent(history_size=3)

    # Add 5 entries
    for i in range(5):
        agent._add_to_history(i, f"Action {i}", Direction.NORTH)

    # Should only keep last 3
    self.assertEqual(len(agent._history), 3)
    self.assertEqual(agent._history[0]["timestep"], 2)
    self.assertEqual(agent._history[-1]["timestep"], 4)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest testing/llm_agent_test.py::TestLLMAgentMemory::test_add_to_history_stores_entry -v`

Expected: FAIL with "AttributeError: 'LLMAgent' object has no attribute '_add_to_history'"

**Step 3: Implement _add_to_history**

In `src/overcooked_ai_py/agents/llm/llm_agent.py`, add after `_format_history`:

```python
def _add_to_history(self, timestep, reasoning, action):
    """Add entry to history and maintain size limit."""
    from overcooked_ai_py.mdp.actions import Action

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

**Step 4: Run tests to verify they pass**

Run: `python -m pytest testing/llm_agent_test.py::TestLLMAgentMemory -v`

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/llm_agent.py testing/llm_agent_test.py
git commit -m "feat: add history storage to LLMAgent

- Add _add_to_history() method
- Automatically trim history to history_size
- Add tests for history storage and trimming"
```

---

## Task 3: Add Reasoning Extraction Method

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/llm_agent.py`
- Test: `testing/llm_agent_test.py`

**Step 1: Write the failing test**

Add to `testing/llm_agent_test.py` in `TestLLMAgentMemory`:

```python
def test_extract_reasoning_from_action_message(self):
    """Extracts reasoning from AIMessage with action tool call."""
    from langchain_core.messages import AIMessage
    from overcooked_ai_py.agents.llm.tools import ACTION_TOOL_NAMES

    agent = LLMAgent(history_size=5)

    messages = [
        AIMessage(
            content="I'll move up to the pot",
            tool_calls=[{"name": "move_up", "args": {}, "id": "1"}]
        )
    ]

    result = agent._extract_reasoning(messages)
    self.assertEqual(result, "I'll move up to the pot")

def test_extract_reasoning_empty_content(self):
    """Returns fallback when content is empty."""
    from langchain_core.messages import AIMessage

    agent = LLMAgent(history_size=5)

    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "move_up", "args": {}, "id": "1"}]
        )
    ]

    result = agent._extract_reasoning(messages)
    self.assertEqual(result, "(no reasoning provided)")

def test_extract_reasoning_no_messages(self):
    """Returns fallback when no messages."""
    agent = LLMAgent(history_size=5)
    result = agent._extract_reasoning([])
    self.assertEqual(result, "(no messages returned)")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest testing/llm_agent_test.py::TestLLMAgentMemory::test_extract_reasoning_from_action_message -v`

Expected: FAIL with "AttributeError: 'LLMAgent' object has no attribute '_extract_reasoning'"

**Step 3: Implement _extract_reasoning**

In `src/overcooked_ai_py/agents/llm/llm_agent.py`, add after `_add_to_history`:

```python
def _extract_reasoning(self, messages):
    """Extract reasoning text from LLM's final decision.

    Returns the content of the AIMessage that called an action tool,
    or a descriptive fallback if extraction fails.
    """
    from langchain_core.messages import AIMessage
    from overcooked_ai_py.agents.llm.tools import ACTION_TOOL_NAMES

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

**Step 4: Run tests to verify they pass**

Run: `python -m pytest testing/llm_agent_test.py::TestLLMAgentMemory -v`

Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/llm_agent.py testing/llm_agent_test.py
git commit -m "feat: add reasoning extraction to LLMAgent

- Add _extract_reasoning() method with error handling
- Extract from AIMessage with action tool calls
- Fallback for empty content and errors
- Add tests for reasoning extraction"
```

---

## Task 4: Make History Size Configurable

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/llm_agent.py`
- Test: `testing/llm_agent_test.py`

**Step 1: Write the failing test**

Add to `testing/llm_agent_test.py` in `TestLLMAgentMemory`:

```python
def test_history_size_configurable(self):
    """History size can be configured via constructor."""
    agent = LLMAgent(history_size=5)
    self.assertEqual(agent.history_size, 5)

    agent2 = LLMAgent()  # Default
    self.assertEqual(agent2.history_size, 10)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest testing/llm_agent_test.py::TestLLMAgentMemory::test_history_size_configurable -v`

Expected: FAIL (history_size is hardcoded to 10)

**Step 3: Update __init__ signature**

In `src/overcooked_ai_py/agents/llm/llm_agent.py`, modify `__init__`:

```python
def __init__(self, model_name="gpt-4o", debug=False, horizon=None, api_base=None, history_size=10):
    self.model_name = model_name
    self.debug = debug
    self.horizon = horizon
    self.api_base = api_base
    self.history_size = history_size  # Now parameterized
    self._graph = None
    self._system_prompt = None
    self._history = []
    super().__init__()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest testing/llm_agent_test.py::TestLLMAgentMemory::test_history_size_configurable -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/llm_agent.py testing/llm_agent_test.py
git commit -m "feat: make history size configurable

- Add history_size parameter to __init__
- Default to 10 entries
- Add test for configuration"
```

---

## Task 5: Update Reset to Clear History

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/llm_agent.py`
- Test: `testing/llm_agent_test.py`

**Step 1: Write the failing test**

Add to `testing/llm_agent_test.py` in `TestLLMAgentMemory`:

```python
def test_reset_clears_history(self):
    """Reset clears history for new episode."""
    from overcooked_ai_py.mdp.actions import Direction

    agent = LLMAgent(history_size=5)
    agent._add_to_history(1, "Test", Direction.NORTH)

    self.assertEqual(len(agent._history), 1)

    agent.reset()

    self.assertEqual(len(agent._history), 0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest testing/llm_agent_test.py::TestLLMAgentMemory::test_reset_clears_history -v`

Expected: FAIL (history is not cleared in reset)

**Step 3: Update reset method**

In `src/overcooked_ai_py/agents/llm/llm_agent.py`, modify `reset`:

```python
def reset(self):
    """Reset agent state between episodes."""
    super().reset()
    self._graph = None
    self._system_prompt = None
    self._history = []  # Clear history
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest testing/llm_agent_test.py::TestLLMAgentMemory::test_reset_clears_history -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/llm_agent.py testing/llm_agent_test.py
git commit -m "feat: clear history on reset

- Update reset() to clear _history
- Add test for reset behavior"
```

---

## Task 6: Integrate History into action() Method

**Files:**
- Modify: `src/overcooked_ai_py/agents/llm/llm_agent.py`

**Step 1: Update action() to inject history**

In `src/overcooked_ai_py/agents/llm/llm_agent.py`, modify the `action` method:

Find this code (around line 79-92):

```python
def action(self, state):
    """Decide an action for the current state.
    ...
    """
    # Serialize state to text
    state_text = serialize_state(self.mdp, state, self.agent_index, self.horizon)

    # Update tool context
    set_state(state, self.agent_index)

    # Run LangGraph agent
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=self._system_prompt),
        HumanMessage(content=f"Current game state:\n{state_text}\n\nDecide your action."),
    ]
```

Replace with:

```python
def action(self, state):
    """Decide an action for the current state.
    ...
    """
    # Serialize state to text
    state_text = serialize_state(self.mdp, state, self.agent_index, self.horizon)

    # Update tool context
    set_state(state, self.agent_index)

    # Build history text
    history_text = self._format_history()

    # Construct prompt with history
    if history_text:
        prompt = f"{history_text}\n\nCurrent game state:\n{state_text}\n\nDecide your action."
    else:
        prompt = f"Current game state:\n{state_text}\n\nDecide your action."

    # Run LangGraph agent
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=self._system_prompt),
        HumanMessage(content=prompt),
    ]
```

**Step 2: Add error handling and reasoning capture**

Find this code (around line 93-108):

```python
    result = self._graph.invoke({"messages": messages})

    # Extract the action from the tool module
    chosen = get_chosen_action()
    if chosen is None:
        # LLM didn't call an action tool — default to STAY
        if self.debug:
            print(f"  [LLMAgent] No action tool called, defaulting to STAY")
        chosen = Action.STAY

    if self.debug:
        action_name = Action.ACTION_TO_CHAR.get(chosen, str(chosen))
        print(f"  [Step {state.timestep}] Player {self.agent_index} -> {action_name}")

    action_probs = self.a_probs_from_action(chosen)
    return chosen, {"action_probs": action_probs}
```

Replace with:

```python
    # Execute graph with error handling
    try:
        result = self._graph.invoke({"messages": messages})
        reasoning = self._extract_reasoning(result["messages"])
    except Exception as e:
        # Graph failed - log warning and use fallback
        if self.debug:
            print(f"  [LLMAgent] Graph execution failed: {e}")
        reasoning = "(graph execution failed)"
        result = None

    # Extract the action from the tool module
    chosen = get_chosen_action()
    if chosen is None:
        # LLM didn't call an action tool — default to STAY
        if self.debug:
            print(f"  [LLMAgent] No action tool called, defaulting to STAY")
        chosen = Action.STAY

    # Store in history
    self._add_to_history(state.timestep, reasoning, chosen)

    if self.debug:
        action_name = Action.ACTION_TO_CHAR.get(chosen, str(chosen))
        print(f"  [Step {state.timestep}] Player {self.agent_index} -> {action_name}")

    action_probs = self.a_probs_from_action(chosen)
    return chosen, {"action_probs": action_probs}
```

**Step 3: Manual test with run_llm_agent.py**

Run: `python scripts/run_llm_agent.py --layout cramped_room --horizon 10 --debug 2>&1 | head -50`

Expected: See "RECENT HISTORY:" in debug output after first action

**Step 4: Commit**

```bash
git add src/overcooked_ai_py/agents/llm/llm_agent.py
git commit -m "feat: integrate memory into action method

- Inject history into prompt before current state
- Add error handling around graph invocation
- Capture reasoning after each action
- Store (timestep, reasoning, action) in history"
```

---

## Task 7: Add Integration Tests

**Files:**
- Test: `testing/llm_agent_test.py`

**Step 1: Write integration test**

Add to `testing/llm_agent_test.py`:

```python
class TestLLMAgentMemoryIntegration(unittest.TestCase):
    """Integration tests for memory system."""

    def test_memory_accumulates_across_actions(self):
        """History accumulates across multiple action calls."""
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        from unittest.mock import Mock, patch

        # Create a simple test environment
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        env = OvercookedEnv.from_mdp(mdp, horizon=10)

        # Create agent with small history
        agent = LLMAgent(model_name="gpt-4o", debug=False, history_size=3)

        # Mock the graph to avoid actual LLM calls
        with patch.object(agent, '_graph') as mock_graph:
            # Setup mock to return valid result
            from langchain_core.messages import AIMessage
            mock_graph.invoke.return_value = {
                "messages": [
                    AIMessage(
                        content="Test reasoning",
                        tool_calls=[{"name": "wait", "args": {}, "id": "1"}]
                    )
                ]
            }

            # Set up agent
            agent.set_mdp(mdp)
            env.reset()

            # Mock get_chosen_action to return STAY
            from overcooked_ai_py.mdp.actions import Action
            with patch('overcooked_ai_py.agents.llm.llm_agent.get_chosen_action', return_value=Action.STAY):
                # Take several actions
                state = env.state
                for i in range(5):
                    agent.action(state)
                    state.timestep = i + 1  # Increment timestep

                # Should have 3 entries (history_size limit)
                self.assertEqual(len(agent._history), 3)

                # Should have most recent entries
                self.assertEqual(agent._history[0]["timestep"], 2)
                self.assertEqual(agent._history[-1]["timestep"], 4)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it passes**

Run: `python -m pytest testing/llm_agent_test.py::TestLLMAgentMemoryIntegration -v`

Expected: PASS

**Step 3: Commit**

```bash
git add testing/llm_agent_test.py
git commit -m "test: add integration tests for memory system

- Test memory accumulation across multiple actions
- Test history trimming with mocked LLM calls"
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Add section after the LLM Agent Architecture section (around line 180):

```markdown
## LLM Agent Memory System

The LLM agent includes a configurable memory system that tracks recent (reasoning, action) history:

**Memory Features:**
- **Configurable History Size**: Default 10 entries, adjustable via `history_size` parameter
- **Instance-Based**: Each `LLMAgent` maintains its own independent memory
- **Automatic Trimming**: History is automatically trimmed to the configured size
- **Error Resilient**: Defensive error handling for graph failures

**History Format:**

Each action creates a history entry:
```python
{
    "timestep": 42,
    "reasoning": "I'll move to the pot to add my onion",
    "action": "^"  # Human-readable action character
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
   - Step 45: "I need to get to the onion dispenser" → move_up
   - Step 46: "Continuing toward the dispenser" → move_up
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
```

**Step 2: Verify formatting**

Run: `head -n 220 CLAUDE.md | tail -n 40`

Expected: See the new section properly formatted

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document LLM agent memory system

- Add memory system section to CLAUDE.md
- Document configuration, usage, and behavior
- Reference design doc for future enhancements"
```

---

## Task 9: Final Verification

**Files:**
- All modified files

**Step 1: Run full test suite**

Run: `python -m pytest testing/llm_agent_test.py -v`

Expected: All tests PASS

**Step 2: Run manual integration test**

Run: `python scripts/run_llm_agent.py --layout cramped_room --horizon 20 --debug 2>&1 | grep -A 3 "RECENT HISTORY"`

Expected: After first few actions, see "RECENT HISTORY:" sections in output

**Step 3: Verify all files committed**

Run: `git status`

Expected: "nothing to commit, working tree clean"

**Step 4: Final commit (if needed)**

If there are any uncommitted changes:

```bash
git add -A
git commit -m "chore: final cleanup for memory system"
```

---

## Success Criteria

- ✅ All tests pass (`pytest testing/llm_agent_test.py`)
- ✅ History is configurable via `history_size` parameter
- ✅ History accumulates and trims correctly
- ✅ Error handling prevents crashes on graph failures
- ✅ History is injected into prompts
- ✅ Reasoning is extracted and stored after each action
- ✅ Documentation is updated
- ✅ All changes are committed

## Testing Checklist

- [ ] Unit tests for `_format_history()` pass
- [ ] Unit tests for `_add_to_history()` pass
- [ ] Unit tests for `_extract_reasoning()` pass
- [ ] Unit tests for configuration pass
- [ ] Unit tests for reset behavior pass
- [ ] Integration tests pass
- [ ] Manual test with `run_llm_agent.py` shows history in debug output
