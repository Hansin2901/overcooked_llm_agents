import unittest
from unittest.mock import Mock, patch

from overcooked_ai_py.agents.llm.llm_agent import LLMAgent
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


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
            {"timestep": 1, "reasoning": "Getting onion", "action": "↑"},
            {"timestep": 2, "reasoning": "Moving to pot", "action": "→"},
        ]
        result = agent._format_history()
        expected = (
            "RECENT HISTORY:\n"
            '- Step 1: "Getting onion" → ↑\n'
            '- Step 2: "Moving to pot" → →'
        )
        self.assertEqual(result, expected)

    def test_format_history_respects_zero_size(self):
        """History size of 0 returns empty string."""
        agent = LLMAgent(history_size=0)
        agent._history = [{"timestep": 1, "reasoning": "Test", "action": "↑"}]
        result = agent._format_history()
        self.assertEqual(result, "")

    def test_add_to_history_stores_entry(self):
        """Adding to history stores the entry correctly."""
        from overcooked_ai_py.mdp.actions import Direction

        agent = LLMAgent(history_size=5)
        agent._add_to_history(42, "Moving up", Direction.NORTH)

        self.assertEqual(len(agent._history), 1)
        self.assertEqual(agent._history[0]["timestep"], 42)
        self.assertEqual(agent._history[0]["reasoning"], "Moving up")
        self.assertEqual(agent._history[0]["action"], "↑")

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

    def test_history_size_configurable(self):
        """History size can be configured via constructor."""
        agent = LLMAgent(history_size=5)
        self.assertEqual(agent.history_size, 5)

        agent2 = LLMAgent()  # Default
        self.assertEqual(agent2.history_size, 10)

    def test_reset_clears_history(self):
        """Reset clears history for new episode."""
        from overcooked_ai_py.mdp.actions import Direction

        agent = LLMAgent(history_size=5)
        agent._add_to_history(1, "Test", Direction.NORTH)

        self.assertEqual(len(agent._history), 1)

        agent.reset()

        self.assertEqual(len(agent._history), 0)



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
            agent.set_agent_index(0)
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


class TestLLMAgentObservability(unittest.TestCase):
    def test_llm_agent_emits_action_commit(self):
        sink = Mock()
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        state = mdp.get_standard_start_state()

        agent = LLMAgent(
            model_name="gpt-4o",
            observability=sink,
            invoke_config={"callbacks": ["dummy-callback"]},
        )
        agent.agent_index = 0
        agent.mdp = mdp
        agent._system_prompt = "test system prompt"
        agent._graph = Mock()
        agent._graph.invoke.return_value = {"messages": []}

        with patch("overcooked_ai_py.agents.llm.llm_agent.serialize_state", return_value="state"):
            with patch("overcooked_ai_py.agents.llm.llm_agent.set_state"):
                with patch("overcooked_ai_py.agents.llm.llm_agent.get_chosen_action", return_value=Action.STAY):
                    agent.action(state)

        sink.start_role.assert_called_once_with("llm_agent")
        sink.end_role.assert_called_once()
        sink.emit.assert_any_call(
            "action.commit",
            unittest.mock.ANY,
            step=state.timestep,
            agent_role="llm_agent",
        )


if __name__ == "__main__":
    unittest.main()
