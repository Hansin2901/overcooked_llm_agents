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


if __name__ == "__main__":
    unittest.main()
