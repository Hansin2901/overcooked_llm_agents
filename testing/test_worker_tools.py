"""Tests for the worker tools factory pattern.

Verifies that:
- Factory creates tools successfully
- Two workers with different ToolStates don't share state
- Action tools set the correct action on their bound ToolState
- Observation tools read from their bound ToolState
"""

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from overcooked_ai_py.agents.llm.tool_state import ToolState
from overcooked_ai_py.agents.llm.worker_tools import create_worker_tools
from overcooked_ai_py.mdp.actions import Action, Direction


class TestCreateWorkerTools(unittest.TestCase):
    """Test that the factory function creates the expected tool sets."""

    def test_returns_three_element_tuple(self):
        ts = ToolState()
        result = create_worker_tools(ts)
        self.assertEqual(len(result), 3)

    def test_observation_tool_names(self):
        ts = ToolState()
        obs_tools, _, _ = create_worker_tools(ts)
        names = {t.name for t in obs_tools}
        self.assertEqual(names, {"get_surroundings", "get_pot_details", "check_path"})

    def test_action_tool_names(self):
        ts = ToolState()
        _, action_tools, action_tool_names = create_worker_tools(ts)
        expected = {"move_up", "move_down", "move_left", "move_right", "wait", "interact"}
        names_from_tools = {t.name for t in action_tools}
        self.assertEqual(names_from_tools, expected)
        self.assertEqual(action_tool_names, expected)

    def test_total_tool_count(self):
        ts = ToolState()
        obs, act, _ = create_worker_tools(ts)
        self.assertEqual(len(obs), 3)
        self.assertEqual(len(act), 6)


class TestActionToolsSetAction(unittest.TestCase):
    """Test that each action tool sets the correct action on its ToolState."""

    def setUp(self):
        self.ts = ToolState()
        _, self.action_tools, _ = create_worker_tools(self.ts)
        self.tool_map = {t.name: t for t in self.action_tools}

    def test_move_up(self):
        self.tool_map["move_up"].invoke({})
        self.assertEqual(self.ts.chosen_action, Direction.NORTH)

    def test_move_down(self):
        self.tool_map["move_down"].invoke({})
        self.assertEqual(self.ts.chosen_action, Direction.SOUTH)

    def test_move_left(self):
        self.tool_map["move_left"].invoke({})
        self.assertEqual(self.ts.chosen_action, Direction.WEST)

    def test_move_right(self):
        self.tool_map["move_right"].invoke({})
        self.assertEqual(self.ts.chosen_action, Direction.EAST)

    def test_wait(self):
        self.tool_map["wait"].invoke({})
        self.assertEqual(self.ts.chosen_action, Action.STAY)

    def test_interact(self):
        self.tool_map["interact"].invoke({})
        self.assertEqual(self.ts.chosen_action, Action.INTERACT)


class TestWorkersDoNotShareState(unittest.TestCase):
    """Two workers with different ToolStates must be fully isolated."""

    def test_action_tools_isolated(self):
        ts1 = ToolState()
        ts2 = ToolState()
        _, act1, _ = create_worker_tools(ts1)
        _, act2, _ = create_worker_tools(ts2)

        act1_map = {t.name: t for t in act1}
        act2_map = {t.name: t for t in act2}

        # Worker 1 moves up
        act1_map["move_up"].invoke({})
        # Worker 2 moves right
        act2_map["move_right"].invoke({})

        self.assertEqual(ts1.chosen_action, Direction.NORTH)
        self.assertEqual(ts2.chosen_action, Direction.EAST)

    def test_action_does_not_leak(self):
        ts1 = ToolState()
        ts2 = ToolState()
        _, act1, _ = create_worker_tools(ts1)
        _, _, _ = create_worker_tools(ts2)

        act1_map = {t.name: t for t in act1}
        act1_map["interact"].invoke({})

        # ts2 should still have no action
        self.assertIsNone(ts2.chosen_action)


class TestObservationToolsReadFromToolState(unittest.TestCase):
    """Observation tools should read from their bound ToolState, not globals."""

    def _make_mock_mdp(self):
        mdp = MagicMock()
        mdp.width = 5
        mdp.height = 5
        mdp.terrain_mtx = [
            [" ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " "],
        ]
        return mdp

    def _make_mock_state(self, agent_index=0):
        state = MagicMock()

        player0 = MagicMock()
        player0.position = (2, 2)
        player0.pos_and_or = ((2, 2), Direction.NORTH)

        player1 = MagicMock()
        player1.position = (4, 4)
        player1.pos_and_or = ((4, 4), Direction.SOUTH)

        state.players = [player0, player1]
        state.has_object = MagicMock(return_value=False)
        return state

    def test_get_surroundings_uses_tool_state(self):
        ts = ToolState()
        ts.mdp = self._make_mock_mdp()
        ts.state = self._make_mock_state()
        ts.agent_index = 0

        obs_tools, _, _ = create_worker_tools(ts)
        tool_map = {t.name: t for t in obs_tools}

        result = tool_map["get_surroundings"].invoke({})
        self.assertIn("up:", result)
        self.assertIn("down:", result)
        self.assertIn("left:", result)
        self.assertIn("right:", result)
        self.assertIn("floor", result)

    def test_get_surroundings_different_states(self):
        """Two tool sets with different states produce different outputs."""
        ts1 = ToolState()
        ts1.agent_index = 0

        mdp1 = self._make_mock_mdp()
        mdp1.terrain_mtx[1][2] = "P"  # pot above player at (2,2)
        ts1.mdp = mdp1
        ts1.state = self._make_mock_state()

        ts2 = ToolState()
        ts2.agent_index = 0

        mdp2 = self._make_mock_mdp()
        mdp2.terrain_mtx[1][2] = "D"  # dish dispenser above player
        ts2.mdp = mdp2
        ts2.state = self._make_mock_state()

        obs1, _, _ = create_worker_tools(ts1)
        obs2, _, _ = create_worker_tools(ts2)

        result1 = {t.name: t for t in obs1}["get_surroundings"].invoke({})
        result2 = {t.name: t for t in obs2}["get_surroundings"].invoke({})

        self.assertIn("pot", result1)
        self.assertNotIn("dish_dispenser", result1)
        self.assertIn("dish_dispenser", result2)
        self.assertNotIn("pot", result2)

    def test_get_pot_details_uses_tool_state(self):
        ts = ToolState()
        ts.mdp = self._make_mock_mdp()
        ts.mdp.get_pot_locations.return_value = [(1, 1)]
        ts.mdp.get_pot_states.return_value = {"empty": [(1, 1)]}
        ts.state = self._make_mock_state()
        ts.agent_index = 0

        obs_tools, _, _ = create_worker_tools(ts)
        tool_map = {t.name: t for t in obs_tools}

        result = tool_map["get_pot_details"].invoke({})
        self.assertIn("Pot at (1, 1): empty", result)

    def test_check_path_uses_tool_state(self):
        ts = ToolState()
        ts.mdp = self._make_mock_mdp()
        ts.mdp.get_pot_locations.return_value = [(1, 1)]
        ts.state = self._make_mock_state()
        ts.agent_index = 0

        mp = MagicMock()
        mp.motion_goals_for_pos = {(1, 1): [((1, 0), Direction.SOUTH)]}
        mp.is_valid_motion_start_goal_pair.return_value = True
        mp.get_gridworld_distance.return_value = 3
        ts.motion_planner = mp

        obs_tools, _, _ = create_worker_tools(ts)
        tool_map = {t.name: t for t in obs_tools}

        result = tool_map["check_path"].invoke({"target": "pot"})
        self.assertIn("Nearest pot is at (1, 1), 3 steps away", result)

    def test_check_path_unknown_target(self):
        ts = ToolState()
        ts.mdp = self._make_mock_mdp()
        ts.state = self._make_mock_state()
        ts.agent_index = 0
        ts.motion_planner = MagicMock()

        obs_tools, _, _ = create_worker_tools(ts)
        tool_map = {t.name: t for t in obs_tools}

        result = tool_map["check_path"].invoke({"target": "banana"})
        self.assertIn("Unknown target", result)


if __name__ == "__main__":
    unittest.main()
