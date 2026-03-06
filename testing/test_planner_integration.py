"""Integration tests for planner with mocked graph behavior."""

import json
import unittest
from unittest.mock import Mock, patch

from overcooked_ai_py.agents.llm.planner import Planner
from overcooked_ai_py.agents.llm.tool_state import ToolState
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner


class _AssigningGraph:
    """Mock graph that deterministically assigns tasks via the real tool."""

    def __init__(self, action_tools):
        self._assign_tasks = next(t for t in action_tools if t.name == "assign_tasks")
        self.invoke_calls = 0

    def invoke(self, payload, config=None):
        self.invoke_calls += 1
        self._assign_tasks.invoke(
            {
                "assignments": json.dumps(
                    {"worker_0": "Pick up onion", "worker_1": "Get dish"}
                )
            }
        )
        return payload


class TestPlannerIntegration(unittest.TestCase):
    """Integration tests for planner with simulated task assignment flow."""

    def setUp(self):
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.motion_planner = MotionPlanner(self.mdp)
        self.state = self.mdp.get_standard_start_state()

        self.mock_obs = Mock()
        self.mock_obs.emit = Mock()
        self.mock_obs.start_role = Mock()
        self.mock_obs.end_role = Mock()

    def _make_worker_state(self, agent_index: int) -> ToolState:
        ts = ToolState()
        ts.init(self.mdp, self.motion_planner)
        ts.set_state(self.state, agent_index)
        return ts

    @patch("overcooked_ai_py.agents.llm.planner.build_react_graph")
    def test_planner_initialization(self, mock_build_graph):
        mock_build_graph.return_value = Mock()

        planner = Planner(model_name="gpt-4o-mini", observability=self.mock_obs)
        planner.register_worker("worker_0", self._make_worker_state(0))
        planner.register_worker("worker_1", self._make_worker_state(1))
        planner.init(self.mdp, self.motion_planner)

        self.assertIsNotNone(planner)
        self.assertIsNotNone(planner._graph)
        self.assertEqual(planner._last_plan_step, -1)

    @patch("overcooked_ai_py.agents.llm.planner.build_react_graph")
    def test_planner_can_assign_tasks(self, mock_build_graph):
        captured = {}

        def _build_graph(**kwargs):
            graph = _AssigningGraph(kwargs["action_tools"])
            captured["graph"] = graph
            return graph

        mock_build_graph.side_effect = _build_graph

        planner = Planner(model_name="gpt-4o-mini", observability=self.mock_obs)
        worker_0_state = self._make_worker_state(0)
        worker_1_state = self._make_worker_state(1)
        planner.register_worker("worker_0", worker_0_state)
        planner.register_worker("worker_1", worker_1_state)
        planner.init(self.mdp, self.motion_planner)

        planner.maybe_replan(self.state)

        self.assertEqual(captured["graph"].invoke_calls, 1)
        self.assertIsNotNone(worker_0_state.current_task)
        self.assertEqual(worker_0_state.current_task.description, "Pick up onion")
        self.assertIsNotNone(worker_1_state.current_task)
        self.assertEqual(worker_1_state.current_task.description, "Get dish")
        self.assertEqual(planner._last_plan_step, self.state.timestep)

        emitted_events = [call.args[0] for call in self.mock_obs.emit.call_args_list]
        self.assertIn("planner.assignment", emitted_events)


if __name__ == "__main__":
    unittest.main()
