"""Unit tests for WorkerAgent class without LLM calls."""

import unittest
from unittest.mock import Mock, MagicMock

from overcooked_ai_py.agents.llm.planner import Planner
from overcooked_ai_py.agents.llm.task import Task
from overcooked_ai_py.agents.llm.worker_agent import WorkerAgent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner


class TestWorkerAgentUnit(unittest.TestCase):
    """Unit tests for WorkerAgent without LLM dependencies."""

    def setUp(self):
        """Set up test environment."""
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=400)

        # Create a shared planner
        self.planner = Planner(
            model_name="gpt-4o-mini",
            replan_interval=5,
            debug=False,
            horizon=400
        )

        # Create worker
        self.worker = WorkerAgent(
            planner=self.planner,
            worker_id="worker_0",
            model_name="gpt-4o-mini",
            debug=False,
            horizon=400
        )

    def test_init(self):
        """Test WorkerAgent initialization."""
        self.assertEqual(self.worker.planner, self.planner)
        self.assertEqual(self.worker.worker_id, "worker_0")
        self.assertEqual(self.worker.model_name, "gpt-4o-mini")
        self.assertFalse(self.worker.debug)
        self.assertEqual(self.worker.horizon, 400)

        # Check internal state
        self.assertIsNotNone(self.worker._tool_state)
        self.assertIsNone(self.worker._graph)
        self.assertIsNone(self.worker._system_prompt)

        # Check base Agent initialization
        self.assertIsNone(self.worker.agent_index)
        self.assertIsNone(self.worker.mdp)

    def test_set_mdp(self):
        """Test set_mdp registers worker and builds graph."""
        # Set agent index
        self.worker.set_agent_index(0)

        # Set MDP
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Check worker is registered
        self.assertIn("worker_0", self.planner._worker_registry)

        # Check tool state
        self.assertEqual(self.worker._tool_state.mdp, self.mdp)
        self.assertIsNotNone(self.worker._tool_state.motion_planner)

        # Check graph and system prompt
        self.assertIsNotNone(self.worker._graph)
        self.assertIsNotNone(self.worker._system_prompt)
        self.assertIn("worker_0", self.worker._system_prompt)

    def test_action_with_mocked_llm(self):
        """Test action() logic with mocked LLM graph."""
        # Set up worker
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock the graphs to avoid LLM calls
        self.planner._graph = Mock()
        self.worker._graph = Mock()

        # Mock the worker graph to set an action
        def mock_worker_invoke(messages):
            # Simulate choosing NORTH action
            self.worker._tool_state.set_action(Direction.NORTH)

        self.worker._graph.invoke = mock_worker_invoke

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Get action
        action, info = self.worker.action(state)

        # Check action
        self.assertEqual(action, Direction.NORTH)
        self.assertIn("action_probs", info)

        # Check planner was called
        self.planner._graph.invoke.assert_called_once()

    def test_action_defaults_to_stay(self):
        """Test that action defaults to STAY when graph doesn't set action."""
        # Set up worker
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock the graphs
        self.planner._graph = Mock()
        self.worker._graph = Mock()

        # Mock worker graph to NOT set any action
        def mock_worker_invoke(messages):
            # Don't set any action
            pass

        self.worker._graph.invoke = mock_worker_invoke

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Get action
        action, info = self.worker.action(state)

        # Check defaults to STAY
        self.assertEqual(action, Action.STAY)

    def test_task_increments(self):
        """Test that task steps_active increments."""
        # Set up worker
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock graphs
        self.planner._graph = Mock()
        self.worker._graph = Mock()

        # Create and assign a task
        task = Task(
            description="Test task",
            worker_id="worker_0",
            created_at=0,
            completed=False,
            steps_active=0
        )
        self.worker._tool_state.set_task(task)

        # Mock get_task to return the task
        self.planner.get_task = Mock(return_value=task)

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Get action
        initial_steps = task.steps_active
        self.worker.action(state)

        # Check steps incremented
        self.assertEqual(task.steps_active, initial_steps + 1)

    def test_planner_called_once_per_timestep(self):
        """Test that planner is only triggered once per timestep."""
        # Create two workers
        worker1 = WorkerAgent(
            planner=self.planner,
            worker_id="worker_1",
            model_name="gpt-4o-mini",
            debug=False,
            horizon=400
        )

        # Set up workers
        self.worker.set_agent_index(0)
        worker1.set_agent_index(1)
        self.worker.set_mdp(self.mdp)
        worker1.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock graphs
        self.planner._graph = Mock()
        self.worker._graph = Mock()
        worker1._graph = Mock()

        # Reset environment
        self.env.reset()
        state = self.env.state

        # First worker calls action
        self.worker.action(state)
        first_call_count = self.planner._graph.invoke.call_count

        # Second worker calls action (same timestep)
        worker1.action(state)
        second_call_count = self.planner._graph.invoke.call_count

        # Planner should only be called once
        self.assertEqual(first_call_count, 1)
        self.assertEqual(second_call_count, 1)

    def test_reset(self):
        """Test that reset() clears state."""
        # Set up worker
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Set some state
        task = Task(
            description="Test",
            worker_id="worker_0",
            created_at=0,
        )
        self.worker._tool_state.set_task(task)

        # Reset
        self.worker.reset()

        # Check state cleared
        self.assertIsNone(self.worker.agent_index)
        self.assertIsNone(self.worker.mdp)
        self.assertIsNone(self.worker._graph)
        self.assertIsNone(self.worker._system_prompt)
        self.assertIsNone(self.worker._tool_state.state)
        self.assertIsNone(self.worker._tool_state.current_task)

    def test_custom_config(self):
        """Test custom API configuration."""
        worker = WorkerAgent(
            planner=self.planner,
            worker_id="test",
            model_name="openai/custom",
            debug=True,
            horizon=200,
            api_base="https://test.com",
            api_key="key123"
        )

        self.assertEqual(worker.model_name, "openai/custom")
        self.assertTrue(worker.debug)
        self.assertEqual(worker.horizon, 200)
        self.assertEqual(worker.api_base, "https://test.com")
        self.assertEqual(worker.api_key, "key123")

    def test_action_info_structure(self):
        """Test that action info has correct structure."""
        # Set up worker
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock graphs
        self.planner._graph = Mock()
        self.worker._graph = Mock()

        # Mock to set an action
        def mock_invoke(messages):
            self.worker._tool_state.set_action(Action.INTERACT)

        self.worker._graph.invoke = mock_invoke

        # Get action
        self.env.reset()
        state = self.env.state
        action, info = self.worker.action(state)

        # Check info structure
        self.assertIn("action_probs", info)
        self.assertEqual(len(info["action_probs"]), Action.NUM_ACTIONS)
        self.assertAlmostEqual(sum(info["action_probs"]), 1.0, places=4)

    def test_debug_output(self):
        """Test that debug mode prints output."""
        # Create debug worker
        debug_worker = WorkerAgent(
            planner=self.planner,
            worker_id="debug_test",
            model_name="gpt-4o-mini",
            debug=True,
            horizon=400
        )

        debug_worker.set_agent_index(0)
        debug_worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock graphs
        self.planner._graph = Mock()
        debug_worker._graph = Mock()

        # Get action (should print debug output)
        self.env.reset()
        state = self.env.state

        # Just verify it doesn't crash with debug=True
        action, info = debug_worker.action(state)
        self.assertIn(action, Action.ALL_ACTIONS)

    def test_worker_emits_action_commit(self):
        """Test worker emits action.commit event when selecting an action."""
        sink = Mock()
        state = self.mdp.get_standard_start_state()

        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)
        self.worker.observability = sink

        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        self.planner._graph = Mock()
        self.worker._graph = Mock()
        self.worker._tool_state.set_action(Action.STAY)

        self.worker.action(state)
        sink.start_role.assert_called_once_with("worker_0")
        sink.end_role.assert_called_once()
        sink.emit.assert_any_call(
            "action.commit",
            unittest.mock.ANY,
            step=state.timestep,
            agent_role="worker_0",
        )


if __name__ == "__main__":
    unittest.main()
