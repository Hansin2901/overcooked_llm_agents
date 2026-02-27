"""Tests for WorkerAgent class."""

import unittest

from overcooked_ai_py.agents.llm.planner import Planner
from overcooked_ai_py.agents.llm.worker_agent import WorkerAgent
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner


class TestWorkerAgent(unittest.TestCase):
    """Test suite for WorkerAgent class."""

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

        # Create two workers
        self.worker0 = WorkerAgent(
            planner=self.planner,
            worker_id="worker_0",
            model_name="gpt-4o-mini",
            debug=False,
            horizon=400
        )

        self.worker1 = WorkerAgent(
            planner=self.planner,
            worker_id="worker_1",
            model_name="gpt-4o-mini",
            debug=False,
            horizon=400
        )

    def test_init(self):
        """Test WorkerAgent initialization."""
        self.assertEqual(self.worker0.planner, self.planner)
        self.assertEqual(self.worker0.worker_id, "worker_0")
        self.assertEqual(self.worker0.model_name, "gpt-4o-mini")
        self.assertFalse(self.worker0.debug)
        self.assertEqual(self.worker0.horizon, 400)

        # Check that internal state is initialized
        self.assertIsNotNone(self.worker0._tool_state)
        self.assertIsNone(self.worker0._graph)
        self.assertIsNone(self.worker0._system_prompt)

        # Check base Agent initialization
        self.assertIsNone(self.worker0.agent_index)
        self.assertIsNone(self.worker0.mdp)

    def test_set_mdp(self):
        """Test set_mdp registers worker and builds graph."""
        # Set agent indices
        self.worker0.set_agent_index(0)
        self.worker1.set_agent_index(1)

        # Set MDP on both workers
        self.worker0.set_mdp(self.mdp)
        self.worker1.set_mdp(self.mdp)

        # Initialize planner after workers are registered
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Check that workers are registered with planner
        self.assertIn("worker_0", self.planner._worker_registry)
        self.assertIn("worker_1", self.planner._worker_registry)

        # Check that tool state is initialized
        self.assertEqual(self.worker0._tool_state.mdp, self.mdp)
        self.assertIsNotNone(self.worker0._tool_state.motion_planner)

        # Check that graph and system prompt are built
        self.assertIsNotNone(self.worker0._graph)
        self.assertIsNotNone(self.worker0._system_prompt)

        # Check system prompt contains worker ID
        self.assertIn("worker_0", self.worker0._system_prompt)
        self.assertIn("Player 0", self.worker0._system_prompt)

    def test_action_produces_valid_action(self):
        """Test that action() produces a valid game action."""
        # Set up workers
        self.worker0.set_agent_index(0)
        self.worker1.set_agent_index(1)
        self.worker0.set_mdp(self.mdp)
        self.worker1.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Get action from worker
        action, info = self.worker0.action(state)

        # Check that action is valid
        self.assertIn(action, Action.ALL_ACTIONS)

        # Check that info contains action_probs
        self.assertIn("action_probs", info)
        self.assertEqual(len(info["action_probs"]), Action.NUM_ACTIONS)

    def test_action_triggers_planner(self):
        """Test that action() triggers planner on first call."""
        # Set up workers
        self.worker0.set_agent_index(0)
        self.worker1.set_agent_index(1)
        self.worker0.set_mdp(self.mdp)
        self.worker1.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Check that planner hasn't run yet
        self.assertEqual(self.planner._last_plan_step, -1)

        # First worker calls action
        self.worker0.action(state)

        # Check that planner has run
        self.assertEqual(self.planner._last_plan_step, state.timestep)

        # Second worker calls action (same timestep)
        initial_plan_step = self.planner._last_plan_step
        self.worker1.action(state)

        # Check that planner didn't run again (same timestep)
        self.assertEqual(self.planner._last_plan_step, initial_plan_step)

    def test_action_reads_task(self):
        """Test that action() reads worker's task from planner."""
        # Set up workers
        self.worker0.set_agent_index(0)
        self.worker1.set_agent_index(1)
        self.worker0.set_mdp(self.mdp)
        self.worker1.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Worker calls action (this triggers planner and assigns tasks)
        self.worker0.action(state)

        # Check that worker can read its task
        task = self.planner.get_task("worker_0")

        # Task may or may not exist depending on planner LLM
        # But get_task should not raise an error
        self.assertTrue(task is None or hasattr(task, "description"))

    def test_task_steps_active_increments(self):
        """Test that task.steps_active increments each step."""
        # Set up workers
        self.worker0.set_agent_index(0)
        self.worker1.set_agent_index(1)
        self.worker0.set_mdp(self.mdp)
        self.worker1.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Manually assign a task to worker_0
        from overcooked_ai_py.agents.llm.task import Task
        task = Task(
            description="Test task",
            worker_id="worker_0",
            created_at=0,
            completed=False,
            steps_active=0
        )
        self.worker0._tool_state.set_task(task)

        # Get current steps_active
        initial_steps = task.steps_active

        # Worker takes action
        self.worker0.action(state)

        # Check that steps_active incremented
        self.assertEqual(task.steps_active, initial_steps + 1)

    def test_defaults_to_stay_when_no_action(self):
        """Test that worker defaults to STAY if no action chosen."""
        # Set up worker
        self.worker0.set_agent_index(0)
        self.worker0.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Mock the graph to not set any action
        def mock_invoke(messages):
            # Don't set chosen_action
            pass

        original_invoke = self.worker0._graph.invoke
        self.worker0._graph.invoke = mock_invoke

        # Get action
        action, info = self.worker0.action(state)

        # Restore original invoke
        self.worker0._graph.invoke = original_invoke

        # Check that action is STAY
        self.assertEqual(action, Action.STAY)

    def test_reset(self):
        """Test that reset() clears state correctly."""
        # Set up worker
        self.worker0.set_agent_index(0)
        self.worker0.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Reset environment and take an action to populate state
        self.env.reset()
        state = self.env.state
        self.worker0.action(state)

        # Reset worker
        self.worker0.reset()

        # Check that base Agent state is cleared
        self.assertIsNone(self.worker0.agent_index)
        self.assertIsNone(self.worker0.mdp)

        # Check that tool state is cleared
        self.assertIsNone(self.worker0._tool_state.state)
        self.assertIsNone(self.worker0._tool_state.agent_index)
        self.assertIsNone(self.worker0._tool_state.chosen_action)
        self.assertIsNone(self.worker0._tool_state.current_task)

        # Graph should be reset to None
        self.assertIsNone(self.worker0._graph)

    def test_custom_api_config(self):
        """Test WorkerAgent with custom API configuration."""
        worker = WorkerAgent(
            planner=self.planner,
            worker_id="worker_test",
            model_name="openai/test-model",
            debug=True,
            horizon=200,
            api_base="https://test.api.com/v1",
            api_key="test-key-123"
        )

        self.assertEqual(worker.model_name, "openai/test-model")
        self.assertEqual(worker.api_base, "https://test.api.com/v1")
        self.assertEqual(worker.api_key, "test-key-123")
        self.assertTrue(worker.debug)
        self.assertEqual(worker.horizon, 200)

    def test_multiple_timesteps(self):
        """Test worker behavior over multiple timesteps."""
        # Set up workers
        self.worker0.set_agent_index(0)
        self.worker1.set_agent_index(1)
        self.worker0.set_mdp(self.mdp)
        self.worker1.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Run for a few timesteps
        self.env.reset()
        for _ in range(3):
            state = self.env.state

            # Both workers take actions
            action0, _ = self.worker0.action(state)
            action1, _ = self.worker1.action(state)

            # Check both actions are valid
            self.assertIn(action0, Action.ALL_ACTIONS)
            self.assertIn(action1, Action.ALL_ACTIONS)

            # Step environment
            joint_action = (action0, action1)
            self.env.step(joint_action)


if __name__ == "__main__":
    unittest.main()
