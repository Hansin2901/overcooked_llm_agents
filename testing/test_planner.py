"""Unit tests for Planner class."""

import unittest
from unittest.mock import Mock, patch, MagicMock

from overcooked_ai_py.agents.llm.planner import Planner
from overcooked_ai_py.agents.llm.task import Task
from overcooked_ai_py.agents.llm.tool_state import ToolState
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState


class TestPlannerInit(unittest.TestCase):
    """Test Planner initialization and configuration."""

    def test_init_default_params(self):
        """Test Planner initializes with default parameters."""
        planner = Planner()
        self.assertEqual(planner.model_name, "gpt-4o")
        self.assertEqual(planner.replan_interval, 5)
        self.assertFalse(planner.debug)
        self.assertIsNone(planner.horizon)
        self.assertIsNone(planner.api_base)
        self.assertIsNone(planner.api_key)
        self.assertIsInstance(planner._tool_state, ToolState)
        self.assertEqual(planner._last_plan_step, -1)
        self.assertEqual(planner._worker_registry, {})

    def test_init_custom_params(self):
        """Test Planner initializes with custom parameters."""
        planner = Planner(
            model_name="anthropic/claude-sonnet-4-20250514",
            replan_interval=10,
            debug=True,
            horizon=200,
            api_base="https://custom.api/v1",
            api_key="test-key",
        )
        self.assertEqual(planner.model_name, "anthropic/claude-sonnet-4-20250514")
        self.assertEqual(planner.replan_interval, 10)
        self.assertTrue(planner.debug)
        self.assertEqual(planner.horizon, 200)
        self.assertEqual(planner.api_base, "https://custom.api/v1")
        self.assertEqual(planner.api_key, "test-key")


class TestWorkerRegistration(unittest.TestCase):
    """Test worker registration functionality."""

    def setUp(self):
        self.planner = Planner()

    def test_register_single_worker(self):
        """Test registering a single worker."""
        worker_ts = ToolState()
        self.planner.register_worker("worker_0", worker_ts)
        self.assertIn("worker_0", self.planner._worker_registry)
        self.assertIs(self.planner._worker_registry["worker_0"], worker_ts)

    def test_register_multiple_workers(self):
        """Test registering multiple workers."""
        worker_ts_0 = ToolState()
        worker_ts_1 = ToolState()
        self.planner.register_worker("worker_0", worker_ts_0)
        self.planner.register_worker("worker_1", worker_ts_1)
        self.assertEqual(len(self.planner._worker_registry), 2)
        self.assertIs(self.planner._worker_registry["worker_0"], worker_ts_0)
        self.assertIs(self.planner._worker_registry["worker_1"], worker_ts_1)


class TestPlannerInitMethod(unittest.TestCase):
    """Test Planner.init() method."""

    def setUp(self):
        self.planner = Planner(horizon=200)
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")

        # Create mock motion planner
        self.motion_planner = Mock()

    @patch("overcooked_ai_py.agents.llm.planner.build_planner_system_prompt")
    @patch("overcooked_ai_py.agents.llm.planner.create_planner_tools")
    @patch("overcooked_ai_py.agents.llm.planner.build_react_graph")
    def test_init_sets_up_graph(self, mock_graph, mock_tools, mock_prompt):
        """Test that init() sets up the graph correctly."""
        # Setup mocks
        mock_prompt.return_value = "test system prompt"
        mock_tools.return_value = (["obs"], ["act"], {"assign_tasks"})
        mock_graph.return_value = Mock()

        # Register workers
        worker_ts_0 = ToolState()
        worker_ts_1 = ToolState()
        self.planner.register_worker("worker_0", worker_ts_0)
        self.planner.register_worker("worker_1", worker_ts_1)

        # Call init
        self.planner.init(self.mdp, self.motion_planner)

        # Verify calls
        mock_prompt.assert_called_once_with(
            self.mdp, ["worker_0", "worker_1"], 200
        )
        mock_tools.assert_called_once()
        mock_graph.assert_called_once()

        # Verify state
        self.assertIsNotNone(self.planner._system_prompt)
        self.assertIsNotNone(self.planner._graph)


class TestReplanLogic(unittest.TestCase):
    """Test replanning decision logic."""

    def setUp(self):
        self.planner = Planner(replan_interval=5)
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")

        # Create initial state
        self.state = self.mdp.get_standard_start_state()

        # Register workers
        self.worker_ts_0 = ToolState()
        self.worker_ts_1 = ToolState()
        self.planner.register_worker("worker_0", self.worker_ts_0)
        self.planner.register_worker("worker_1", self.worker_ts_1)

    def test_should_replan_first_step(self):
        """Test that planner should replan on first step."""
        self.assertTrue(self.planner.should_replan(self.state))

    def test_should_not_replan_same_step_twice(self):
        """Test that planner doesn't replan twice in same timestep."""
        self.planner._last_plan_step = self.state.timestep
        self.assertFalse(self.planner.should_replan(self.state))

    def test_should_replan_after_interval(self):
        """Test that planner replans after interval elapses."""
        self.planner._last_plan_step = 0
        self.state.timestep = 5  # Exactly at interval
        self.assertTrue(self.planner.should_replan(self.state))

    def test_should_not_replan_before_interval(self):
        """Test that planner doesn't replan before interval."""
        self.planner._last_plan_step = 0
        self.state.timestep = 3  # Before interval

        # Give workers tasks so they're not idle
        self.worker_ts_0.current_task = Task(
            description="test", worker_id="worker_0", created_at=0
        )
        self.worker_ts_1.current_task = Task(
            description="test", worker_id="worker_1", created_at=0
        )

        self.assertFalse(self.planner.should_replan(self.state))

    def test_should_replan_when_worker_has_no_task(self):
        """Test that planner replans when a worker has no task."""
        self.planner._last_plan_step = 0
        self.state.timestep = 2  # Before interval

        # One worker has task, one doesn't
        self.worker_ts_0.current_task = Task(
            description="test", worker_id="worker_0", created_at=0
        )
        self.worker_ts_1.current_task = None

        self.assertTrue(self.planner.should_replan(self.state))

    def test_should_replan_when_worker_task_completed(self):
        """Test that planner replans when a worker completes their task."""
        self.planner._last_plan_step = 0
        self.state.timestep = 2  # Before interval

        # One worker has incomplete task, one has completed
        self.worker_ts_0.current_task = Task(
            description="test", worker_id="worker_0", created_at=0
        )
        self.worker_ts_1.current_task = Task(
            description="test", worker_id="worker_1", created_at=0, completed=True
        )

        self.assertTrue(self.planner.should_replan(self.state))


class TestTasksAssigned(unittest.TestCase):
    """Test _tasks_assigned termination condition."""

    def setUp(self):
        self.planner = Planner()
        self.worker_ts_0 = ToolState()
        self.worker_ts_1 = ToolState()
        self.planner.register_worker("worker_0", self.worker_ts_0)
        self.planner.register_worker("worker_1", self.worker_ts_1)

    def test_tasks_assigned_fresh_task(self):
        """Test that _tasks_assigned returns True when fresh tasks exist."""
        # Create fresh task (steps_active == 0)
        self.worker_ts_0.current_task = Task(
            description="test", worker_id="worker_0", created_at=0, steps_active=0
        )

        result = self.planner._tasks_assigned()
        self.assertTrue(result)

    def test_tasks_assigned_old_task(self):
        """Test that _tasks_assigned returns None when tasks are old."""
        # Create old task (steps_active > 0)
        self.worker_ts_0.current_task = Task(
            description="test", worker_id="worker_0", created_at=0, steps_active=5
        )
        self.worker_ts_1.current_task = Task(
            description="test", worker_id="worker_1", created_at=0, steps_active=3
        )

        result = self.planner._tasks_assigned()
        self.assertIsNone(result)

    def test_tasks_assigned_no_tasks(self):
        """Test that _tasks_assigned returns None when no tasks exist."""
        self.worker_ts_0.current_task = None
        self.worker_ts_1.current_task = None

        result = self.planner._tasks_assigned()
        self.assertIsNone(result)


class TestGetTask(unittest.TestCase):
    """Test get_task method."""

    def setUp(self):
        self.planner = Planner()
        self.worker_ts_0 = ToolState()
        self.worker_ts_1 = ToolState()
        self.planner.register_worker("worker_0", self.worker_ts_0)
        self.planner.register_worker("worker_1", self.worker_ts_1)

    def test_get_task_with_task(self):
        """Test getting a worker's task when they have one."""
        task = Task(description="test task", worker_id="worker_0", created_at=0)
        self.worker_ts_0.current_task = task

        result = self.planner.get_task("worker_0")
        self.assertIs(result, task)

    def test_get_task_without_task(self):
        """Test getting a worker's task when they don't have one."""
        self.worker_ts_0.current_task = None

        result = self.planner.get_task("worker_0")
        self.assertIsNone(result)

    def test_get_task_unknown_worker(self):
        """Test getting task for unknown worker."""
        result = self.planner.get_task("unknown_worker")
        self.assertIsNone(result)


class TestReset(unittest.TestCase):
    """Test reset method."""

    def setUp(self):
        self.planner = Planner()
        self.worker_ts_0 = ToolState()
        self.planner.register_worker("worker_0", self.worker_ts_0)

    @patch("overcooked_ai_py.agents.llm.planner.build_react_graph")
    def test_reset_clears_state(self, mock_graph):
        """Test that reset clears episode state but preserves worker registry.

        Worker registry must persist across episodes because workers are
        registered once during setup and reused across episodes.
        """
        # Set up some state
        mock_graph.return_value = Mock()
        self.planner._last_plan_step = 10
        self.planner._graph = mock_graph.return_value

        # Reset
        self.planner.reset()

        # Verify episode state cleared
        self.assertEqual(self.planner._last_plan_step, -1)
        self.assertIsNone(self.planner._graph)

        # Verify worker registry PERSISTS (not cleared)
        self.assertIn("worker_0", self.planner._worker_registry)
        self.assertIs(self.planner._worker_registry["worker_0"], self.worker_ts_0)


class TestMaybeReplan(unittest.TestCase):
    """Test maybe_replan integration."""

    def setUp(self):
        self.planner = Planner(replan_interval=5, horizon=200)
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.state = self.mdp.get_standard_start_state()

        # Register workers
        self.worker_ts_0 = ToolState()
        self.worker_ts_1 = ToolState()
        self.planner.register_worker("worker_0", self.worker_ts_0)
        self.planner.register_worker("worker_1", self.worker_ts_1)

    @patch("overcooked_ai_py.agents.llm.planner.serialize_state")
    def test_maybe_replan_skips_when_not_needed(self, mock_serialize):
        """Test that maybe_replan skips when not needed."""
        # Already planned this step
        self.planner._last_plan_step = self.state.timestep

        self.planner.maybe_replan(self.state)

        # Should not call serialize_state
        mock_serialize.assert_not_called()

    @patch("overcooked_ai_py.agents.llm.planner.serialize_state")
    @patch("overcooked_ai_py.agents.llm.planner.build_react_graph")
    @patch("overcooked_ai_py.agents.llm.planner.create_planner_tools")
    @patch("overcooked_ai_py.agents.llm.planner.build_planner_system_prompt")
    def test_maybe_replan_runs_when_needed(
        self, mock_prompt, mock_tools, mock_graph, mock_serialize
    ):
        """Test that maybe_replan runs when needed."""
        # Setup mocks
        mock_prompt.return_value = "test prompt"
        mock_tools.return_value = ([], [], set())
        mock_graph_instance = Mock()
        mock_graph.return_value = mock_graph_instance
        mock_serialize.return_value = "test state"

        # Initialize planner
        motion_planner = Mock()
        self.planner.init(self.mdp, motion_planner)

        # First call should trigger replan
        self.planner.maybe_replan(self.state)

        # Verify graph was invoked
        mock_graph_instance.invoke.assert_called_once()
        self.assertEqual(self.planner._last_plan_step, self.state.timestep)


class TestPlannerObservability(unittest.TestCase):
    def setUp(self):
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")

    def test_planner_emits_assignment_event(self):
        sink = Mock()
        state = self.mdp.get_standard_start_state()
        planner = Planner(
            model_name="gpt-4o",
            observability=sink,
            invoke_config={"callbacks": ["dummy-callback"]},
        )
        planner.register_worker("worker_0", ToolState())
        planner._system_prompt = "test planner prompt"
        planner._tool_state.mdp = self.mdp
        planner._graph = Mock()
        planner._graph.invoke.return_value = {"messages": []}

        planner.maybe_replan(state)
        sink.emit.assert_any_call(
            "planner.assignment",
            unittest.mock.ANY,
            step=state.timestep,
            agent_role="planner",
        )


if __name__ == "__main__":
    unittest.main()
