"""Unit tests for planner tools."""

import json
import unittest

from overcooked_ai_py.agents.llm.planner_tools import create_planner_tools
from overcooked_ai_py.agents.llm.task import Task
from overcooked_ai_py.agents.llm.tool_state import ToolState
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner


class TestPlannerTools(unittest.TestCase):
    """Test suite for planner tool factory."""

    def setUp(self):
        """Set up test environment with MDP and tool states."""
        # Create a simple test layout
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.motion_planner = MotionPlanner(self.mdp)

        # Create planner tool state
        self.planner_tool_state = ToolState()
        self.planner_tool_state.init(self.mdp, self.motion_planner)

        # Create worker tool states
        self.worker_0_state = ToolState()
        self.worker_0_state.init(self.mdp, self.motion_planner)

        self.worker_1_state = ToolState()
        self.worker_1_state.init(self.mdp, self.motion_planner)

        # Create worker registry
        self.worker_registry = {
            "worker_0": self.worker_0_state,
            "worker_1": self.worker_1_state,
        }

        # Set up initial state
        self.state = self.mdp.get_standard_start_state()
        self.planner_tool_state.set_state(self.state, 0)
        self.worker_0_state.set_state(self.state, 0)
        self.worker_1_state.set_state(self.state, 1)

        # Create tools
        self.obs_tools, self.action_tools, self.action_tool_names = create_planner_tools(
            self.planner_tool_state, self.worker_registry
        )

    def test_factory_creates_correct_tools(self):
        """Test that the factory creates the expected number and types of tools."""
        # Should have 4 observation tools
        self.assertEqual(len(self.obs_tools), 4)

        # Should have 1 action tool
        self.assertEqual(len(self.action_tools), 1)

        # Action tool names should contain assign_tasks
        self.assertEqual(self.action_tool_names, {"assign_tasks"})

        # Check observation tool names
        obs_tool_names = {tool.name for tool in self.obs_tools}
        expected_obs_names = {"get_surroundings", "get_pot_details", "check_path", "get_worker_status"}
        self.assertEqual(obs_tool_names, expected_obs_names)

        # Check action tool name
        action_tool_names = {tool.name for tool in self.action_tools}
        self.assertEqual(action_tool_names, {"assign_tasks"})

    def test_assign_tasks_valid_json(self):
        """Test assign_tasks with valid JSON creates Tasks in correct worker ToolStates."""
        # Find the assign_tasks tool
        assign_tasks = next(t for t in self.action_tools if t.name == "assign_tasks")

        # Create valid assignments
        assignments = json.dumps({
            "worker_0": "Pick up onion from dispenser",
            "worker_1": "Get a dish from dispenser"
        })

        # Call the tool
        result = assign_tasks.invoke({"assignments": assignments})

        # Check result message
        self.assertIn("Assigned", result)
        self.assertIn("worker_0", result)
        self.assertIn("worker_1", result)

        # Verify tasks were created in worker ToolStates
        worker_0_task = self.worker_0_state.current_task
        self.assertIsNotNone(worker_0_task)
        self.assertEqual(worker_0_task.description, "Pick up onion from dispenser")
        self.assertEqual(worker_0_task.worker_id, "worker_0")
        self.assertEqual(worker_0_task.created_at, self.state.timestep)
        self.assertFalse(worker_0_task.completed)

        worker_1_task = self.worker_1_state.current_task
        self.assertIsNotNone(worker_1_task)
        self.assertEqual(worker_1_task.description, "Get a dish from dispenser")
        self.assertEqual(worker_1_task.worker_id, "worker_1")

    def test_assign_tasks_invalid_json(self):
        """Test assign_tasks with invalid JSON returns error."""
        assign_tasks = next(t for t in self.action_tools if t.name == "assign_tasks")

        # Invalid JSON string
        invalid_json = "this is not valid json"
        result = assign_tasks.invoke({"assignments": invalid_json})

        # Should return error message
        self.assertIn("Error", result)
        self.assertIn("Invalid JSON", result)

        # No tasks should be created
        self.assertIsNone(self.worker_0_state.current_task)
        self.assertIsNone(self.worker_1_state.current_task)

    def test_assign_tasks_unknown_worker_id(self):
        """Test assign_tasks with unknown worker_id returns error."""
        assign_tasks = next(t for t in self.action_tools if t.name == "assign_tasks")

        # Create assignments with unknown worker
        assignments = json.dumps({
            "worker_0": "Pick up onion",
            "worker_999": "Unknown worker task"
        })

        result = assign_tasks.invoke({"assignments": assignments})

        # Should indicate error for unknown worker
        self.assertIn("worker_999", result)
        self.assertIn("Unknown", result)

        # worker_0 should still be assigned (partial success)
        self.assertIsNotNone(self.worker_0_state.current_task)
        self.assertEqual(self.worker_0_state.current_task.description, "Pick up onion")

    def test_assign_tasks_non_dict_json(self):
        """Test assign_tasks with non-dict JSON returns error."""
        assign_tasks = next(t for t in self.action_tools if t.name == "assign_tasks")

        # Valid JSON but not a dict
        assignments = json.dumps(["task1", "task2"])
        result = assign_tasks.invoke({"assignments": assignments})

        # Should return error
        self.assertIn("Error", result)
        self.assertIn("object", result.lower())

    def test_assign_tasks_non_string_task(self):
        """Test assign_tasks with non-string task description returns error."""
        assign_tasks = next(t for t in self.action_tools if t.name == "assign_tasks")

        # Task description is not a string
        assignments = json.dumps({
            "worker_0": 123  # number instead of string
        })

        result = assign_tasks.invoke({"assignments": assignments})

        # Should indicate error
        self.assertIn("must be a string", result)

    def test_get_worker_status_idle(self):
        """Test get_worker_status returns correct status for idle worker."""
        # Find the get_worker_status tool
        get_worker_status = next(t for t in self.obs_tools if t.name == "get_worker_status")

        # Worker should be idle initially
        result = get_worker_status.invoke({"worker_id": "worker_0"})

        # Parse the JSON response
        status = json.loads(result)
        self.assertEqual(status["status"], "idle")
        self.assertIsNone(status["task"])

    def test_get_worker_status_working(self):
        """Test get_worker_status returns correct status for working worker."""
        # Assign a task to worker_0
        task = Task(
            description="Pick up onion",
            worker_id="worker_0",
            created_at=0,
            completed=False,
            steps_active=5
        )
        self.worker_0_state.set_task(task)

        # Get status
        get_worker_status = next(t for t in self.obs_tools if t.name == "get_worker_status")
        result = get_worker_status.invoke({"worker_id": "worker_0"})

        # Parse response
        status = json.loads(result)
        self.assertEqual(status["status"], "working")
        self.assertEqual(status["task"], "Pick up onion")
        self.assertEqual(status["steps_active"], 5)

    def test_get_worker_status_completed(self):
        """Test get_worker_status returns correct status for completed task."""
        # Assign a completed task to worker_0
        task = Task(
            description="Deliver soup",
            worker_id="worker_0",
            created_at=0,
            completed=True,
            steps_active=10
        )
        self.worker_0_state.set_task(task)

        # Get status
        get_worker_status = next(t for t in self.obs_tools if t.name == "get_worker_status")
        result = get_worker_status.invoke({"worker_id": "worker_0"})

        # Parse response
        status = json.loads(result)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["task"], "Deliver soup")
        self.assertEqual(status["steps_active"], 10)

    def test_get_worker_status_unknown_worker(self):
        """Test get_worker_status with unknown worker_id returns error."""
        get_worker_status = next(t for t in self.obs_tools if t.name == "get_worker_status")

        result = get_worker_status.invoke({"worker_id": "worker_999"})

        # Should return error message
        self.assertIn("Error", result)
        self.assertIn("Unknown worker_id", result)
        self.assertIn("worker_999", result)

    def test_observation_tools_work(self):
        """Test that observation tools (get_surroundings, get_pot_details, check_path) work."""
        # Test get_surroundings
        get_surroundings = next(t for t in self.obs_tools if t.name == "get_surroundings")
        result = get_surroundings.invoke({})
        self.assertIsInstance(result, str)
        self.assertIn("up:", result)
        self.assertIn("down:", result)

        # Test get_pot_details
        get_pot_details = next(t for t in self.obs_tools if t.name == "get_pot_details")
        result = get_pot_details.invoke({})
        self.assertIsInstance(result, str)

        # Test check_path
        check_path = next(t for t in self.obs_tools if t.name == "check_path")
        result = check_path.invoke({"target": "pot"})
        self.assertIsInstance(result, str)

    def test_multiple_task_assignments(self):
        """Test assigning tasks multiple times overwrites previous tasks."""
        assign_tasks = next(t for t in self.action_tools if t.name == "assign_tasks")

        # First assignment
        assignments1 = json.dumps({"worker_0": "Task 1"})
        assign_tasks.invoke({"assignments": assignments1})
        self.assertEqual(self.worker_0_state.current_task.description, "Task 1")

        # Second assignment (should overwrite)
        assignments2 = json.dumps({"worker_0": "Task 2"})
        assign_tasks.invoke({"assignments": assignments2})
        self.assertEqual(self.worker_0_state.current_task.description, "Task 2")


if __name__ == "__main__":
    unittest.main()
