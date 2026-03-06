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
            model_name="gpt-4o-mini", replan_interval=5, debug=False, horizon=400
        )

        # Create worker
        self.worker = WorkerAgent(
            planner=self.planner,
            worker_id="worker_0",
            model_name="gpt-4o-mini",
            debug=False,
            horizon=400,
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
        self.assertIsNone(self.worker._llm)
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

        # Check llm and system prompt
        self.assertIsNotNone(self.worker._llm)
        self.assertIsNotNone(self.worker._system_prompt)
        self.assertIn("worker_0", self.worker._system_prompt)
        self.assertIn("JSON", self.worker._system_prompt)

    def test_action_with_mocked_llm(self):
        """Test action() logic with mocked LLM."""
        from unittest.mock import patch
        from langchain_core.messages import AIMessage

        # Set up worker
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock the planner graph
        self.planner._graph = Mock()

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Mock LLM to return valid JSON
        mock_response = AIMessage(content='{"action":"move_up"}')
        with patch.object(self.worker, "_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response

            # Get action
            action, info = self.worker.action(state)

            # Check action
            self.assertEqual(action, Direction.NORTH)
            self.assertIn("action_probs", info)

            # Check planner was called
            self.planner._graph.invoke.assert_called_once()

    def test_action_defaults_to_stay(self):
        """Test that action defaults to STAY when LLM returns invalid JSON."""
        from unittest.mock import patch
        from langchain_core.messages import AIMessage

        # Set up worker
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock the planner graph
        self.planner._graph = Mock()

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Mock LLM to return invalid JSON
        mock_response = AIMessage(content="not valid json")
        with patch.object(self.worker, "_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response

            # Get action
            action, info = self.worker.action(state)

            # Check defaults to STAY
            self.assertEqual(action, Action.STAY)

    def test_task_increments(self):
        """Test that task steps_active increments."""
        from unittest.mock import patch
        from langchain_core.messages import AIMessage

        # Set up worker
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock planner graph
        self.planner._graph = Mock()

        # Create and assign a task
        task = Task(
            description="Test task",
            worker_id="worker_0",
            created_at=0,
            completed=False,
            steps_active=0,
        )
        self.worker._tool_state.set_task(task)

        # Mock get_task to return the task
        self.planner.get_task = Mock(return_value=task)

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Mock LLM
        mock_response = AIMessage(content='{"action":"wait"}')
        with patch.object(self.worker, "_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response

            # Get action
            initial_steps = task.steps_active
            self.worker.action(state)

            # Check steps incremented
            self.assertEqual(task.steps_active, initial_steps + 1)

    def test_planner_called_once_per_timestep(self):
        """Test that planner is only triggered once per timestep."""
        from unittest.mock import patch
        from langchain_core.messages import AIMessage

        # Create two workers
        worker1 = WorkerAgent(
            planner=self.planner,
            worker_id="worker_1",
            model_name="gpt-4o-mini",
            debug=False,
            horizon=400,
        )

        # Set up workers
        self.worker.set_agent_index(0)
        worker1.set_agent_index(1)
        self.worker.set_mdp(self.mdp)
        worker1.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock planner graph
        self.planner._graph = Mock()

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Mock LLMs for both workers
        mock_response = AIMessage(content='{"action":"wait"}')
        with (
            patch.object(self.worker, "_llm") as mock_llm0,
            patch.object(worker1, "_llm") as mock_llm1,
        ):
            mock_llm0.invoke.return_value = mock_response
            mock_llm1.invoke.return_value = mock_response

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
        self.assertIsNone(self.worker._llm)
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
            api_key="key123",
        )

        self.assertEqual(worker.model_name, "openai/custom")
        self.assertTrue(worker.debug)
        self.assertEqual(worker.horizon, 200)
        self.assertEqual(worker.api_base, "https://test.com")
        self.assertEqual(worker.api_key, "key123")

    def test_action_info_structure(self):
        """Test that action info has correct structure."""
        from unittest.mock import patch
        from langchain_core.messages import AIMessage

        # Set up worker
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock planner graph
        self.planner._graph = Mock()

        # Get action
        self.env.reset()
        state = self.env.state

        # Mock LLM
        mock_response = AIMessage(content='{"action":"interact"}')
        with patch.object(self.worker, "_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response

            action, info = self.worker.action(state)

            # Check info structure
            self.assertIn("action_probs", info)
            self.assertEqual(len(info["action_probs"]), Action.NUM_ACTIONS)
            self.assertAlmostEqual(sum(info["action_probs"]), 1.0, places=4)

    def test_debug_output(self):
        """Test that debug mode prints output."""
        from unittest.mock import patch
        from langchain_core.messages import AIMessage

        # Create debug worker
        debug_worker = WorkerAgent(
            planner=self.planner,
            worker_id="debug_test",
            model_name="gpt-4o-mini",
            debug=True,
            horizon=400,
        )

        debug_worker.set_agent_index(0)
        debug_worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock planner graph
        self.planner._graph = Mock()

        # Get action (should print debug output)
        self.env.reset()
        state = self.env.state

        # Mock LLM
        mock_response = AIMessage(content='{"action":"wait"}')
        with patch.object(debug_worker, "_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response

            # Just verify it doesn't crash with debug=True
            action, info = debug_worker.action(state)
            self.assertIn(action, Action.ALL_ACTIONS)

    def test_worker_emits_action_commit(self):
        """Test worker emits action.commit event when selecting an action."""
        from unittest.mock import patch, ANY
        from langchain_core.messages import AIMessage

        sink = Mock()
        state = self.mdp.get_standard_start_state()

        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)
        self.worker.observability = sink

        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        self.planner._graph = Mock()

        # Mock LLM
        mock_response = AIMessage(content='{"action":"wait"}')
        with patch.object(self.worker, "_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response

            self.worker.action(state)
            sink.start_role.assert_called_once_with("worker_0")
            sink.end_role.assert_called_once()
            sink.emit.assert_any_call(
                "action.commit",
                ANY,
                step=state.timestep,
                agent_role="worker_0",
            )

    def test_worker_one_shot_action_parsing(self):
        """Test worker action path performs one-shot parse without requiring observation tool loop."""
        from unittest.mock import patch
        from langchain_core.messages import AIMessage

        # Set up worker
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock planner graph
        self.planner._graph = Mock()

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Mock one LLM response with valid JSON action
        mock_response = AIMessage(content='{"action":"move_up"}')
        with patch.object(self.worker, "_llm", create=True) as mock_llm:
            mock_llm.invoke.return_value = mock_response

            # Get action
            action, info = self.worker.action(state)

            # Check action was parsed correctly
            self.assertEqual(action, Direction.NORTH)
            self.assertIn("action_probs", info)

            # Verify only one LLM call was made
            self.assertEqual(mock_llm.invoke.call_count, 1)

    def test_worker_invalid_one_shot_output_falls_back_to_stay(self):
        """Test worker invalid one-shot output falls back safely to STAY."""
        from unittest.mock import patch
        from langchain_core.messages import AIMessage

        # Set up worker
        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        # Initialize planner
        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        # Mock planner graph
        self.planner._graph = Mock()

        # Reset environment
        self.env.reset()
        state = self.env.state

        # Mock malformed JSON output
        mock_response = AIMessage(content="invalid json{")
        with patch.object(self.worker, "_llm", create=True) as mock_llm:
            mock_llm.invoke.return_value = mock_response

            # Get action
            action, info = self.worker.action(state)

            # Check defaults to STAY
            self.assertEqual(action, Action.STAY)
            self.assertIn("action_probs", info)


class TestWorkerMemory(unittest.TestCase):
    """Tests for WorkerAgent memory system."""

    def setUp(self):
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room")
        self.planner = Planner(
            model_name="gpt-4o-mini",
            replan_interval=5,
            debug=False,
            horizon=400,
        )
        self.worker = WorkerAgent(
            planner=self.planner,
            worker_id="worker_0",
            model_name="gpt-4o-mini",
            debug=False,
            horizon=400,
        )

    # --- Storage tests (for Task 2A) ---

    def test_default_history_size(self):
        """WorkerAgent defaults to history_size=5."""
        self.assertEqual(self.worker.history_size, 5)
        self.assertEqual(self.worker._history, [])

    def test_custom_history_size(self):
        """WorkerAgent accepts custom history_size."""
        w = WorkerAgent(
            planner=self.planner,
            worker_id="w",
            model_name="gpt-4o-mini",
            history_size=3,
        )
        self.assertEqual(w.history_size, 3)

    def test_add_to_history(self):
        """_add_to_history stores entry with correct fields."""
        self.worker._add_to_history(
            timestep=5,
            position=(2, 1),
            action=Direction.NORTH,
            held="onion",
            task_description="Pick up onion",
        )
        self.assertEqual(len(self.worker._history), 1)
        entry = self.worker._history[0]
        self.assertEqual(entry["timestep"], 5)
        self.assertEqual(entry["position"], (2, 1))
        self.assertEqual(entry["action"], "↑")
        self.assertEqual(entry["held"], "onion")
        self.assertEqual(entry["task"], "Pick up onion")

    def test_history_trimmed_to_size(self):
        """History is trimmed when it exceeds history_size."""
        self.worker.history_size = 3
        for i in range(5):
            self.worker._add_to_history(
                timestep=i,
                position=(0, 0),
                action=Action.STAY,
                held="nothing",
                task_description="task",
            )
        self.assertEqual(len(self.worker._history), 3)
        self.assertEqual(self.worker._history[0]["timestep"], 2)

    def test_history_disabled_when_size_zero(self):
        """No entries stored when history_size=0."""
        self.worker.history_size = 0
        self.worker._add_to_history(
            timestep=0,
            position=(0, 0),
            action=Action.STAY,
            held="nothing",
            task_description="t",
        )
        self.assertEqual(len(self.worker._history), 0)

    # --- Formatting tests (for Task 2A) ---

    def test_format_history_empty(self):
        """Empty history returns empty string."""
        self.assertEqual(self.worker._format_history(), "")

    def test_format_history_single_entry(self):
        """Single entry formats correctly."""
        self.worker._add_to_history(
            timestep=5,
            position=(2, 1),
            action=Direction.NORTH,
            held="nothing",
            task_description="Pick up onion",
        )
        result = self.worker._format_history()
        self.assertIn("RECENT HISTORY:", result)
        self.assertIn("Step 5", result)
        self.assertIn("(2, 1)", result)
        self.assertIn("Pick up onion", result)

    def test_format_history_with_held_item(self):
        """Entry with held item includes it."""
        self.worker._add_to_history(
            timestep=5,
            position=(2, 1),
            action=Direction.NORTH,
            held="onion",
            task_description="Deliver to pot",
        )
        result = self.worker._format_history()
        self.assertIn("holding onion", result)

    def test_format_history_no_held_item_omitted(self):
        """Entry with 'nothing' held does not say 'holding nothing'."""
        self.worker._add_to_history(
            timestep=5,
            position=(2, 1),
            action=Direction.NORTH,
            held="nothing",
            task_description="Pick up onion",
        )
        result = self.worker._format_history()
        self.assertNotIn("holding", result)

    def test_format_history_task_boundary(self):
        """Task change inserts boundary marker."""
        self.worker._add_to_history(
            timestep=5,
            position=(2, 1),
            action=Direction.NORTH,
            held="nothing",
            task_description="Pick up onion",
        )
        self.worker._add_to_history(
            timestep=6,
            position=(2, 0),
            action=Action.INTERACT,
            held="onion",
            task_description="Deliver to pot",
        )
        result = self.worker._format_history()
        self.assertIn("--- New task ---", result)

    def test_format_history_same_task_no_boundary(self):
        """Same task across entries has no boundary marker."""
        self.worker._add_to_history(
            timestep=5,
            position=(2, 1),
            action=Direction.NORTH,
            held="nothing",
            task_description="Pick up onion",
        )
        self.worker._add_to_history(
            timestep=6,
            position=(2, 0),
            action=Action.INTERACT,
            held="nothing",
            task_description="Pick up onion",
        )
        result = self.worker._format_history()
        self.assertNotIn("--- New task ---", result)

    def test_format_history_disabled(self):
        """history_size=0 returns empty string."""
        self.worker.history_size = 0
        self.assertEqual(self.worker._format_history(), "")

    # --- Integration tests (for Task 2B) ---

    def test_action_records_history(self):
        """action() adds an entry to history."""
        from unittest.mock import patch
        from langchain_core.messages import AIMessage

        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        self.planner._graph = Mock()

        env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        env.reset()
        state = env.state

        # Mock LLM
        mock_response = AIMessage(content='{"action":"move_up"}')
        with patch.object(self.worker, "_llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response

            self.worker.action(state)
            self.assertEqual(len(self.worker._history), 1)
            entry = self.worker._history[0]
            self.assertEqual(entry["timestep"], state.timestep)
            self.assertEqual(entry["action"], "↑")

    def test_action_injects_history_into_prompt(self):
        """action() includes history text in the prompt sent to LLM."""
        from unittest.mock import patch
        from langchain_core.messages import AIMessage

        self.worker.set_agent_index(0)
        self.worker.set_mdp(self.mdp)

        mp = MotionPlanner(self.mdp)
        self.planner.init(self.mdp, mp)

        self.planner._graph = Mock()

        # Seed history
        self.worker._add_to_history(
            timestep=0,
            position=(1, 1),
            action=Direction.NORTH,
            held="nothing",
            task_description="Pick up onion",
        )

        captured_messages = []

        def mock_invoke(messages):
            captured_messages.append(messages)
            return AIMessage(content='{"action":"move_up"}')

        env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        env.reset()
        state = env.state

        with patch.object(self.worker, "_llm") as mock_llm:
            mock_llm.invoke = mock_invoke

            self.worker.action(state)

            # Check the HumanMessage contains history
            self.assertEqual(len(captured_messages), 1)
            messages = captured_messages[0]
            human_msg = messages[1]
            self.assertIn("RECENT HISTORY:", human_msg.content)

    def test_reset_clears_history(self):
        """reset() clears the history list."""
        self.worker._add_to_history(
            timestep=0,
            position=(0, 0),
            action=Action.STAY,
            held="nothing",
            task_description="t",
        )
        self.assertEqual(len(self.worker._history), 1)
        self.worker.reset()
        self.assertEqual(len(self.worker._history), 0)


if __name__ == "__main__":
    unittest.main()
