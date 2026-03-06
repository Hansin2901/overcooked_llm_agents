"""Tests for ToolState class."""

import unittest
from unittest.mock import MagicMock

from overcooked_ai_py.agents.llm.tool_state import ToolState
from overcooked_ai_py.agents.llm.task import Task


class TestToolStateInit(unittest.TestCase):
    """Test __init__ defaults."""

    def test_defaults_are_none(self):
        ts = ToolState()
        self.assertIsNone(ts.mdp)
        self.assertIsNone(ts.state)
        self.assertIsNone(ts.agent_index)
        self.assertIsNone(ts.motion_planner)
        self.assertIsNone(ts.chosen_action)
        self.assertIsNone(ts.current_task)


class TestToolStateInitMethod(unittest.TestCase):
    """Test init() stores mdp and motion_planner."""

    def test_init_stores_references(self):
        ts = ToolState()
        mdp = MagicMock(name="mdp")
        mp = MagicMock(name="motion_planner")
        ts.init(mdp, mp)
        self.assertIs(ts.mdp, mdp)
        self.assertIs(ts.motion_planner, mp)


class TestToolStateSetState(unittest.TestCase):
    """Test set_state() stores state and agent_index, resets chosen_action."""

    def test_set_state_stores_values(self):
        ts = ToolState()
        state = MagicMock(name="state")
        ts.set_state(state, 0)
        self.assertIs(ts.state, state)
        self.assertEqual(ts.agent_index, 0)

    def test_set_state_resets_chosen_action(self):
        ts = ToolState()
        ts.chosen_action = "some_action"
        ts.set_state(MagicMock(), 1)
        self.assertIsNone(ts.chosen_action)


class TestToolStateSetAction(unittest.TestCase):
    """Test set_action() stores action."""

    def test_set_action(self):
        ts = ToolState()
        ts.set_action("interact")
        self.assertEqual(ts.chosen_action, "interact")


class TestToolStateSetTask(unittest.TestCase):
    """Test set_task() stores task."""

    def test_set_task(self):
        ts = ToolState()
        task = Task(description="pick up onion", worker_id="w0", created_at=5)
        ts.set_task(task)
        self.assertIs(ts.current_task, task)


class TestToolStateGetStatus(unittest.TestCase):
    """Test get_status() returns correct dict for idle, working, completed."""

    def test_idle_when_no_task(self):
        ts = ToolState()
        status = ts.get_status()
        self.assertEqual(status, {"status": "idle", "task": None})

    def test_working_when_task_not_completed(self):
        ts = ToolState()
        task = Task(
            description="deliver soup",
            worker_id="w0",
            created_at=10,
            completed=False,
            steps_active=3,
        )
        ts.set_task(task)
        status = ts.get_status()
        self.assertEqual(status["status"], "working")
        self.assertEqual(status["task"], "deliver soup")
        self.assertEqual(status["steps_active"], 3)

    def test_completed_when_task_completed(self):
        ts = ToolState()
        task = Task(
            description="pick up dish",
            worker_id="w1",
            created_at=0,
            completed=True,
            steps_active=7,
        )
        ts.set_task(task)
        status = ts.get_status()
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["task"], "pick up dish")
        self.assertEqual(status["steps_active"], 7)


class TestToolStateReset(unittest.TestCase):
    """Test reset() clears state but preserves mdp/motion_planner."""

    def test_reset_clears_transient_state(self):
        ts = ToolState()
        mdp = MagicMock(name="mdp")
        mp = MagicMock(name="motion_planner")
        ts.init(mdp, mp)
        ts.set_state(MagicMock(), 0)
        ts.set_action("interact")
        ts.set_task(Task(description="x", worker_id="w0", created_at=0))

        ts.reset()

        self.assertIsNone(ts.state)
        self.assertIsNone(ts.agent_index)
        self.assertIsNone(ts.chosen_action)
        self.assertIsNone(ts.current_task)

    def test_reset_preserves_mdp_and_planner(self):
        ts = ToolState()
        mdp = MagicMock(name="mdp")
        mp = MagicMock(name="motion_planner")
        ts.init(mdp, mp)
        ts.set_state(MagicMock(), 1)

        ts.reset()

        self.assertIs(ts.mdp, mdp)
        self.assertIs(ts.motion_planner, mp)


if __name__ == "__main__":
    unittest.main()
