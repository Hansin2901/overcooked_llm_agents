from typing import Optional
from overcooked_ai_py.agents.llm.task import Task


class ToolState:
    """Encapsulates tool context for one agent (worker or planner).

    Each worker has its own ToolState. Workers cannot access each other's.
    """
    def __init__(self):
        self.mdp = None
        self.state = None
        self.agent_index = None
        self.motion_planner = None
        self.chosen_action = None
        self.current_task: Optional[Task] = None

    def init(self, mdp, motion_planner):
        self.mdp = mdp
        self.motion_planner = motion_planner

    def set_state(self, state, agent_index):
        self.state = state
        self.agent_index = agent_index
        self.chosen_action = None  # Reset each step

    def set_action(self, action):
        self.chosen_action = action

    def set_task(self, task: Task):
        self.current_task = task

    def get_status(self) -> dict:
        """Return status for planner to query."""
        if self.current_task is None:
            return {"status": "idle", "task": None}
        return {
            "status": "completed" if self.current_task.completed else "working",
            "task": self.current_task.description,
            "steps_active": self.current_task.steps_active,
        }

    def reset(self):
        self.state = None
        self.agent_index = None
        self.chosen_action = None
        self.current_task = None
