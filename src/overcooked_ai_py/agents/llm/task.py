from dataclasses import dataclass


@dataclass
class Task:
    """A high-level task assigned by the planner to a specific worker."""
    description: str           # Natural language task description
    worker_id: str             # Which worker this is assigned to
    created_at: int            # Timestep when created
    completed: bool = False    # Whether worker signals completion
    steps_active: int = 0      # Timesteps this task has been active
