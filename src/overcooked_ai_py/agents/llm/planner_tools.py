"""Factory for creating planner action tools.

The planner has a single responsibility: assign tasks to workers.
Workers are responsible for their own observations.
"""

import json

from langchain_core.tools import tool

from overcooked_ai_py.agents.llm.task import Task
from overcooked_ai_py.agents.llm.tool_state import ToolState


def create_planner_tools(
    planner_tool_state: ToolState,
    worker_registry: dict[str, ToolState],
) -> tuple:
    """Create planner tools.

    Args:
        planner_tool_state: Planner's own ToolState (for observation tools)
        worker_registry: Maps worker_id -> worker's ToolState.
            Workers never see this dict -- only the planner does.

    Returns: (observation_tools, action_tools, action_tool_names)
    """

    # -------------------------------------------------------------------
    # Observation Tools
    # -------------------------------------------------------------------

    # Observation tools removed in Task 1: Planner now only assigns tasks.
    # Workers are responsible for their own observations.

    # -------------------------------------------------------------------
    # Action Tools (termination)
    # -------------------------------------------------------------------

    @tool
    def assign_tasks(assignments: str) -> str:
        """Assign tasks to workers. This ends your planning turn.

        Args:
            assignments: A JSON string mapping worker_id to task description.
                Example: '{"worker_0": "Pick up onion", "worker_1": "Get a dish"}'
        """
        try:
            parsed = json.loads(assignments)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON — {e}"

        if not isinstance(parsed, dict):
            return "Error: Assignments must be a JSON object mapping worker_id to task description."

        errors = []
        assigned = []
        for worker_id, description in parsed.items():
            if worker_id not in worker_registry:
                errors.append(f"Unknown worker_id '{worker_id}'")
                continue
            if not isinstance(description, str):
                errors.append(
                    f"Task for '{worker_id}' must be a string, got {type(description).__name__}"
                )
                continue

            timestep = 0
            if planner_tool_state.state is not None:
                timestep = getattr(planner_tool_state.state, "timestep", 0)

            task = Task(
                description=description,
                worker_id=worker_id,
                created_at=timestep,
            )
            worker_registry[worker_id].set_task(task)
            assigned.append(f"{worker_id}: {description}")

        result_parts = []
        if assigned:
            result_parts.append("Assigned: " + "; ".join(assigned))
        if errors:
            result_parts.append("Errors: " + "; ".join(errors))
        return ". ".join(result_parts) if result_parts else "No assignments made."

    # -------------------------------------------------------------------
    # Collect and return
    # -------------------------------------------------------------------

    observation_tools = []
    action_tools = [assign_tasks]
    action_tool_names = {"assign_tasks"}
    return observation_tools, action_tools, action_tool_names
