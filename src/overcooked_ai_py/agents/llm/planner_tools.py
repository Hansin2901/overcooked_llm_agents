"""Factory for creating planner observation and action tools.

The planner gets tools to assign tasks to workers and query their status,
plus read-only observation tools. The planner receives the worker_registry
(dict of worker_id -> ToolState) -- workers never see this registry.
"""

import json

import numpy as np
from langchain_core.tools import tool

from overcooked_ai_py.agents.llm.task import Task
from overcooked_ai_py.agents.llm.tool_state import ToolState
from overcooked_ai_py.mdp.actions import Action, Direction


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

    @tool
    def get_surroundings() -> str:
        """Check what is adjacent to you in each direction (up/down/left/right). Returns terrain type and any objects at each adjacent cell."""
        player = planner_tool_state.state.players[planner_tool_state.agent_index]
        pos = player.position
        lines = []
        for d, name in [(Direction.NORTH, "up"), (Direction.SOUTH, "down"),
                         (Direction.EAST, "right"), (Direction.WEST, "left")]:
            adj = Action.move_in_direction(pos, d)
            x, y = adj
            if 0 <= x < planner_tool_state.mdp.width and 0 <= y < planner_tool_state.mdp.height:
                terrain = planner_tool_state.mdp.terrain_mtx[y][x]
                terrain_name = {
                    " ": "floor", "X": "counter", "P": "pot",
                    "O": "onion_dispenser", "T": "tomato_dispenser",
                    "D": "dish_dispenser", "S": "serving_location", "#": "wall",
                }.get(terrain, terrain)

                # Check for objects
                obj_desc = ""
                if planner_tool_state.state.has_object(adj):
                    obj = planner_tool_state.state.get_object(adj)
                    if obj.name == "soup":
                        if obj.is_ready:
                            obj_desc = " [READY SOUP]"
                        elif obj.is_cooking:
                            obj_desc = f" [cooking, {obj.cook_time - obj._cooking_tick} ticks left]"
                        else:
                            if len(obj.ingredients) >= 3 and not planner_tool_state.mdp.old_dynamics:
                                obj_desc = " [FULL 3/3, NOT COOKING - INTERACT with empty hands to start]"
                            else:
                                obj_desc = f" [{len(obj.ingredients)}/3 ingredients]"
                    else:
                        obj_desc = f" [{obj.name}]"

                # Check for other player
                partner = planner_tool_state.state.players[1 - planner_tool_state.agent_index]
                player_desc = " [PARTNER HERE]" if adj == partner.position else ""

                lines.append(f"  {name}: {terrain_name}{obj_desc}{player_desc}")
            else:
                lines.append(f"  {name}: out of bounds")

        return "\n".join(lines)

    @tool
    def get_pot_details() -> str:
        """Get detailed status of all pots: ingredients list, cooking timer, ready flag."""
        pot_states = planner_tool_state.mdp.get_pot_states(planner_tool_state.state)
        lines = []
        for pot_pos in planner_tool_state.mdp.get_pot_locations():
            if pot_pos in pot_states.get("empty", []):
                lines.append(f"Pot at {pot_pos}: empty (0/3 ingredients)")
            elif planner_tool_state.state.has_object(pot_pos):
                soup = planner_tool_state.state.get_object(pot_pos)
                ingredients = soup.ingredients
                if soup.is_ready:
                    lines.append(f"Pot at {pot_pos}: READY! Ingredients: {', '.join(ingredients)}. Pick up a dish, face this pot, and interact to collect soup.")
                elif soup.is_cooking:
                    remaining = soup.cook_time - soup._cooking_tick
                    lines.append(f"Pot at {pot_pos}: COOKING, {remaining} ticks remaining. Ingredients: {', '.join(ingredients)}.")
                else:
                    if len(ingredients) >= 3 and not planner_tool_state.mdp.old_dynamics:
                        lines.append(
                            f"Pot at {pot_pos}: FULL (3/3) but NOT COOKING ({', '.join(ingredients)}). "
                            f"A worker with empty hands must INTERACT to start cooking."
                        )
                    else:
                        lines.append(f"Pot at {pot_pos}: {len(ingredients)}/3 ingredients ({', '.join(ingredients)}). Needs {3 - len(ingredients)} more.")
        if not lines:
            return "No pots found."
        return "\n".join(lines)

    @tool
    def check_path(target: str) -> str:
        """Check the number of steps to reach the nearest target location.

        Args:
            target: one of 'onion_dispenser', 'tomato_dispenser', 'dish_dispenser', 'pot', 'serving', 'dish' (counter dish), 'counter'
        """
        player = planner_tool_state.state.players[planner_tool_state.agent_index]
        start = player.pos_and_or

        target_map = {
            "onion_dispenser": planner_tool_state.mdp.get_onion_dispenser_locations,
            "tomato_dispenser": planner_tool_state.mdp.get_tomato_dispenser_locations,
            "dish_dispenser": planner_tool_state.mdp.get_dish_dispenser_locations,
            "pot": planner_tool_state.mdp.get_pot_locations,
            "serving": planner_tool_state.mdp.get_serving_locations,
            "counter": planner_tool_state.mdp.get_counter_locations,
        }

        if target == "dish":
            # Find dishes on counters
            counter_objects = planner_tool_state.mdp.get_counter_objects_dict(planner_tool_state.state)
            positions = counter_objects.get("dish", [])
            if not positions:
                return "No dishes found on counters."
        elif target in target_map:
            positions = target_map[target]()
        else:
            return f"Unknown target '{target}'. Use one of: {', '.join(list(target_map.keys()) + ['dish'])}"

        if not positions:
            return f"No {target} locations found."

        min_cost = np.inf
        best_pos = None
        for feature_pos in positions:
            if feature_pos not in planner_tool_state.motion_planner.motion_goals_for_pos:
                continue
            for goal in planner_tool_state.motion_planner.motion_goals_for_pos[feature_pos]:
                if not planner_tool_state.motion_planner.is_valid_motion_start_goal_pair(start, goal):
                    continue
                cost = planner_tool_state.motion_planner.get_gridworld_distance(start, goal)
                if cost < min_cost:
                    min_cost = cost
                    best_pos = feature_pos

        if best_pos is None or min_cost == np.inf:
            return f"Cannot reach any {target} from current position."
        return f"Nearest {target} is at {best_pos}, {int(min_cost)} steps away."

    @tool
    def get_worker_status(worker_id: str) -> str:
        """Get the current status of a worker: idle, working, or completed, plus task details.

        Args:
            worker_id: The worker identifier (e.g. 'worker_0')
        """
        if worker_id not in worker_registry:
            return f"Error: Unknown worker_id '{worker_id}'. Valid workers: {', '.join(sorted(worker_registry.keys()))}"
        status = worker_registry[worker_id].get_status()
        return json.dumps(status)

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
                errors.append(f"Task for '{worker_id}' must be a string, got {type(description).__name__}")
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

    observation_tools = [get_surroundings, get_pot_details, check_path, get_worker_status]
    action_tools = [assign_tasks]
    action_tool_names = {"assign_tasks"}
    return observation_tools, action_tools, action_tool_names
