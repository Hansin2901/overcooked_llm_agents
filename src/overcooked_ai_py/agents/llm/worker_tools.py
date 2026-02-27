"""Factory for creating per-worker observation and action tools.

Each worker gets its own tool set bound to its own ToolState via closures.
This ensures workers cannot access each other's state.
"""

import numpy as np
from langchain_core.tools import tool

from overcooked_ai_py.agents.llm.tool_state import ToolState
from overcooked_ai_py.mdp.actions import Action, Direction


def create_worker_tools(tool_state: ToolState) -> tuple:
    """Create worker tools bound to a specific ToolState.

    Each worker gets its own copy of these tools, bound to its own ToolState.
    This ensures workers cannot access each other's state.

    Returns:
        (observation_tools, action_tools, action_tool_names)
    """

    # -------------------------------------------------------------------
    # Observation Tools
    # -------------------------------------------------------------------

    @tool
    def get_surroundings() -> str:
        """Check what is adjacent to you in each direction (up/down/left/right). Returns terrain type and any objects at each adjacent cell."""
        player = tool_state.state.players[tool_state.agent_index]
        pos = player.position
        lines = []
        for d, name in [(Direction.NORTH, "up"), (Direction.SOUTH, "down"),
                         (Direction.EAST, "right"), (Direction.WEST, "left")]:
            adj = Action.move_in_direction(pos, d)
            x, y = adj
            if 0 <= x < tool_state.mdp.width and 0 <= y < tool_state.mdp.height:
                terrain = tool_state.mdp.terrain_mtx[y][x]
                terrain_name = {
                    " ": "floor", "X": "counter", "P": "pot",
                    "O": "onion_dispenser", "T": "tomato_dispenser",
                    "D": "dish_dispenser", "S": "serving_location", "#": "wall",
                }.get(terrain, terrain)

                # Check for objects
                obj_desc = ""
                if tool_state.state.has_object(adj):
                    obj = tool_state.state.get_object(adj)
                    if obj.name == "soup":
                        if obj.is_ready:
                            obj_desc = " [READY SOUP]"
                        elif obj.is_cooking:
                            obj_desc = f" [cooking, {obj.cook_time - obj._cooking_tick} ticks left]"
                        else:
                            if len(obj.ingredients) >= 3 and not tool_state.mdp.old_dynamics:
                                obj_desc = " [FULL 3/3, NOT COOKING - INTERACT with empty hands to start]"
                            else:
                                obj_desc = f" [{len(obj.ingredients)}/3 ingredients]"
                    else:
                        obj_desc = f" [{obj.name}]"

                # Check for other player
                partner = tool_state.state.players[1 - tool_state.agent_index]
                player_desc = " [PARTNER HERE]" if adj == partner.position else ""

                lines.append(f"  {name}: {terrain_name}{obj_desc}{player_desc}")
            else:
                lines.append(f"  {name}: out of bounds")

        return "\n".join(lines)

    @tool
    def get_pot_details() -> str:
        """Get detailed status of all pots: ingredients list, cooking timer, ready flag."""
        pot_states = tool_state.mdp.get_pot_states(tool_state.state)
        lines = []
        for pot_pos in tool_state.mdp.get_pot_locations():
            if pot_pos in pot_states.get("empty", []):
                lines.append(f"Pot at {pot_pos}: empty (0/3 ingredients)")
            elif tool_state.state.has_object(pot_pos):
                soup = tool_state.state.get_object(pot_pos)
                ingredients = soup.ingredients
                if soup.is_ready:
                    lines.append(f"Pot at {pot_pos}: READY! Ingredients: {', '.join(ingredients)}. Pick up a dish, face this pot, and interact to collect soup.")
                elif soup.is_cooking:
                    remaining = soup.cook_time - soup._cooking_tick
                    lines.append(f"Pot at {pot_pos}: COOKING, {remaining} ticks remaining. Ingredients: {', '.join(ingredients)}.")
                else:
                    if len(ingredients) >= 3 and not tool_state.mdp.old_dynamics:
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
        player = tool_state.state.players[tool_state.agent_index]
        start = player.pos_and_or

        target_map = {
            "onion_dispenser": tool_state.mdp.get_onion_dispenser_locations,
            "tomato_dispenser": tool_state.mdp.get_tomato_dispenser_locations,
            "dish_dispenser": tool_state.mdp.get_dish_dispenser_locations,
            "pot": tool_state.mdp.get_pot_locations,
            "serving": tool_state.mdp.get_serving_locations,
            "counter": tool_state.mdp.get_counter_locations,
        }

        if target == "dish":
            # Find dishes on counters
            counter_objects = tool_state.mdp.get_counter_objects_dict(tool_state.state)
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
            if feature_pos not in tool_state.motion_planner.motion_goals_for_pos:
                continue
            for goal in tool_state.motion_planner.motion_goals_for_pos[feature_pos]:
                if not tool_state.motion_planner.is_valid_motion_start_goal_pair(start, goal):
                    continue
                cost = tool_state.motion_planner.get_gridworld_distance(start, goal)
                if cost < min_cost:
                    min_cost = cost
                    best_pos = feature_pos

        if best_pos is None or min_cost == np.inf:
            return f"Cannot reach any {target} from current position."
        return f"Nearest {target} is at {best_pos}, {int(min_cost)} steps away."

    # -------------------------------------------------------------------
    # Action Tools
    # -------------------------------------------------------------------

    @tool
    def move_up() -> str:
        """Move north (up). Changes your position and facing direction."""
        tool_state.set_action(Direction.NORTH)
        return "Moving up."

    @tool
    def move_down() -> str:
        """Move south (down). Changes your position and facing direction."""
        tool_state.set_action(Direction.SOUTH)
        return "Moving down."

    @tool
    def move_left() -> str:
        """Move west (left). Changes your position and facing direction."""
        tool_state.set_action(Direction.WEST)
        return "Moving left."

    @tool
    def move_right() -> str:
        """Move east (right). Changes your position and facing direction."""
        tool_state.set_action(Direction.EAST)
        return "Moving right."

    @tool
    def wait() -> str:
        """Stay in place. Do nothing this turn."""
        tool_state.set_action(Action.STAY)
        return "Waiting."

    @tool
    def interact() -> str:
        """Interact with the object/terrain you are currently facing. Used to pick up items, place items in pots, pick up soup with dish, and deliver soup."""
        tool_state.set_action(Action.INTERACT)
        return "Interacting."

    # -------------------------------------------------------------------
    # Collect and return
    # -------------------------------------------------------------------

    observation_tools = [get_surroundings, get_pot_details, check_path]
    action_tools = [move_up, move_down, move_left, move_right, wait, interact]
    action_tool_names = {t.name for t in action_tools}
    return observation_tools, action_tools, action_tool_names
