"""Observation and action tools for the LLM Overcooked agent.

Observation tools return strings and don't end the turn.
Action tools commit to a game action and terminate the LangGraph loop.
"""

import numpy as np
from langchain_core.tools import tool

from overcooked_ai_py.mdp.actions import Action, Direction


# Sentinel value to signal that an action tool was called
ACTION_CHOSEN = "__action_chosen__"

# Module-level references set by `init_tools()`
_mdp = None
_state = None
_agent_index = None
_motion_planner = None
_chosen_action = None


def init_tools(mdp, motion_planner):
    """Set the mdp and motion planner references (called once per episode)."""
    global _mdp, _motion_planner
    _mdp = mdp
    _motion_planner = motion_planner


def set_state(state, agent_index):
    """Update the current state before each LLM step."""
    global _state, _agent_index, _chosen_action
    _state = state
    _agent_index = agent_index
    _chosen_action = None


def get_chosen_action():
    """Return the action chosen by the last action tool call, or None."""
    return _chosen_action


# ---------------------------------------------------------------------------
# Observation Tools
# ---------------------------------------------------------------------------

@tool
def get_surroundings() -> str:
    """Check what is adjacent to you in each direction (up/down/left/right). Returns terrain type and any objects at each adjacent cell."""
    player = _state.players[_agent_index]
    pos = player.position
    lines = []
    for d, name in [(Direction.NORTH, "up"), (Direction.SOUTH, "down"),
                     (Direction.EAST, "right"), (Direction.WEST, "left")]:
        adj = Action.move_in_direction(pos, d)
        x, y = adj
        if 0 <= x < _mdp.width and 0 <= y < _mdp.height:
            terrain = _mdp.terrain_mtx[y][x]
            terrain_name = {
                " ": "floor", "X": "counter", "P": "pot",
                "O": "onion_dispenser", "T": "tomato_dispenser",
                "D": "dish_dispenser", "S": "serving_location", "#": "wall",
            }.get(terrain, terrain)

            # Check for objects
            obj_desc = ""
            if _state.has_object(adj):
                obj = _state.get_object(adj)
                if obj.name == "soup":
                    if obj.is_ready:
                        obj_desc = " [READY SOUP]"
                    elif obj.is_cooking:
                        obj_desc = f" [cooking, {obj.cook_time - obj._cooking_tick} ticks left]"
                    else:
                        obj_desc = f" [{len(obj.ingredients)}/3 ingredients]"
                else:
                    obj_desc = f" [{obj.name}]"

            # Check for other player
            partner = _state.players[1 - _agent_index]
            player_desc = " [PARTNER HERE]" if adj == partner.position else ""

            lines.append(f"  {name}: {terrain_name}{obj_desc}{player_desc}")
        else:
            lines.append(f"  {name}: out of bounds")

    return "\n".join(lines)


@tool
def get_pot_details() -> str:
    """Get detailed status of all pots: ingredients list, cooking timer, ready flag."""
    pot_states = _mdp.get_pot_states(_state)
    lines = []
    for pot_pos in _mdp.get_pot_locations():
        if pot_pos in pot_states.get("empty", []):
            lines.append(f"Pot at {pot_pos}: empty (0/3 ingredients)")
        elif _state.has_object(pot_pos):
            soup = _state.get_object(pot_pos)
            ingredients = soup.ingredients
            if soup.is_ready:
                lines.append(f"Pot at {pot_pos}: READY! Ingredients: {', '.join(ingredients)}. Pick up a dish, face this pot, and interact to collect soup.")
            elif soup.is_cooking:
                remaining = soup.cook_time - soup._cooking_tick
                lines.append(f"Pot at {pot_pos}: COOKING, {remaining} ticks remaining. Ingredients: {', '.join(ingredients)}.")
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
    player = _state.players[_agent_index]
    start = player.pos_and_or

    target_map = {
        "onion_dispenser": _mdp.get_onion_dispenser_locations,
        "tomato_dispenser": _mdp.get_tomato_dispenser_locations,
        "dish_dispenser": _mdp.get_dish_dispenser_locations,
        "pot": _mdp.get_pot_locations,
        "serving": _mdp.get_serving_locations,
        "counter": _mdp.get_counter_locations,
    }

    if target == "dish":
        # Find dishes on counters
        counter_objects = _mdp.get_counter_objects_dict(_state)
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
        if feature_pos not in _motion_planner.motion_goals_for_pos:
            continue
        for goal in _motion_planner.motion_goals_for_pos[feature_pos]:
            if not _motion_planner.is_valid_motion_start_goal_pair(start, goal):
                continue
            cost = _motion_planner.get_gridworld_distance(start, goal)
            if cost < min_cost:
                min_cost = cost
                best_pos = feature_pos

    if best_pos is None or min_cost == np.inf:
        return f"Cannot reach any {target} from current position."
    return f"Nearest {target} is at {best_pos}, {int(min_cost)} steps away."


# ---------------------------------------------------------------------------
# Action Tools
# ---------------------------------------------------------------------------

def _set_action(action):
    """Internal: record the chosen action."""
    global _chosen_action
    _chosen_action = action


@tool
def move_up() -> str:
    """Move north (up). Changes your position and facing direction."""
    _set_action(Direction.NORTH)
    return "Moving up."


@tool
def move_down() -> str:
    """Move south (down). Changes your position and facing direction."""
    _set_action(Direction.SOUTH)
    return "Moving down."


@tool
def move_left() -> str:
    """Move west (left). Changes your position and facing direction."""
    _set_action(Direction.WEST)
    return "Moving left."


@tool
def move_right() -> str:
    """Move east (right). Changes your position and facing direction."""
    _set_action(Direction.EAST)
    return "Moving right."


@tool
def wait() -> str:
    """Stay in place. Do nothing this turn."""
    _set_action(Action.STAY)
    return "Waiting."


@tool
def interact() -> str:
    """Interact with the object/terrain you are currently facing. Used to pick up items, place items in pots, pick up soup with dish, and deliver soup."""
    _set_action(Action.INTERACT)
    return "Interacting."


# Tool collections
OBSERVATION_TOOLS = [get_surroundings, get_pot_details, check_path]
ACTION_TOOLS = [move_up, move_down, move_left, move_right, wait, interact]
ALL_TOOLS = OBSERVATION_TOOLS + ACTION_TOOLS
ACTION_TOOL_NAMES = {t.name for t in ACTION_TOOLS}
