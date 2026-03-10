"""Factory for creating planner observation and action tools.

The planner gets tools to assign tasks to workers and query their status,
plus read-only observation tools. The planner receives the worker_registry
(dict of worker_id -> ToolState) -- workers never see this registry.
"""

import json
import re

import numpy as np
from langchain_core.tools import tool

from overcooked_ai_py.agents.llm.task import Task
from overcooked_ai_py.agents.llm.state_serializer import _layout_recipe_context
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

    @tool
    def get_surroundings() -> str:
        """Check what is adjacent in each direction."""
        player = planner_tool_state.state.players[planner_tool_state.agent_index]
        pos = player.position
        lines = []
        for d, name in [
            (Direction.NORTH, "up"),
            (Direction.SOUTH, "down"),
            (Direction.EAST, "right"),
            (Direction.WEST, "left"),
        ]:
            adj = Action.move_in_direction(pos, d)
            x, y = adj
            if 0 <= x < planner_tool_state.mdp.width and 0 <= y < planner_tool_state.mdp.height:
                terrain = planner_tool_state.mdp.terrain_mtx[y][x]
                terrain_name = {
                    " ": "floor",
                    "X": "counter",
                    "P": "pot",
                    "O": "onion_dispenser",
                    "T": "tomato_dispenser",
                    "D": "dish_dispenser",
                    "S": "serving_location",
                    "#": "wall",
                }.get(terrain, terrain)

                obj_desc = ""
                if planner_tool_state.state.has_object(adj):
                    obj = planner_tool_state.state.get_object(adj)
                    if obj.name == "soup":
                        if obj.is_ready:
                            obj_desc = " [READY SOUP]"
                        elif obj.is_cooking:
                            remaining = obj.cook_time - obj._cooking_tick
                            obj_desc = f" [cooking, {remaining} ticks left]"
                        elif len(obj.ingredients) >= 3 and not planner_tool_state.mdp.old_dynamics:
                            obj_desc = " [FULL 3/3, NOT COOKING - interact with empty hands to start]"
                        else:
                            obj_desc = f" [{len(obj.ingredients)}/3 ingredients]"
                    else:
                        obj_desc = f" [{obj.name}]"

                partner = planner_tool_state.state.players[1 - planner_tool_state.agent_index]
                player_desc = " [PARTNER HERE]" if adj == partner.position else ""
                lines.append(f"  {name}: {terrain_name}{obj_desc}{player_desc}")
            else:
                lines.append(f"  {name}: out of bounds")

        return "\n".join(lines)

    @tool
    def get_pot_details() -> str:
        """Get detailed status of all pots."""
        pot_states = planner_tool_state.mdp.get_pot_states(planner_tool_state.state)
        recipe_context = _layout_recipe_context(planner_tool_state.mdp)
        lines = []
        for pot_pos in planner_tool_state.mdp.get_pot_locations():
            if pot_pos in pot_states.get("empty", []):
                lines.append(f"Pot at {pot_pos}: empty (0/3 ingredients)")
            elif planner_tool_state.state.has_object(pot_pos):
                soup = planner_tool_state.state.get_object(pot_pos)
                ingredients = soup.ingredients
                if soup.is_ready:
                    lines.append(
                        f"Pot at {pot_pos}: READY! Ingredients: {', '.join(ingredients)}. "
                        "Pick up a dish, face this pot, and interact to collect soup."
                    )
                elif soup.is_cooking:
                    remaining = soup.cook_time - soup._cooking_tick
                    lines.append(
                        f"Pot at {pot_pos}: COOKING, {remaining} ticks remaining. "
                        f"Ingredients: {', '.join(ingredients)}."
                    )
                elif len(ingredients) >= 3 and not planner_tool_state.mdp.old_dynamics:
                    lines.append(
                        f"Pot at {pot_pos}: FULL (3/3) but NOT COOKING ({', '.join(ingredients)}). "
                        "A worker with empty hands must INTERACT to start cooking."
                    )
                else:
                    needed = 3 - len(ingredients)
                    if recipe_context["onion_only_three"]:
                        need_text = f"Needs {needed} more onion(s)."
                    elif recipe_context["tomato_only_three"]:
                        need_text = f"Needs {needed} more tomato(es)."
                    else:
                        need_text = f"Needs {needed} more ingredient(s)."
                    lines.append(
                        f"Pot at {pot_pos}: {len(ingredients)}/3 ingredients "
                        f"({', '.join(ingredients)}). {need_text}"
                    )
        if not lines:
            return "No pots found."
        return "\n".join(lines)

    @tool
    def check_path(target: str) -> str:
        """Check steps to the nearest target."""
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
            counter_objects = planner_tool_state.mdp.get_counter_objects_dict(planner_tool_state.state)
            positions = counter_objects.get("dish", [])
            if not positions:
                return "No dishes found on counters."
        elif target in target_map:
            positions = target_map[target]()
        else:
            valid_targets = list(target_map.keys()) + ["dish"]
            return f"Unknown target '{target}'. Use one of: {', '.join(valid_targets)}"

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
        """Get the current status of a worker."""
        if worker_id not in worker_registry:
            valid_workers = ", ".join(sorted(worker_registry.keys()))
            return f"Error: Unknown worker_id '{worker_id}'. Valid workers: {valid_workers}"
        return json.dumps(worker_registry[worker_id].get_status())

    def _extract_pot_target(description: str):
        match = re.search(r"pot at\s*\((\d+)\s*,\s*(\d+)\)", description, re.IGNORECASE)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None

    def _extract_coords(description: str):
        return [
            (int(match.group(1)), int(match.group(2)))
            for match in re.finditer(r"\((\d+)\s*,\s*(\d+)\)", description)
        ]

    def _terrain_at(pos):
        if pos is None:
            return None
        x, y = pos
        if 0 <= x < planner_tool_state.mdp.width and 0 <= y < planner_tool_state.mdp.height:
            return planner_tool_state.mdp.terrain_mtx[y][x]
        return None

    def _first_coord_with_terrain(coords, terrain_chars):
        for pos in coords:
            if _terrain_at(pos) in terrain_chars:
                return pos
        return None

    def _extract_serving_target(description: str):
        match = re.search(
            r"serv(?:e|ing)(?: location)? at\s*\((\d+)\s*,\s*(\d+)\)",
            description,
            re.IGNORECASE,
        )
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None

    def _looks_like_raw_action_script(description: str) -> bool:
        lower = description.lower().strip()
        semantic_markers = [
            "dispenser",
            "pot",
            "serving",
            "serve",
            "dish",
            "onion",
            "tomato",
            "soup",
            "counter",
        ]
        if any(marker in lower for marker in semantic_markers):
            return False

        primitive_tokens = re.findall(
            r"\b(left|right|up|down|interact|stay|wait|move)\b",
            lower,
        )
        return bool(primitive_tokens)

    def _is_ambiguous_pot_task(description: str) -> bool:
        lower = description.lower().strip()
        if "pot" not in lower:
            return False
        meaningful_pot_markers = [
            "drop onion",
            "drop tomato",
            "deliver to pot",
            "place onion",
            "place tomato",
            "start cooking",
            "collect soup",
            "pick soup",
            "ready soup",
            "with dish",
        ]
        return not any(marker in lower for marker in meaningful_pot_markers)

    def _canonicalize_assignment(description: str) -> str | None:
        lower = description.lower().strip()
        coords = _extract_coords(description)
        layout_name = getattr(planner_tool_state.mdp, "layout_name", "")
        pot_pos = _extract_pot_target(description) or _first_coord_with_terrain(coords, {"P"})
        serving_pos = _extract_serving_target(description) or _first_coord_with_terrain(coords, {"S"})
        onion_disp = _first_coord_with_terrain(coords, {"O"})
        tomato_disp = _first_coord_with_terrain(coords, {"T"})
        dish_disp = _first_coord_with_terrain(coords, {"D"})
        counter_pos = _first_coord_with_terrain(coords, {"X"})

        if _looks_like_raw_action_script(description):
            return None

        if layout_name == "counter_circuit" and _is_ambiguous_pot_task(description):
            return None

        if "start cooking" in lower:
            if pot_pos is None:
                return None
            return f"Go to pot at {pot_pos}, interact to start cooking"

        if "serve" in lower and pot_pos is not None and serving_pos is not None:
            return (
                f"Go to pot at {pot_pos}, collect soup with dish, "
                f"deliver to serving at {serving_pos}"
            )

        if (
            layout_name == "counter_circuit"
            and ("pick dish" in lower or "dish dispenser" in lower)
            and pot_pos is not None
            and serving_pos is not None
        ):
            if dish_disp is not None:
                return (
                    f"Go to dish dispenser at {dish_disp}, pick dish, "
                    f"collect soup from pot at {pot_pos}, deliver to serving at {serving_pos}"
                )
            if counter_pos is not None:
                return (
                    f"Go to shared counter at {counter_pos}, pick dish, "
                    f"collect soup from pot at {pot_pos}, deliver to serving at {serving_pos}"
                )

        if any(phrase in lower for phrase in ["drop dish", "place dish"]):
            if counter_pos is not None:
                return f"Go to shared counter at {counter_pos}, interact to drop dish"
            if serving_pos is not None:
                return f"Go to serving at {serving_pos}, deliver soup"
            return None

        if "pick dish" in lower or "dish dispenser" in lower:
            if dish_disp is not None and counter_pos is not None:
                return (
                    f"Go to dish dispenser at {dish_disp}, pick dish, "
                    f"deliver to shared counter at {counter_pos}"
                )
            if dish_disp is not None:
                return f"Go to dish dispenser at {dish_disp}, pick dish"
            if counter_pos is not None and pot_pos is not None and serving_pos is not None:
                return (
                    f"Go to shared counter at {counter_pos}, pick dish, "
                    f"collect soup from pot at {pot_pos}, deliver to serving at {serving_pos}"
                )
            if counter_pos is not None:
                return f"Go to shared counter at {counter_pos}, pick dish"
            return None

        if "pick onion" in lower or "onion dispenser" in lower:
            if onion_disp is not None and pot_pos is not None:
                return (
                    f"Go to onion dispenser at {onion_disp}, pick onion, "
                    f"deliver to pot at {pot_pos}"
                )
            if onion_disp is not None and counter_pos is not None:
                return (
                    f"Go to onion dispenser at {onion_disp}, pick onion, "
                    f"deliver to shared counter at {counter_pos}"
                )
            if counter_pos is not None and pot_pos is not None:
                return (
                    f"Go to shared counter at {counter_pos}, pick onion, "
                    f"deliver to pot at {pot_pos}"
                )
            if onion_disp is not None:
                return f"Go to onion dispenser at {onion_disp}, pick onion"
            if counter_pos is not None:
                return f"Go to shared counter at {counter_pos}, pick onion"
            return None

        if "pick tomato" in lower or "tomato dispenser" in lower:
            if tomato_disp is not None and pot_pos is not None:
                return (
                    f"Go to tomato dispenser at {tomato_disp}, pick tomato, "
                    f"deliver to pot at {pot_pos}"
                )
            if tomato_disp is not None and counter_pos is not None:
                return (
                    f"Go to tomato dispenser at {tomato_disp}, pick tomato, "
                    f"deliver to shared counter at {counter_pos}"
                )
            if counter_pos is not None and pot_pos is not None:
                return (
                    f"Go to shared counter at {counter_pos}, pick tomato, "
                    f"deliver to pot at {pot_pos}"
                )
            if tomato_disp is not None:
                return f"Go to tomato dispenser at {tomato_disp}, pick tomato"
            if counter_pos is not None:
                return f"Go to shared counter at {counter_pos}, pick tomato"
            return None

        if any(phrase in lower for phrase in ["drop onion", "place onion", "deliver to pot"]):
            if pot_pos is None:
                if counter_pos is not None:
                    return f"Go to shared counter at {counter_pos}, interact to drop onion"
                return None
            return f"Go to pot at {pot_pos}, interact to drop onion"

        if any(phrase in lower for phrase in ["drop tomato", "place tomato"]):
            if pot_pos is None:
                if counter_pos is not None:
                    return f"Go to shared counter at {counter_pos}, interact to drop tomato"
                return None
            return f"Go to pot at {pot_pos}, interact to drop tomato"

        if lower in {"wait", "stay"} or lower.startswith("wait ") or lower.startswith("stay "):
            return "Move to a non-blocking nearby tile and stay ready to support the other worker."

        return description

    def _pot_is_full_idle(pot_pos) -> bool:
        if pot_pos is None:
            return False
        if not planner_tool_state.state.has_object(pot_pos):
            return False
        obj = planner_tool_state.state.get_object(pot_pos)
        return (
            obj.name == "soup"
            and not planner_tool_state.mdp.old_dynamics
            and not obj.is_cooking
            and not obj.is_ready
            and len(obj.ingredients) >= 3
        )

    @tool
    def assign_tasks(
        assignments: str = "",
        worker_0: str = "",
        worker_1: str = "",
    ) -> str:
        """Assign tasks to workers and end the planning turn."""
        if assignments:
            if isinstance(assignments, dict):
                parsed = assignments
            else:
                try:
                    parsed = json.loads(assignments)
                except json.JSONDecodeError as exc:
                    return f"Error: Invalid assignments JSON - {exc}"
        else:
            parsed = {}
            if worker_0:
                parsed["worker_0"] = worker_0
            if worker_1:
                parsed["worker_1"] = worker_1

        if not isinstance(parsed, dict) or not parsed:
            return "Error: Assignments must be a JSON object mapping worker_id to task description."

        errors = []
        assigned = []
        timestep = getattr(planner_tool_state.state, "timestep", 0) if planner_tool_state.state else 0

        for worker_id in sorted(worker_registry.keys()):
            description = parsed.get(worker_id)
            if not description:
                existing = worker_registry[worker_id].current_task
                if existing and not existing.completed:
                    description = existing.description
                else:
                    description = (
                        "Move to a non-blocking nearby tile and stay ready to support "
                        "the other worker."
                    )

            if not isinstance(description, str):
                errors.append(
                    f"Task for '{worker_id}' must be a string, got {type(description).__name__}"
                )
                continue

            normalized = _canonicalize_assignment(description)
            if normalized is None:
                errors.append(
                    f"Rejected invalid raw-action task for '{worker_id}': {description}"
                )
                existing = worker_registry[worker_id].current_task
                if existing and not existing.completed:
                    description = existing.description
                else:
                    description = (
                        "Move to a non-blocking nearby tile and stay ready to support "
                        "the other worker."
                    )
            else:
                description = normalized

            if "start cooking" in description.lower():
                pot_pos = _extract_pot_target(description)
                if not _pot_is_full_idle(pot_pos):
                    errors.append(
                        f"Rejected invalid start-cooking task for '{worker_id}': pot is not full"
                    )
                    existing = worker_registry[worker_id].current_task
                    if existing and not existing.completed:
                        description = existing.description
                    else:
                        description = (
                            "Move to a non-blocking nearby tile and stay ready to support "
                            "the other worker."
                        )

            task = Task(
                description=description,
                worker_id=worker_id,
                created_at=timestep,
            )
            worker_registry[worker_id].set_task(task)
            assigned.append(f"{worker_id}: {description}")

        unknown_workers = sorted(set(parsed.keys()) - set(worker_registry.keys()))
        if unknown_workers:
            errors.extend(f"Unknown worker_id '{worker_id}'" for worker_id in unknown_workers)

        result_parts = []
        if assigned:
            result_parts.append("Assigned: " + "; ".join(assigned))
        if errors:
            result_parts.append("Errors: " + "; ".join(errors))
        return ". ".join(result_parts) if result_parts else "No assignments made."

    observation_tools = [get_surroundings, get_pot_details, check_path, get_worker_status]
    action_tools = [assign_tasks]
    action_tool_names = {"assign_tasks"}
    return observation_tools, action_tools, action_tool_names
