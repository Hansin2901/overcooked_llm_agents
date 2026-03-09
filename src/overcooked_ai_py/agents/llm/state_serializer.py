"""Serialize OvercookedState into concise text for LLM consumption."""

from collections import deque

from overcooked_ai_py.mdp.actions import Action, Direction


# Map terrain chars to human-readable names
TERRAIN_LEGEND = {
    " ": "floor",
    "X": "counter",
    "P": "pot",
    "O": "onion_dispenser",
    "T": "tomato_dispenser",
    "S": "serving_location",
    "#": "wall",
}

DIRECTION_NAMES = {
    Direction.NORTH: "up",
    Direction.SOUTH: "down",
    Direction.EAST: "right",
    Direction.WEST: "left",
}


def _uses_old_dynamics(mdp) -> bool:
    """Return whether this layout uses old auto-start cooking dynamics."""
    return bool(getattr(mdp, "old_dynamics", False))


def _layout_recipe_context(mdp):
    """Summarize available ingredients and the recipe constraints for the layout."""
    available = []
    if mdp.get_onion_dispenser_locations():
        available.append("onion")
    if mdp.get_tomato_dispenser_locations():
        available.append("tomato")

    orders = getattr(mdp, "start_all_orders", None) or []
    recipe_strings = []
    for order in orders:
        ingredients = order.get("ingredients", []) if isinstance(order, dict) else []
        if ingredients:
            recipe_strings.append(", ".join(ingredients))

    unique_recipes = sorted(set(recipe_strings))
    onion_only_three = unique_recipes == ["onion, onion, onion"]
    tomato_only_three = unique_recipes == ["tomato, tomato, tomato"]

    if onion_only_three:
        recipe_rule = (
            "This layout only serves onion soup. You need exactly 3 onions in a pot "
            "before cooking can start."
        )
    elif tomato_only_three:
        recipe_rule = (
            "This layout only serves tomato soup. You need exactly 3 tomatoes in a pot "
            "before cooking can start."
        )
    elif unique_recipes:
        recipe_rule = "Valid recipes on this layout: " + "; ".join(unique_recipes) + "."
    else:
        recipe_rule = "Use the current order list to determine the correct recipe."

    if available:
        available_rule = "Available ingredient dispensers: " + ", ".join(available) + "."
    else:
        available_rule = "No ingredient dispenser metadata found."

    return {
        "onion_only_three": onion_only_three,
        "tomato_only_three": tomato_only_three,
        "recipe_rule": recipe_rule,
        "available_rule": available_rule,
    }


def _shortest_position_distances(mdp, start_pos):
    """Compute shortest grid distances from a start position to all valid floor tiles."""
    valid_positions = set(mdp.get_valid_player_positions())
    queue = deque([(start_pos, 0)])
    distances = {start_pos: 0}

    while queue:
        pos, steps = queue.popleft()
        for direction in Direction.ALL_DIRECTIONS:
            nxt = Action.move_in_direction(pos, direction)
            if nxt in valid_positions and nxt not in distances:
                distances[nxt] = steps + 1
                queue.append((nxt, steps + 1))

    return distances


def _reachable_feature_distances(distance_map, feature_pos):
    """Return the shortest step count to stand adjacent to a terrain feature."""
    reachable = []
    for direction in Direction.ALL_DIRECTIONS:
        adj = Action.move_in_direction(feature_pos, direction)
        if adj in distance_map:
            reachable.append(distance_map[adj])
    return min(reachable) if reachable else None


def _format_feature_summary(label, reachable_features):
    """Summarize the nearest reachable terrain feature."""
    if not reachable_features:
        return f"cannot reach any {label}"
    pos, steps = reachable_features[0]
    return f"closest {label} {pos} in {steps} step(s)"


def _format_positions(positions):
    """Format a list of positions as a short string."""
    if not positions:
        return "none"
    return ", ".join(str(pos) for pos in positions)


def _shared_handoff_counters(mdp, distance_maps):
    """Return counters that both players can stand adjacent to."""
    shared = []
    for counter_pos in mdp.get_counter_locations():
        if all(
            _reachable_feature_distances(distance_map, counter_pos) is not None
            for distance_map in distance_maps
        ):
            shared.append(counter_pos)
    return shared


def _layout_strategy_context(mdp):
    """Build layout-aware coordination guidance for prompts."""
    feature_getters = {
        "onion dispensers": mdp.get_onion_dispenser_locations(),
        "tomato dispensers": mdp.get_tomato_dispenser_locations(),
        "dish dispensers": mdp.get_dish_dispenser_locations(),
        "pots": mdp.get_pot_locations(),
        "serving locations": mdp.get_serving_locations(),
    }

    distance_maps = [
        _shortest_position_distances(mdp, start_pos)
        for start_pos in mdp.start_player_positions
    ]

    access = []
    for distance_map in distance_maps:
        player_access = {}
        for label, positions in feature_getters.items():
            reachable = []
            for pos in positions:
                steps = _reachable_feature_distances(distance_map, pos)
                if steps is not None:
                    reachable.append((pos, steps))
            player_access[label] = sorted(reachable, key=lambda item: item[1])
        access.append(player_access)

    lines = [f"Layout name: {mdp.layout_name}."]
    for idx, start_pos in enumerate(mdp.start_player_positions):
        summaries = [
            _format_feature_summary("onion dispenser", access[idx]["onion dispensers"]),
            _format_feature_summary("dish dispenser", access[idx]["dish dispensers"]),
            _format_feature_summary("pot", access[idx]["pots"]),
            _format_feature_summary("serving location", access[idx]["serving locations"]),
        ]
        if feature_getters["tomato dispensers"]:
            summaries.insert(
                1,
                _format_feature_summary("tomato dispenser", access[idx]["tomato dispensers"]),
            )
        lines.append(f"Player {idx} starts at {start_pos}: " + "; ".join(summaries) + ".")

    layout_name = getattr(mdp, "layout_name", "")
    if layout_name == "cramped_room":
        lines.extend(
            [
                "This is a tight single-room layout with one shared pot, one dish dispenser, and frequent body blocking in the center.",
                "Player 0 starts on the dish-side and Player 1 starts on the pot-and-serving side.",
                "Preferred split: keep one worker focused on pot filling and cooking while the other supports without blocking the center lane.",
                "Important sequencing rule for this layout: do NOT pick up a dish early while the pot still needs onions. First fill the pot and start cooking, then fetch exactly one dish when the soup is cooking or nearly ready.",
                "Do not leave a worker holding a dish and waiting for many steps. If soup is not yet cooking, that worker should help clear space, position for the next handoff, or contribute to the ingredient pipeline.",
                "Until the pot reaches 3/3 ingredients, both workers should usually contribute directly to the onion pipeline unless one must briefly move aside to avoid blocking.",
                "Do not assign vague wait/support tasks in cramped_room while the pot still needs ingredients. Only use a one-step repositioning move if it immediately helps unblock the other worker.",
                "Do not tell a worker to drop an onion on a floor tile or random counter in cramped_room. Onions should normally go directly from dispenser to pot.",
                "Once the pot is full, assign exactly one worker to start cooking and exactly one worker to fetch the next needed dish. Do not have both workers wait at the pot.",
            ]
        )
    elif layout_name == "asymmetric_advantages":
        left_onion = _format_positions([pos for pos, _ in access[1]["onion dispensers"]])
        right_onion = _format_positions([pos for pos, _ in access[0]["onion dispensers"]])
        left_dish = _format_positions([pos for pos, _ in access[1]["dish dispensers"]])
        right_dish = _format_positions([pos for pos, _ in access[0]["dish dispensers"]])
        left_serve = _format_positions([pos for pos, _ in access[1]["serving locations"]])
        right_serve = _format_positions([pos for pos, _ in access[0]["serving locations"]])
        lines.extend(
            [
                "This map is split into left and right support zones around shared central pots.",
                f"Player 0 is effectively the right-side worker: onion {right_onion}, dish {right_dish}, serving {right_serve}.",
                f"Player 1 is effectively the left-side worker: onion {left_onion}, dish {left_dish}, serving {left_serve}.",
                "Use the two pots as the coordination point. Each worker should usually source ingredients, dishes, and deliveries from their own side instead of crossing through the middle.",
            ]
        )
    elif layout_name == "coordination_ring":
        lines.extend(
            [
                "This layout is a loop around a central blocker. Head-on traffic on the same arc wastes time.",
                "Player 0 starts nearest the pots. Player 1 starts nearest the onions, dish dispenser, and serving side.",
                "Use a ring pipeline: Player 1 should usually feed onions and dishes from the lower-left side while Player 0 manages pot-side cooking work near the upper-right side.",
                "Avoid sending both workers around the same segment of the ring unless a soup is ready and must be served immediately.",
            ]
        )
    elif layout_name == "forced_coordination":
        shared_counters = _shared_handoff_counters(mdp, distance_maps)
        lines.extend(
            [
                "This layout is intentionally split. Player 0 on the right cannot reach onions or dishes. Player 1 on the left cannot reach pots or serving.",
                f"The shared handoff counters are { _format_positions(shared_counters) }.",
                "Required strategy: Player 1 gathers onions and dishes and stages them on the shared counters. Player 0 picks from those counters, fills pots, starts cooking, picks up soup, and serves.",
                "Do not assign solo soup cycles here. The map requires explicit handoffs.",
            ]
        )
    elif layout_name == "counter_circuit":
        lines.extend(
            [
                "This layout is a long circuit around a central counter island. Travel distance is the main cost.",
                "Player 0 starts on the ingredient-and-serving side. Player 1 starts on the dish-and-pot side.",
                "This layout is not onion-only: both onion and tomato dispensers exist, and the current orders can require mixed recipes.",
                "Preferred split: Player 0 feeds needed ingredients and can finish nearby deliveries, while Player 1 manages dish pickup, pot interactions, cooking starts, and soup collection on the top side.",
                "Use counters for staging and pipeline work. Do not send both workers on long laps unless the order state truly requires it.",
            ]
        )
    else:
        lines.append(
            "Use the closest-worker principle, respect any exclusive station access, and prefer complementary roles over duplicate movement."
        )

    return "\n".join(f"- {line}" for line in lines)


def serialize_state(mdp, state, agent_index, horizon=None):
    """Convert an OvercookedState to a text description for the LLM.

    Args:
        mdp: OvercookedGridworld instance
        state: OvercookedState
        agent_index: which player we are (0 or 1)
        horizon: total episode horizon (for remaining time display)

    Returns:
        str: text description of the current state
    """
    parts = []

    # Timestep
    parts.append(f"Timestep: {state.timestep}" + (f" / {horizon}" if horizon else ""))

    # Grid with player positions marked
    parts.append(_serialize_grid(mdp, state, agent_index))

    # Player info
    parts.append(_serialize_players(state, agent_index))

    # Pot status
    parts.append(_serialize_pots(mdp, state))

    # Counter objects
    parts.append(_serialize_counters(mdp, state))

    # Orders
    parts.append(_serialize_orders(state))

    return "\n\n".join(parts)


def _serialize_grid(mdp, state, agent_index):
    """Render terrain grid with player positions marked."""
    lines = ["GRID:"]
    p0_pos = state.players[0].position
    p1_pos = state.players[1].position

    you_marker = str(agent_index)
    partner_marker = str(1 - agent_index)

    for y in range(mdp.height):
        row = ""
        for x in range(mdp.width):
            pos = (x, y)
            if pos == p0_pos and agent_index == 0:
                row += "Y"  # You
            elif pos == p1_pos and agent_index == 1:
                row += "Y"  # You
            elif pos == p0_pos:
                row += "@"  # Partner
            elif pos == p1_pos:
                row += "@"  # Partner
            else:
                row += mdp.terrain_mtx[y][x]
        lines.append(f"  {row}")

    lines.append("Legend: Y=you, @=partner, X=counter, O=onion_disp, T=tomato_disp, "
                 "P=pot, S=serving, D=dish_disp, #=wall, ' '=floor")
    return "\n".join(lines)


def _serialize_players(state, agent_index):
    """Describe both players' positions, facing, and held objects."""
    lines = []
    player = state.players[agent_index]
    partner = state.players[1 - agent_index]

    facing = DIRECTION_NAMES.get(player.orientation, str(player.orientation))
    held = _describe_held(player)
    lines.append(f"YOU: pos={player.position}, facing={facing}, holding={held}")

    p_facing = DIRECTION_NAMES.get(partner.orientation, str(partner.orientation))
    p_held = _describe_held(partner)
    lines.append(f"PARTNER: pos={partner.position}, facing={p_facing}, holding={p_held}")

    return "\n".join(lines)


def _describe_held(player):
    """Describe what a player is holding."""
    if not player.has_object():
        return "nothing"
    obj = player.get_object()
    if obj.name == "soup":
        ingredients = obj.ingredients if hasattr(obj, "ingredients") else []
        return f"soup({', '.join(ingredients)})"
    return obj.name


def _serialize_pots(mdp, state):
    """Describe each pot's status."""
    pot_states = mdp.get_pot_states(state)
    recipe_context = _layout_recipe_context(mdp)
    lines = ["POTS:"]

    if not mdp.get_pot_locations():
        lines.append("  (no pots)")
        return "\n".join(lines)

    for pot_pos in mdp.get_pot_locations():
        if pot_pos in pot_states.get("empty", []):
            lines.append(f"  Pot at {pot_pos}: empty")
        elif state.has_object(pot_pos):
            soup = state.get_object(pot_pos)
            ingredients = soup.ingredients
            if soup.is_ready:
                lines.append(f"  Pot at {pot_pos}: READY to serve ({', '.join(ingredients)})")
            elif soup.is_cooking:
                remaining = soup.cook_time - soup._cooking_tick
                lines.append(f"  Pot at {pot_pos}: cooking {remaining} ticks left ({', '.join(ingredients)})")
            else:
                if len(ingredients) >= 3 and not _uses_old_dynamics(mdp):
                    lines.append(
                        f"  Pot at {pot_pos}: FULL (3/3) but NOT cooking ({', '.join(ingredients)}). "
                        f"INTERACT with empty hands to start cooking."
                    )
                else:
                    needed = 3 - len(ingredients)
                    if recipe_context["onion_only_three"]:
                        need_text = f"Needs {needed} more onion(s) before cooking can start."
                    elif recipe_context["tomato_only_three"]:
                        need_text = f"Needs {needed} more tomato(es) before cooking can start."
                    else:
                        need_text = f"Needs {needed} more ingredient(s) before cooking can start."
                    lines.append(
                        f"  Pot at {pot_pos}: has {len(ingredients)}/3 ingredients ({', '.join(ingredients)}). "
                        f"{need_text}"
                    )

    return "\n".join(lines)


def _serialize_counters(mdp, state):
    """Describe objects sitting on counters."""
    counter_objects = mdp.get_counter_objects_dict(state)
    lines = ["COUNTER OBJECTS:"]

    if not counter_objects:
        lines.append("  (none)")
    else:
        for obj_name, positions in counter_objects.items():
            lines.append(f"  {obj_name}: {positions}")

    return "\n".join(lines)


def _serialize_orders(state):
    """Describe current orders to fulfill."""
    lines = ["ORDERS:"]
    if state.all_orders:
        for order in state.all_orders:
            lines.append(f"  {order}")
    else:
        lines.append("  (no specific orders)")

    if state.bonus_orders:
        lines.append("BONUS ORDERS:")
        for order in state.bonus_orders:
            lines.append(f"  {order}")

    return "\n".join(lines)


def build_system_prompt(mdp, agent_index, horizon=None):
    """Build the one-time system prompt describing game rules and layout.

    Args:
        mdp: OvercookedGridworld instance
        agent_index: which player we are (0 or 1)
        horizon: total episode length

    Returns:
        str: system prompt
    """
    # Build terrain grid
    grid_lines = []
    for y in range(mdp.height):
        row = ""
        for x in range(mdp.width):
            row += mdp.terrain_mtx[y][x]
        grid_lines.append(row)
    grid_str = "\n".join(grid_lines)

    # Key locations
    locations = []
    for name, getter in [
        ("Pots", mdp.get_pot_locations),
        ("Onion dispensers", mdp.get_onion_dispenser_locations),
        ("Tomato dispensers", mdp.get_tomato_dispenser_locations),
        ("Dish dispensers", mdp.get_dish_dispenser_locations),
        ("Serving locations", mdp.get_serving_locations),
    ]:
        locs = getter()
        if locs:
            locations.append(f"  {name}: {locs}")

    locations_str = "\n".join(locations) if locations else "  (none listed)"

    horizon_str = f"\nThe episode lasts {horizon} timesteps." if horizon else ""
    recipe_context = _layout_recipe_context(mdp)
    layout_context = _layout_strategy_context(mdp)

    if _uses_old_dynamics(mdp):
        soup_pipeline = (
            "pick up ingredient (onion/tomato) from dispenser -> place in pot (3 needed) "
            "-> pot starts cooking automatically -> pick up a dish -> use dish on ready pot "
            "to get soup -> deliver soup to serving location"
        )
        cook_rule = "When pot has 3 ingredients, it starts cooking automatically."
    else:
        soup_pipeline = (
            "pick up ingredient (onion/tomato) from dispenser -> place in pot (3 needed) "
            "-> INTERACT with the full pot (while holding nothing) to start cooking "
            "-> pick up a dish -> use dish on ready pot to get soup -> deliver soup to serving location"
        )
        cook_rule = (
            "When pot has 3 ingredients it is FULL but idle; it will NOT cook until someone "
            "INTERACTs with that pot while holding nothing."
        )

        return f"""You are a chef in Overcooked. ...

GAME RULES:
- Make soups by: {soup_pipeline}
- {recipe_context["available_rule"]}
- {recipe_context["recipe_rule"]}
- CRITICAL: You can hold ONLY ONE ITEM at a time. Put something down before picking up anything else.
- INTERACT action: picks up items, places items, starts cooking, serves soup. You must be FACING the target square.
- To face a direction, move in that direction (even if blocked, your orientation updates).
- Coordinates are (x, y) where x increases rightward, y increases downward.{horizon_str}

LAYOUT-SPECIFIC GUIDANCE:
{layout_context}

ADJACENCY & INTERACT RULES (FOLLOW THESE EXACTLY):
- Compute Manhattan distance: |your_x - target_x| + |your_y - target_y|.
- If distance == 1 (you are ADJACENT) AND you are facing the target:
  → DO NOT MOVE AGAIN. Call INTERACT immediately.
- If distance == 1 but you are NOT facing the target:
  → Move ONCE to turn and face the target (even if blocked), then INTERACT.
- If distance > 1:
  → Move closer using the shortest path; ONLY once you become adjacent, switch to INTERACT.

STUCK / NO-MOVE RULE:
- If you choose a MOVE action and your position does NOT change on the next step:
  → Assume you are blocked or already adjacent.
  → On the very next step, either:
    - Try INTERACT (if target is adjacent), OR
    - Move in a different direction rather than repeating the same move.

PARTNER AWARENESS:
- If your partner is already adjacent to a pot or dispenser with the right item:
  → Prefer complementary tasks (e.g., get dishes, start cooking, or serve), instead of duplicating their movement.

...
"""

#     return f"""You are an AI chef in Overcooked, a cooperative cooking game. You are Player {agent_index}.

# RULES:
# - Make soups by: {soup_pipeline}
# - INTERACT action: picks up items, places items, starts interactions. You must be FACING the target square.
# - To face a direction, move in that direction (even if blocked, your orientation updates).
# - You share the kitchen with a partner. Coordinate to avoid blocking each other.
# - Coordinates are (x, y) where x increases rightward, y increases downward.{horizon_str}

# LAYOUT:
# {grid_str}
# Legend: X=counter, O=onion_disp, T=tomato_disp, D=dish_disp, S=serving, P=pot, ' '=floor

# KEY LOCATIONS:
# {locations_str}

# STRATEGY TIPS:
# - To pick up an onion: stand adjacent to an onion dispenser, face it, then INTERACT.
# - To place in pot: stand adjacent to a pot, face it, then INTERACT.
# - {cook_rule}
# - NEVER try to collect soup unless the pot status says READY.
# - When pot is ready: pick up a dish (from dish dispenser 'D' or a counter), stand adjacent to pot facing it, INTERACT to get soup.
# - Deliver soup: carry soup to a serving location, face it, INTERACT.

# Each turn you receive the current game state. You may use observation tools to gather info, then MUST call exactly one action tool to make your move."""


def build_planner_system_prompt(mdp, worker_ids, horizon=None):
    """Build the system prompt for the planner LLM.

    The planner coordinates multiple workers by assigning them complementary tasks.
    Workers cannot communicate with each other, so tasks must be self-contained.

    Args:
        mdp: OvercookedGridworld instance
        worker_ids: list of worker identifiers (e.g., ["worker_0", "worker_1"])
        horizon: total episode length

    Returns:
        str: system prompt for the planner
    """
    # Build terrain grid
    grid_lines = []
    for y in range(mdp.height):
        row = ""
        for x in range(mdp.width):
            row += mdp.terrain_mtx[y][x]
        grid_lines.append(row)
    grid_str = "\n".join(grid_lines)

    # Key locations
    locations = []
    for name, getter in [
        ("Pots", mdp.get_pot_locations),
        ("Onion dispensers", mdp.get_onion_dispenser_locations),
        ("Tomato dispensers", mdp.get_tomato_dispenser_locations),
        ("Dish dispensers", mdp.get_dish_dispenser_locations),
        ("Serving locations", mdp.get_serving_locations),
    ]:
        locs = getter()
        if locs:
            locations.append(f"  {name}: {locs}")

    locations_str = "\n".join(locations) if locations else "  (none listed)"

    horizon_str = f"\nThe episode lasts {horizon} timesteps." if horizon else ""
    recipe_context = _layout_recipe_context(mdp)
    layout_context = _layout_strategy_context(mdp)

    # Format worker list
    workers_str = "\n".join(f"  - {wid}" for wid in worker_ids)

    if _uses_old_dynamics(mdp):
        soup_pipeline = (
            "pick up ingredient (onion/tomato) from dispenser -> place in pot (3 needed) "
            "-> pot starts cooking automatically -> pick up a dish -> use dish on ready pot "
            "to get soup -> deliver soup to serving location"
        )
        cook_rule = "When a pot reaches 3 ingredients, it starts cooking automatically."
    else:
        soup_pipeline = (
            "pick up ingredient (onion/tomato) from dispenser -> place in pot (3 needed) "
            "-> INTERACT with the full pot (while holding nothing) to start cooking "
            "-> pick up a dish -> use dish on ready pot to get soup -> deliver soup to serving location"
        )
        cook_rule = (
            "A pot with 3/3 ingredients is NOT ready soup and does NOT cook by itself; "
            "assign a worker to start cooking via INTERACT."
        )

        return f"""You are the PLANNER in the cooperative cooking game Overcooked. Coordinate multiple workers to make and deliver soups as efficiently as possible.

1. Core Principles
Your planning must always be guided by these three principles:
- Maximize Efficiency: Minimize the total time required to complete all orders. This is the most critical goal.
- Maximize Parallelism: Ensure multiple agents are working simultaneously whenever possible to reduce idle time.
- Ensure Accuracy: Adhere 100% to all action definitions, rules, and constraints outlined below.

RULES:
- Each worker holds ONE item at a time.
- INTERACT picks up or drops items; must face the target square.
- Coordinates: (x,y), x increases right, y increases downward.
- Do not collect soup from a pot unless it is READY.
- {recipe_context["available_rule"]}
- {recipe_context["recipe_rule"]}
- Never assign "start cooking" unless the pot already contains the full required recipe.

LAYOUT:
{grid_str}
Legend: X=counter, O=onion_disp, T=tomato_disp, D=dish_disp, S=serving, P=pot, ' '=floor

KEY LOCATIONS:
{locations_str}

WORKERS:
{workers_str}

COORDINATION POLICY:
- Replace any generic role split with the layout-specific guidance below.

LAYOUT-SPECIFIC GUIDANCE:
{layout_context}

TASK GUIDELINES:
- Assign atomic tasks with explicit coordinates: e.g., "Go to onion dispenser at (2,1), pick onion, deliver to pot at (3,2)".
- Follow the layout-specific guidance above. If a worker cannot realistically reach a station on this layout, do not assign that task to them.
- Plan multi-step paths to targets, considering counters, obstacles, and the other worker's position.
- Reassign tasks dynamically if items, pots, or paths change.
- Pipeline tasks: prepare next soup while current soup cooks.
- Do not assign early dish pickup before a soup is cooking or nearly ready unless there is a concrete immediate reason.
- Do not assign a worker to hold a dish and idle while the current pot still needs ingredients.
- Do not assign `wait`, `stay ready`, or generic support tasks if a worker can instead fetch an ingredient, place an ingredient, start cooking, fetch a needed dish, or serve a ready soup.
- Do not assign vague counter-drop tasks unless the layout truly requires handoff counters. If the layout does not require a handoff, ingredients should usually go directly into the pot.
- If a worker is already holding a useful item, assign the next task so that worker finishes delivering or using that item instead of abandoning it.
- Prefer assigning path-clearing or repositioning moves to the worker who is not already carrying the critical item.
- Avoid idle time, overlapping paths, and collisions.

PRIORITY RULES:
1. If a soup is ready first complete delivery before doing anything else.
2. Keep pots cooking whenever possible.
3. Before a pot starts cooking, prioritize finishing the ingredient pipeline over staging dishes too early.
4. While soup is cooking, assign complementary prep tasks that fit the layout-specific access pattern.
5. Minimize walking distance.
6. Ensure each worker’s path is clear of obstacles and other workers.

TOOL USAGE:
- You must finish each planning turn by calling the `assign_tasks` tool.
- Pass both workers in a single tool call whenever possible.
- Use explicit short task strings for both workers, even if one worker should support or wait.
- Do not reply with plain JSON or prose instead of using the tool.
"""


def build_worker_system_prompt(mdp, agent_index, worker_id, horizon=None):
    """Build the system prompt for a worker LLM.

    Workers execute tasks assigned by the planner. They don't know about other workers.

    Args:
        mdp: OvercookedGridworld instance
        agent_index: which player we are (0 or 1)
        worker_id: identifier for this worker (e.g., "worker_0")
        horizon: total episode length

    Returns:
        str: system prompt for the worker
    """
    # Build terrain grid
    grid_lines = []
    for y in range(mdp.height):
        row = ""
        for x in range(mdp.width):
            row += mdp.terrain_mtx[y][x]
        grid_lines.append(row)
    grid_str = "\n".join(grid_lines)

    # Key locations
    locations = []
    for name, getter in [
        ("Pots", mdp.get_pot_locations),
        ("Onion dispensers", mdp.get_onion_dispenser_locations),
        ("Tomato dispensers", mdp.get_tomato_dispenser_locations),
        ("Dish dispensers", mdp.get_dish_dispenser_locations),
        ("Serving locations", mdp.get_serving_locations),
    ]:
        locs = getter()
        if locs:
            locations.append(f"  {name}: {locs}")

    locations_str = "\n".join(locations) if locations else "  (none listed)"

    horizon_str = f"\nThe episode lasts {horizon} timesteps." if horizon else ""
    recipe_context = _layout_recipe_context(mdp)
    layout_context = _layout_strategy_context(mdp)

    if _uses_old_dynamics(mdp):
        soup_pipeline = (
            "pick up ingredient (onion/tomato) from dispenser -> place in pot (3 needed) "
            "-> pot starts cooking automatically -> pick up a dish -> use dish on ready pot "
            "to get soup -> deliver soup to serving location"
        )
        cook_rule = "When pot has 3 ingredients, it starts cooking automatically."
    else:
        soup_pipeline = (
            "pick up ingredient (onion/tomato) from dispenser -> place in pot (3 needed) "
            "-> INTERACT with the full pot (while holding nothing) to start cooking "
            "-> pick up a dish -> use dish on ready pot to get soup -> deliver soup to serving location"
        )
        cook_rule = (
            "When pot has 3 ingredients it is FULL but idle. You must INTERACT with empty hands "
            "to start cooking."
        )

    return f"""You are {worker_id}, a chef in Overcooked. You are Player {agent_index}.

YOUR ROLE:
- Execute the task assigned to you by your coordinator
- Focus on completing your current task efficiently
- Navigate the kitchen and interact with objects to accomplish your goal

GAME RULES:
- Make soups by: {soup_pipeline}
- {recipe_context["available_rule"]}
- {recipe_context["recipe_rule"]}
- CRITICAL: You can hold ONLY ONE ITEM at a time. You must put down what you're holding (on a counter or in a pot) before picking up something else.
- INTERACT action: picks up items, places items, starts interactions. You must be FACING the target square.
- To face a direction, move in that direction (even if blocked, your orientation updates).
- Coordinates are (x, y) where x increases rightward, y increases downward.{horizon_str}

LAYOUT:
{grid_str}
Legend: X=counter, O=onion_disp, T=tomato_disp, D=dish_disp, S=serving, P=pot, ' '=floor

KEY LOCATIONS:
{locations_str}

LAYOUT-SPECIFIC GUIDANCE:
{layout_context}

ACTION GUIDE - ALWAYS CHECK IF YOU'RE ALREADY ADJACENT FIRST:
- **ADJACENT means your position differs by exactly 1 in X OR Y coordinate (not both)**
  - Example: You at (1,1), target at (0,1) → ADJACENT (X differs by 1)
  - Example: You at (3,1), target at (4,1) → ADJACENT (X differs by 1)
  - Example: You at (2,1), target at (2,0) → ADJACENT (Y differs by 1)
- To pick up an onion: **stand adjacent to dispenser** and face it, then INTERACT immediately
- To pick up a tomato: **stand adjacent to dispenser** and face it, then INTERACT immediately
- To place ingredient in pot: **stand adjacent to pot** and face it, then INTERACT immediately
- {cook_rule}
- Follow the layout-specific guidance above. If it says your side cannot reach a station, do not keep attempting that route.
- NEVER try to start cooking with only 1/3 or 2/3 ingredients in the pot.
- If pot shows 3/3 ingredients but not READY and not COOKING: go to pot with empty hands and INTERACT to start cooking.
- NEVER try to pick up soup unless pot status says READY.
- To get soup from ready pot: pick up a dish first, then **stand adjacent to pot** and face it before INTERACT
- To pick up a dish: **stand adjacent to dish dispenser** and face it, then INTERACT
- To deliver soup: **stand adjacent to serving location** and face it, then INTERACT

CRITICAL - CHECK ADJACENCY BEFORE EVERY MOVE:
Step 1: Calculate distance: |your_x - target_x| + |your_y - target_y|
Step 2: If distance == 1 → YOU ARE ADJACENT! Check facing direction, then INTERACT
Step 3: If distance > 1 → Move closer (use check_path to find route)

CONCRETE EXAMPLES (MEMORIZE THESE):
- You at (1,1), target at (0,1): |1-0| + |1-1| = 1 → ADJACENT! Face left, INTERACT
- You at (3,1), target at (4,1): |3-4| + |1-1| = 1 → ADJACENT! Face right, INTERACT
- You at (2,1), target at (2,0): |2-2| + |1-0| = 1 → ADJACENT! Face up, INTERACT
- You at (1,1), target at (3,3): |1-3| + |1-3| = 4 → NOT adjacent, need to move

DO NOT move onto the target square - you must INTERACT from the adjacent square!

NAVIGATION TIPS:
- **Use check_path() tool** to find the shortest route to your destination before moving
- You'll see another entity (@) in the kitchen - navigate around them if they're blocking your path
- **The layout has counters and walls** - you cannot move through them! Use check_path() to plan routes around obstacles
- If direct path is blocked, check_path() will tell you how many steps via the valid route
- If check_path() returns a large number of steps, there might be obstacles - plan accordingly
- Always ensure you're facing the correct direction before interacting

WORKFLOW FOR EACH TURN:
1. **Read the game state** - Check your current position and what you're holding
2. **Use observation tools** - Call get_surroundings() to see adjacent cells, or check_path() to find routes
3. **Plan your action** - Based on observations, decide the best move
4. **Execute ONE action** - Call exactly one action tool (move or interact)
5. **IMPORTANT**: If you tried to move but your position didn't change, you're BLOCKED! Try a different direction or route.

TASK FOLLOWING RULES:
- Follow the task from your CURRENT state, not from the beginning of the sentence every turn.
- If you already hold the needed item, skip the pickup part and continue to the delivery/dropoff part.
- If you already completed the pickup and dropoff, only continue to the next clause if it still matches the current state.
- Do not restart the task from the first clause after each step.
- If you are holding an onion and your task is to deliver to a pot, prioritize reaching the pot. Do not wander back toward the dispenser.
- If you are holding a dish and your task is to get soup from a ready pot, go straight toward the pot. Do not go back to the dish dispenser.
- If another worker is briefly blocking you, make one sensible sidestep and then continue the same task.
- Do not choose STAY unless you are truly waiting on cooking, waiting because the task explicitly says to wait, or every productive move is blocked.

Each turn you receive the current game state and your assigned task. Use observation tools first, then call exactly one action tool to make your move."""
