"""Serialize OvercookedState into concise text for LLM consumption."""

from overcooked_ai_py.mdp.actions import Direction


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
                lines.append(f"  Pot at {pot_pos}: has {len(ingredients)}/3 ingredients ({', '.join(ingredients)})")

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

    return f"""You are an AI chef in Overcooked, a cooperative cooking game. You are Player {agent_index}.

RULES:
- Make soups by: pick up ingredient (onion/tomato) from dispenser -> place in pot (3 needed) -> pot cooks automatically -> pick up a dish -> use dish on ready pot to get soup -> deliver soup to serving location
- INTERACT action: picks up items, places items, starts interactions. You must be FACING the target square.
- To face a direction, move in that direction (even if blocked, your orientation updates).
- You share the kitchen with a partner. Coordinate to avoid blocking each other.
- Coordinates are (x, y) where x increases rightward, y increases downward.{horizon_str}

LAYOUT:
{grid_str}
Legend: X=counter, O=onion_disp, T=tomato_disp, D=dish_disp, S=serving, P=pot, ' '=floor

KEY LOCATIONS:
{locations_str}

STRATEGY TIPS:
- To pick up an onion: stand adjacent to an onion dispenser, face it, then INTERACT.
- To place in pot: stand adjacent to a pot, face it, then INTERACT.
- When pot has 3 ingredients, it starts cooking automatically.
- When pot is ready: pick up a dish (from dish dispenser 'D' or a counter), stand adjacent to pot facing it, INTERACT to get soup.
- Deliver soup: carry soup to a serving location, face it, INTERACT.

Each turn you receive the current game state. You may use observation tools to gather info, then MUST call exactly one action tool to make your move."""
