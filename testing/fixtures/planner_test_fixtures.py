"""Test fixtures for planner tests."""

from overcooked_ai_py.mdp.overcooked_mdp import (
    Direction,
    ObjectState,
    OvercookedGridworld,
    PlayerState,
    SoupState,
)


def create_worker_at_dispenser(
    mdp: OvercookedGridworld,
    worker_index: int = 0,
    dispenser_type: str = "onion",
):
    """Create game state with worker positioned at dispenser."""
    state = mdp.get_standard_start_state()

    if dispenser_type == "onion":
        locations = mdp.get_onion_dispenser_locations()
    elif dispenser_type == "tomato":
        locations = mdp.get_tomato_dispenser_locations()
    elif dispenser_type == "dish":
        locations = mdp.get_dish_dispenser_locations()
    else:
        raise ValueError(f"Unknown dispenser type: {dispenser_type}")

    if not locations:
        raise ValueError(f"No {dispenser_type} dispenser in this layout")

    dispenser_pos = locations[0]
    players = list(state.players)
    players[worker_index] = PlayerState(dispenser_pos, Direction.NORTH)
    state.players = tuple(players)
    return state


def create_worker_holding_onion(
    mdp: OvercookedGridworld,
    worker_index: int = 0,
    position: tuple = None,
):
    """Create game state with worker holding an onion."""
    state = mdp.get_standard_start_state()

    if position is None:
        position = state.players[worker_index].position

    onion = ObjectState("onion", position)
    players = list(state.players)
    players[worker_index] = PlayerState(position, Direction.NORTH, held_object=onion)
    state.players = tuple(players)
    return state


def create_worker_holding_dish(
    mdp: OvercookedGridworld,
    worker_index: int = 0,
    position: tuple = None,
):
    """Create game state with worker holding a dish."""
    state = mdp.get_standard_start_state()

    if position is None:
        position = state.players[worker_index].position

    dish = ObjectState("dish", position)
    players = list(state.players)
    players[worker_index] = PlayerState(position, Direction.NORTH, held_object=dish)
    state.players = tuple(players)
    return state


def create_pot_with_ingredients(
    mdp: OvercookedGridworld,
    num_onions: int = 1,
    num_tomatoes: int = 0,
    cooking: bool = False,
):
    """Create game state with pot containing ingredients."""
    state = mdp.get_standard_start_state()

    pot_locations = mdp.get_pot_locations()
    if not pot_locations:
        raise ValueError("No pot in this layout")

    pot_pos = pot_locations[0]
    cooking_tick = 0 if not cooking else 1

    soup = SoupState.get_soup(
        pot_pos,
        num_onions=num_onions,
        num_tomatoes=num_tomatoes,
        cooking_tick=cooking_tick,
        cook_time=20,
    )
    state.objects[pot_pos] = soup
    return state


def create_ready_soup(mdp: OvercookedGridworld):
    """Create game state with ready soup in pot."""
    state = mdp.get_standard_start_state()

    pot_locations = mdp.get_pot_locations()
    if not pot_locations:
        raise ValueError("No pot in this layout")

    pot_pos = pot_locations[0]

    soup = SoupState.get_soup(
        pot_pos,
        num_onions=3,
        num_tomatoes=0,
        cooking_tick=20,
        cook_time=20,
    )
    state.objects[pot_pos] = soup
    return state
