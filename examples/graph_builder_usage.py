"""Example usage of the generic graph_builder for workers and planner.

This demonstrates how to use build_react_graph() for both:
1. Worker agents (with worker_tools)
2. Planner agent (with planner_tools)
"""

from overcooked_ai_py.agents.llm.graph_builder import build_react_graph
from overcooked_ai_py.agents.llm.tool_state import ToolState
from overcooked_ai_py.agents.llm.worker_tools import create_worker_tools
from overcooked_ai_py.agents.llm.planner_tools import create_planner_tools


def create_worker_graph_example(
    worker_id: str,
    tool_state: ToolState,
    model_name: str,
    system_prompt: str,
    debug: bool = False,
):
    """Example: Create a graph for a worker agent.

    Args:
        worker_id: Worker identifier (e.g., 'worker_0')
        tool_state: The worker's ToolState instance
        model_name: LLM model to use
        system_prompt: System prompt with game rules
        debug: Enable debug output

    Returns:
        Compiled LangGraph for the worker
    """
    # Create worker-specific tools
    observation_tools, action_tools, action_tool_names = create_worker_tools(tool_state)

    # Build the graph
    graph = build_react_graph(
        model_name=model_name,
        system_prompt=system_prompt,
        observation_tools=observation_tools,
        action_tools=action_tools,
        action_tool_names=action_tool_names,
        get_chosen_fn=lambda: tool_state.chosen_action,
        debug=debug,
        debug_prefix=f"[{worker_id}]",
    )

    return graph


def create_planner_graph_example(
    planner_tool_state: ToolState,
    worker_registry: dict[str, ToolState],
    model_name: str,
    system_prompt: str,
    debug: bool = False,
):
    """Example: Create a graph for the planner agent.

    Args:
        planner_tool_state: The planner's ToolState instance
        worker_registry: Dict mapping worker_id -> worker's ToolState
        model_name: LLM model to use
        system_prompt: System prompt with planning rules
        debug: Enable debug output

    Returns:
        Compiled LangGraph for the planner
    """
    # Create planner-specific tools
    observation_tools, action_tools, action_tool_names = create_planner_tools(
        planner_tool_state, worker_registry
    )

    # Helper to check if all workers have tasks assigned
    def tasks_assigned() -> bool:
        """Check if all workers have been assigned tasks."""
        return all(
            worker_state.current_task is not None
            for worker_state in worker_registry.values()
        )

    # Build the graph
    graph = build_react_graph(
        model_name=model_name,
        system_prompt=system_prompt,
        observation_tools=observation_tools,
        action_tools=action_tools,
        action_tool_names=action_tool_names,
        get_chosen_fn=tasks_assigned,
        debug=debug,
        debug_prefix="[Planner]",
    )

    return graph


def example_usage():
    """Full example showing how to set up both planner and workers."""
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.planning.planners import MotionPlanner

    # Setup game environment
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    motion_planner = MotionPlanner.from_pickle_or_compute(
        mdp, counter_goals=[], counter_drop=[], counter_pickup=[], cook_time=20
    )

    # Create tool states for workers
    worker_0_state = ToolState()
    worker_0_state.init(mdp, motion_planner)

    worker_1_state = ToolState()
    worker_1_state.init(mdp, motion_planner)

    # Create tool state for planner
    planner_state = ToolState()
    planner_state.init(mdp, motion_planner)

    # Worker registry (planner uses this to assign tasks)
    worker_registry = {
        "worker_0": worker_0_state,
        "worker_1": worker_1_state,
    }

    # Model configuration
    model_name = "gpt-4o"
    debug = True

    # Create planner graph
    planner_system_prompt = """You are a task planner for an Overcooked game.
Analyze the game state and assign tasks to workers to maximize efficiency."""

    planner_graph = create_planner_graph_example(
        planner_tool_state=planner_state,
        worker_registry=worker_registry,
        model_name=model_name,
        system_prompt=planner_system_prompt,
        debug=debug,
    )

    # Create worker graphs
    worker_0_system_prompt = """You are worker_0 in an Overcooked game.
Execute your assigned task by choosing appropriate actions."""

    worker_0_graph = create_worker_graph_example(
        worker_id="worker_0",
        tool_state=worker_0_state,
        model_name=model_name,
        system_prompt=worker_0_system_prompt,
        debug=debug,
    )

    worker_1_system_prompt = """You are worker_1 in an Overcooked game.
Execute your assigned task by choosing appropriate actions."""

    worker_1_graph = create_worker_graph_example(
        worker_id="worker_1",
        tool_state=worker_1_state,
        model_name=model_name,
        system_prompt=worker_1_system_prompt,
        debug=debug,
    )

    print("✓ Planner graph created successfully")
    print("✓ Worker 0 graph created successfully")
    print("✓ Worker 1 graph created successfully")

    return planner_graph, worker_0_graph, worker_1_graph


if __name__ == "__main__":
    print("=" * 60)
    print("GRAPH BUILDER USAGE EXAMPLE")
    print("=" * 60)
    print()

    try:
        planner, w0, w1 = example_usage()
        print()
        print("=" * 60)
        print("SUCCESS: All graphs created successfully!")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"ERROR: {e}")
        print("=" * 60)
        raise
