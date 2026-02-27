"""Worker agent that controls one chef on the floor.

Receives tasks from a shared Planner. Produces one action per timestep.
Cannot see other workers' tasks or communicate with them.
"""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.agents.llm.graph_builder import build_react_graph
from overcooked_ai_py.agents.llm.state_serializer import (
    build_worker_system_prompt,
    serialize_state,
)
from overcooked_ai_py.agents.llm.tool_state import ToolState
from overcooked_ai_py.agents.llm.worker_tools import create_worker_tools
from overcooked_ai_py.mdp.actions import Action


class WorkerAgent(Agent):
    """Worker agent that controls one chef on the floor.

    Receives tasks from a shared Planner. Produces one action per timestep.
    Cannot see other workers' tasks or communicate with them.

    Args:
        planner: Shared Planner instance
        worker_id: This worker's ID (e.g., "worker_0")
        model_name: LiteLLM model for this worker's LLM
        debug: Print reasoning
        horizon: Total episode horizon
        api_base: Custom API endpoint
        api_key: Custom API key
    """

    def __init__(
        self,
        planner,
        worker_id: str,
        model_name: str = "gpt-4o",
        debug: bool = False,
        horizon: Optional[int] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.planner = planner
        self.worker_id = worker_id
        self.model_name = model_name
        self.debug = debug
        self.horizon = horizon
        self.api_base = api_base
        self.api_key = api_key

        self._tool_state = ToolState()
        self._graph = None
        self._system_prompt = None
        super().__init__()

    def set_mdp(self, mdp):
        """Initialize worker with MDP and register with planner.

        Args:
            mdp: OvercookedGridworld instance
        """
        super().set_mdp(mdp)

        # Initialize motion planner for this worker
        from overcooked_ai_py.planning.planners import MotionPlanner
        mp = MotionPlanner(mdp)

        # Initialize tool state
        self._tool_state.init(mdp, mp)

        # Register with planner
        self.planner.register_worker(self.worker_id, self._tool_state)

        # Build worker graph
        self._system_prompt = build_worker_system_prompt(
            mdp, self.agent_index, self.worker_id, self.horizon
        )

        obs_tools, act_tools, act_names = create_worker_tools(self._tool_state)

        self._graph = build_react_graph(
            self.model_name,
            self._system_prompt,
            obs_tools,
            act_tools,
            act_names,
            get_chosen_fn=lambda: self._tool_state.chosen_action,
            debug=self.debug,
            debug_prefix=f"[{self.worker_id}]",
            api_base=self.api_base,
            api_key=self.api_key,
        )

    def action(self, state):
        """Choose an action based on current task and game state.

        Args:
            state: Current OvercookedState

        Returns:
            (action, info_dict) where action is from Action.ALL_ACTIONS
        """
        # Step 1: Trigger planner if needed (first worker to call triggers it)
        self.planner.maybe_replan(state)

        # Step 2: Read this worker's task
        task = self.planner.get_task(self.worker_id)
        task_text = (
            task.description
            if task
            else "No task assigned. Wait for instructions."
        )

        # Step 3: Run worker LLM
        state_text = serialize_state(self.mdp, state, self.agent_index, self.horizon)
        self._tool_state.set_state(state, self.agent_index)

        prompt = (
            f"Your current task: {task_text}\n\n"
            f"Current game state:\n{state_text}\n\n"
            f"Choose one action to execute your task."
        )

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt),
        ]
        self._graph.invoke({"messages": messages})

        # Step 4: Get action
        chosen = self._tool_state.chosen_action
        if chosen is None:
            chosen = Action.STAY

        # Step 5: Track task progress
        if task:
            task.steps_active += 1

        if self.debug:
            action_name = Action.ACTION_TO_CHAR.get(chosen, str(chosen))
            print(
                f"  [Step {state.timestep}] {self.worker_id} "
                f"(Player {self.agent_index}) → {action_name}"
            )

        action_probs = self.a_probs_from_action(chosen)
        return chosen, {"action_probs": action_probs}

    def reset(self):
        """Reset worker state for a new episode."""
        super().reset()
        self._tool_state.reset()
        self._graph = None
        self._system_prompt = None
