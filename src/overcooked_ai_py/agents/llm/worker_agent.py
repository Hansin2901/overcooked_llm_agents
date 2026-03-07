"""Worker agent that controls one chef on the floor."""

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
    """Worker agent that executes tasks from the shared planner."""

    def __init__(
        self,
        planner,
        worker_id: str,
        model_name: str = "gpt-4o",
        debug: bool = False,
        horizon: Optional[int] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        observability=None,
        invoke_config: Optional[dict] = None,
    ):
        self.planner = planner
        self.worker_id = worker_id
        self.model_name = model_name
        self.debug = debug
        self.horizon = horizon
        self.api_base = api_base
        self.api_key = api_key
        self.observability = observability
        self.invoke_config = dict(invoke_config or {})

        self._tool_state = ToolState()
        self._graph = None
        self._system_prompt = None
        super().__init__()

    def set_mdp(self, mdp):
        super().set_mdp(mdp)

        from overcooked_ai_py.planning.planners import MotionPlanner

        motion_planner = MotionPlanner(mdp)
        self._tool_state.init(mdp, motion_planner)
        self.planner.register_worker(self.worker_id, self._tool_state)

        self._system_prompt = build_worker_system_prompt(
            mdp, self.agent_index, self.worker_id, self.horizon
        )
        obs_tools, act_tools, act_names = create_worker_tools(self._tool_state)

        self._graph = build_react_graph(
            model_name=self.model_name,
            system_prompt=self._system_prompt,
            observation_tools=obs_tools,
            action_tools=act_tools,
            action_tool_names=act_names,
            get_chosen_fn=lambda: self._tool_state.chosen_action,
            debug=self.debug,
            debug_prefix=f"[{self.worker_id}]",
            api_base=self.api_base,
            api_key=self.api_key,
            observability=self.observability,
            role_name=self.worker_id,
        )

    def _safe_emit(self, event_type: str, payload: dict, step: int, agent_role: str):
        if self.observability is None:
            return
        try:
            self.observability.emit(
                event_type,
                payload,
                step=step,
                agent_role=agent_role,
            )
        except Exception as exc:
            if self.debug:
                print(f"  [{self.worker_id}] observability emit failed: {exc}")

    def action(self, state):
        """Choose an action based on the current task and game state."""
        self.planner.maybe_replan(state)
        if self.observability is not None:
            try:
                self.observability.start_role(self.worker_id)
            except Exception:
                pass

        try:
            task = self.planner.get_task(self.worker_id)
            task_text = task.description if task else "No task assigned. Wait for instructions."

            state_text = serialize_state(self.mdp, state, self.agent_index, self.horizon)
            self._tool_state.set_state(state, self.agent_index)

            prompt = (
                f"Your current task: {task_text}\n\n"
                f"Current game state:\n{state_text}\n\n"
                "Choose one action to execute your task."
            )
            messages = [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=prompt),
            ]

            if self.debug:
                print(f"  [{self.worker_id}] Invoking graph (step {state.timestep})...")

            try:
                invoke_config = {**self.invoke_config, "recursion_limit": 15}
                try:
                    self._graph.invoke({"messages": messages}, config=invoke_config)
                except TypeError as exc:
                    if "config" in str(exc) or "unexpected keyword argument" in str(exc):
                        self._graph.invoke({"messages": messages})
                    else:
                        raise
                if self.debug:
                    print(f"  [{self.worker_id}] Graph completed")
            except Exception as exc:
                if self.debug:
                    print(f"  [{self.worker_id}] Graph error: {exc}")
                self._tool_state.chosen_action = Action.STAY

            chosen = self._tool_state.chosen_action
            if chosen is None:
                if self.debug:
                    print(f"  [{self.worker_id}] No action chosen, defaulting to STAY")
                chosen = Action.STAY

            if task:
                task.steps_active += 1

            action_name = Action.ACTION_TO_CHAR.get(chosen, str(chosen))
            if self.debug:
                player = state.players[self.agent_index]
                held = player.held_object.name if player.held_object else "nothing"
                print(
                    f"  [Step {state.timestep}] {self.worker_id} at {player.position} "
                    f"holding {held} -> {action_name}"
                )

            self._tool_state.record_step(
                timestep=state.timestep,
                action=action_name,
                task=task.description if task else None,
            )
            self._safe_emit(
                "action.commit",
                {
                    "action": action_name,
                    "task": task.description if task else None,
                },
                step=state.timestep,
                agent_role=self.worker_id,
            )

            action_probs = self.a_probs_from_action(chosen)
            return chosen, {"action_probs": action_probs}
        finally:
            if self.observability is not None:
                try:
                    self.observability.end_role()
                except Exception:
                    pass

    def reset(self):
        super().reset()
        self._tool_state.reset()
        self._graph = None
        self._system_prompt = None
