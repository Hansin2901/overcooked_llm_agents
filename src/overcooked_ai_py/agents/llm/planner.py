"""Central planner that assigns tasks to worker agents."""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from overcooked_ai_py.agents.llm.graph_builder import build_react_graph
from overcooked_ai_py.agents.llm.planner_tools import create_planner_tools
from overcooked_ai_py.agents.llm.state_serializer import (
    build_planner_system_prompt,
    serialize_state,
)
from overcooked_ai_py.agents.llm.task import Task
from overcooked_ai_py.agents.llm.tool_state import ToolState


class Planner:
    """Central planner that assigns tasks to worker agents."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        replan_interval: int = 5,
        debug: bool = False,
        horizon: Optional[int] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        observability=None,
        invoke_config: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.replan_interval = replan_interval
        self.debug = debug
        self.horizon = horizon
        self.api_base = api_base
        self.api_key = api_key
        self.observability = observability
        self.invoke_config = dict(invoke_config or {})

        self._tool_state = ToolState()
        self._graph = None
        self._system_prompt = None
        self._worker_registry: dict[str, ToolState] = {}
        self._last_plan_step = -1
        self._history_size = 5

    def register_worker(self, worker_id: str, worker_tool_state: ToolState):
        self._worker_registry[worker_id] = worker_tool_state

    def init(self, mdp, motion_planner):
        self._tool_state.init(mdp, motion_planner)

        worker_ids = list(self._worker_registry.keys())
        self._system_prompt = build_planner_system_prompt(mdp, worker_ids, self.horizon)

        obs_tools, act_tools, act_names = create_planner_tools(
            self._tool_state, self._worker_registry
        )

        self._graph = build_react_graph(
            model_name=self.model_name,
            system_prompt=self._system_prompt,
            observation_tools=obs_tools,
            action_tools=act_tools,
            action_tool_names=act_names,
            get_chosen_fn=lambda: self._tasks_assigned(),
            debug=self.debug,
            debug_prefix="[Planner]",
            api_base=self.api_base,
            api_key=self.api_key,
            observability=self.observability,
            role_name="planner",
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
                print(f"  [Planner] observability emit failed: {exc}")

    def _tasks_assigned(self) -> Optional[bool]:
        for tool_state in self._worker_registry.values():
            if tool_state.current_task and tool_state.current_task.steps_active == 0:
                return True
        return None

    def should_replan(self, state) -> bool:
        if self._last_plan_step == state.timestep:
            return False
        if self._last_plan_step < 0:
            return True
        if state.timestep - self._last_plan_step >= self.replan_interval:
            return True
        for tool_state in self._worker_registry.values():
            if tool_state.current_task is None or tool_state.current_task.completed:
                return True
        return False

    def _build_status_text(self) -> str:
        lines = []
        for worker_id, tool_state in self._worker_registry.items():
            status = tool_state.get_status()
            if status["status"] == "idle":
                lines.append(f"  {worker_id} (Player {worker_id[-1]}): idle")
            else:
                lines.append(
                    f"  {worker_id} (Player {worker_id[-1]}): {status['status']} - "
                    f"{status['task']} (active for {status['steps_active']} steps)"
                )
        return "\n".join(lines) if lines else "  (none)"

    def _build_history_block(self) -> str:
        all_timesteps = set()
        for tool_state in self._worker_registry.values():
            for entry in tool_state.history:
                timestep = entry.get("timestep")
                if timestep is not None:
                    all_timesteps.add(int(timestep))

        if not all_timesteps:
            return ""

        lines = ["Recent worker history:"]
        for timestep in sorted(all_timesteps)[-self._history_size :]:
            step_parts = []
            for worker_id, tool_state in self._worker_registry.items():
                entries = [
                    entry
                    for entry in tool_state.history
                    if int(entry.get("timestep", -1)) == timestep
                ]
                if not entries:
                    continue
                entry = entries[-1]
                step_parts.append(
                    f"{worker_id}: action={entry.get('action', '?')}, "
                    f"task={entry.get('task') or 'none'}"
                )
            if step_parts:
                lines.append(f"  Step {timestep}: " + " | ".join(step_parts))
        return "\n".join(lines) + "\n\n"

    def maybe_replan(self, state):
        """Run planner if needed. Called by the first worker each step."""
        if not self.should_replan(state):
            return

        try:
            if self.observability is not None:
                try:
                    self.observability.start_role("planner")
                except Exception:
                    pass

            self._tool_state.set_state(state, 0)
            state_text = serialize_state(self._tool_state.mdp, state, 0, self.horizon)
            status_text = self._build_status_text()
            history_block = self._build_history_block()

            prompt = (
                f"{history_block}"
                f"Worker statuses:\n{status_text}\n\n"
                f"Current game state:\n{state_text}\n\n"
                "Assign complementary tasks for both workers. Finish by calling the "
                "assign_tasks tool with both worker_0 and worker_1."
            )
            messages = [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=prompt),
            ]

            if self.debug:
                print(f"  [Planner] Invoking graph (step {state.timestep})...")

            try:
                invoke_config = {**self.invoke_config, "recursion_limit": 20}
                try:
                    self._graph.invoke({"messages": messages}, config=invoke_config)
                except TypeError as exc:
                    if "config" in str(exc) or "unexpected keyword argument" in str(exc):
                        self._graph.invoke({"messages": messages})
                    else:
                        raise
                if self.debug:
                    print("  [Planner] Graph completed")
            except Exception as exc:
                if self.debug:
                    print(f"  [Planner] Graph error: {exc}")

            self._last_plan_step = state.timestep
            assignments = {
                worker_id: (
                    tool_state.current_task.description if tool_state.current_task else None
                )
                for worker_id, tool_state in self._worker_registry.items()
            }
            self._safe_emit(
                "planner.assignment",
                {"assignments": assignments},
                step=state.timestep,
                agent_role="planner",
            )

            if self.debug:
                for worker_id, tool_state in self._worker_registry.items():
                    task = tool_state.current_task.description if tool_state.current_task else "NO TASK"
                    print(f"  [Planner] -> {worker_id}: {task}")
        finally:
            if self.observability is not None:
                try:
                    self.observability.end_role()
                except Exception:
                    pass

    def get_task(self, worker_id: str) -> Optional[Task]:
        tool_state = self._worker_registry.get(worker_id)
        if tool_state is None:
            return None
        return tool_state.current_task

    def reset(self):
        self._tool_state.reset()
        self._graph = None
        self._last_plan_step = -1
