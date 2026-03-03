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
    """Central planner that assigns tasks to worker agents.

    NOT an Agent — it doesn't produce actions.
    Shared by all WorkerAgents. Runs once per replan interval.

    Args:
        model_name: LiteLLM model string
        replan_interval: Steps between replanning (default: 5)
        debug: Print planner reasoning
        horizon: Total episode horizon
        api_base: Custom API endpoint
        api_key: Custom API key
    """

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

        self._tool_state = ToolState()  # Planner's own ToolState
        self._graph = None
        self._system_prompt = None
        self._worker_registry: dict[str, ToolState] = {}
        self._last_plan_step = -1  # Timestep of last planning

    def register_worker(self, worker_id: str, worker_tool_state: ToolState):
        """Register a worker. Called during setup.

        Args:
            worker_id: Unique identifier for the worker (e.g., "worker_0")
            worker_tool_state: The worker's ToolState instance
        """
        self._worker_registry[worker_id] = worker_tool_state

    def init(self, mdp, motion_planner):
        """Initialize planner after all workers are registered.

        Args:
            mdp: OvercookedGridworld instance
            motion_planner: MotionPlanner for distance calculations
        """
        self._tool_state.init(mdp, motion_planner)

        worker_ids = list(self._worker_registry.keys())
        self._system_prompt = build_planner_system_prompt(
            mdp, worker_ids, self.horizon
        )

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
        """Check if planner has assigned tasks (termination condition).

        Returns:
            True if any worker has a fresh task (steps_active == 0)
            None otherwise
        """
        for wid, ts in self._worker_registry.items():
            if ts.current_task and ts.current_task.steps_active == 0:
                return True
        return None

    def should_replan(self, state) -> bool:
        """Check if replanning is needed.

        Args:
            state: Current OvercookedState

        Returns:
            True if replanning should occur, False otherwise
        """
        # Don't replan twice in the same timestep
        if self._last_plan_step == state.timestep:
            return False

        # First step - always plan
        if self._last_plan_step < 0:
            return True

        # Check if interval has elapsed
        steps_since = state.timestep - self._last_plan_step
        if steps_since >= self.replan_interval:
            return True

        # Check if any worker is without a task or has completed
        for wid, ts in self._worker_registry.items():
            if ts.current_task is None or ts.current_task.completed:
                return True

        return False

    def maybe_replan(self, state):
        """Run planner if needed. Called by first worker each step.

        Args:
            state: Current OvercookedState
        """
        if not self.should_replan(state):
            return

        # Serialize state for the planner (use agent_index=0 for full view)
        state_text = serialize_state(self._tool_state.mdp, state, 0, self.horizon)
        self._tool_state.set_state(state, 0)

        # Include worker statuses
        statuses = []
        for wid, ts in self._worker_registry.items():
            status_dict = ts.get_status()
            if status_dict["status"] == "idle":
                statuses.append(f"  {wid} (Player {wid[-1]}): idle")
            else:
                statuses.append(
                    f"  {wid} (Player {wid[-1]}): {status_dict['status']} - "
                    f"{status_dict['task']} (active for {status_dict['steps_active']} steps)"
                )
        status_text = "\n".join(statuses)

        prompt = (
            f"Worker statuses:\n{status_text}\n\n"
            f"Current game state:\n{state_text}\n\n"
            f"Assign tasks to your workers."
        )

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt),
        ]

        # Invoke with recursion limit to prevent infinite loops
        # Planner may need more steps for complex reasoning
        if self.debug:
            print(f"  [Planner] Invoking graph (step {state.timestep})...")

        try:
            invoke_config = {**self.invoke_config, "recursion_limit": 20}
            try:
                self._graph.invoke(
                    {"messages": messages},
                    config=invoke_config,
                )
            except TypeError as e:
                # Unit tests may stub invoke(messages) without config support.
                if "config" in str(e) or "unexpected keyword argument" in str(e):
                    self._graph.invoke({"messages": messages})
                else:
                    raise
            if self.debug:
                print(f"  [Planner] Graph completed")
        except Exception as e:
            if self.debug:
                print(f"  [Planner] Graph error: {e}")

        self._last_plan_step = state.timestep
        assignments = {
            wid: (ts.current_task.description if ts.current_task else None)
            for wid, ts in self._worker_registry.items()
        }
        self._safe_emit(
            "planner.assignment",
            {"assignments": assignments},
            step=state.timestep,
            agent_role="planner",
        )

        if self.debug:
            for wid, ts in self._worker_registry.items():
                if ts.current_task:
                    print(f"  [Planner] → {wid}: {ts.current_task.description}")

    def get_task(self, worker_id: str) -> Optional[Task]:
        """Get a worker's current task. Workers call this to read their own task only.

        Args:
            worker_id: The worker identifier

        Returns:
            The worker's current task, or None if no task or unknown worker
        """
        ts = self._worker_registry.get(worker_id)
        if ts is None:
            return None
        return ts.current_task

    def reset(self):
        """Reset planner state for a new episode.

        Note: Worker registry is NOT cleared - workers persist across episodes
        and are registered once during setup via set_mdp().
        """
        self._tool_state.reset()
        self._graph = None
        self._last_plan_step = -1
        # Do NOT clear worker registry - workers persist across episodes
