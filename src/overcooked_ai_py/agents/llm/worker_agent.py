"""Worker agent that controls one chef on the floor.

Receives tasks from a shared Planner. Produces one action per timestep.
Cannot see other workers' tasks or communicate with them.
"""

import json
from typing import Optional

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.agents.llm.state_serializer import (
    build_worker_system_prompt,
    serialize_state,
)
from overcooked_ai_py.agents.llm.tool_state import ToolState
from overcooked_ai_py.mdp.actions import Action, Direction


# Action name to game action mapping
_ACTION_MAP = {
    "move_up": Direction.NORTH,
    "move_down": Direction.SOUTH,
    "move_left": Direction.WEST,
    "move_right": Direction.EAST,
    "interact": Action.INTERACT,
    "wait": Action.STAY,
}


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
        observability=None,
        invoke_config: Optional[dict] = None,
        history_size: int = 5,
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
        self.history_size = history_size
        self._history = []

        self._tool_state = ToolState()
        self._llm = None
        self._system_prompt = None
        super().__init__()

    def _parse_worker_action(self, text: str):
        """Parse strict JSON {"action":"..."} and return mapped action or None.

        Args:
            text: LLM response text containing JSON

        Returns:
            Action from _ACTION_MAP if valid, None otherwise
        """
        try:
            data = json.loads(text.strip())
            action_name = data.get("action")
            if action_name in _ACTION_MAP:
                return _ACTION_MAP[action_name]
            else:
                if self.debug:
                    print(f"  [{self.worker_id}] Unknown action: {action_name}")
                return None
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            if self.debug:
                print(f"  [{self.worker_id}] JSON parse error: {e}")
            return None

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

        # Build worker system prompt with JSON output instructions
        base_prompt = build_worker_system_prompt(
            mdp, self.agent_index, self.worker_id, self.horizon
        )

        # Add JSON output format instructions
        json_instructions = """

You must respond with ONLY a JSON object in this exact format:
{"action": "action_name"}

Valid actions:
- move_up: Move north (up)
- move_down: Move south (down)
- move_left: Move west (left)
- move_right: Move east (right)
- interact: Pick up/drop items, use counters/dispensers
- wait: Stay in place

Do NOT use tool calls. Output ONLY the JSON object."""

        self._system_prompt = base_prompt + json_instructions

        # Initialize LLM
        llm_kwargs = {
            "model": self.model_name,
            "temperature": 0.2,
            "timeout": 30.0,
        }
        if self.api_base:
            llm_kwargs["api_base"] = self.api_base
        if self.api_key:
            llm_kwargs["api_key"] = self.api_key

        self._llm = ChatLiteLLM(**llm_kwargs)

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

    def _add_to_history(self, timestep, position, action, held, task_description):
        """Record a compact history entry after each action."""
        if self.history_size <= 0:
            return
        action_name = Action.ACTION_TO_CHAR.get(action, str(action))
        self._history.append(
            {
                "timestep": timestep,
                "position": position,
                "action": action_name,
                "held": held,
                "task": task_description,
            }
        )
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size :]

    def _format_history(self):
        """Format history entries for injection into the worker prompt.

        Uses compact format with task-boundary markers:
            RECENT HISTORY:
            - Step 12: at (2,1) -> ^ [Task: Pick up onion]
            --- New task ---
            - Step 14: at (2,0) holding onion -> > [Task: Deliver to pot]
        """
        if not self._history or self.history_size <= 0:
            return ""

        lines = ["RECENT HISTORY:"]
        prev_task = None
        for entry in self._history:
            if prev_task is not None and entry["task"] != prev_task:
                lines.append("--- New task ---")
            held_str = f" holding {entry['held']}" if entry["held"] != "nothing" else ""
            lines.append(
                f"- Step {entry['timestep']}: at {entry['position']}{held_str} "
                f"-> {entry['action']} [Task: {entry['task']}]"
            )
            prev_task = entry["task"]
        return "\n".join(lines)

    def action(self, state):
        """Choose an action based on current task and game state.

        Args:
            state: Current OvercookedState

        Returns:
            (action, info_dict) where action is from Action.ALL_ACTIONS
        """
        # Step 1: Trigger planner if needed (first worker to call triggers it)
        self.planner.maybe_replan(state)
        if self.observability is not None:
            try:
                self.observability.start_role(self.worker_id)
            except Exception:
                pass

        try:
            # Step 2: Read this worker's task
            task = self.planner.get_task(self.worker_id)
            task_text = (
                task.description if task else "No task assigned. Wait for instructions."
            )

            # Step 3: Run worker LLM
            state_text = serialize_state(
                self.mdp, state, self.agent_index, self.horizon
            )
            self._tool_state.set_state(state, self.agent_index)

            # Build history text
            history_text = self._format_history()

            # Build prompt with basic context
            if history_text:
                prompt = (
                    f"{history_text}\n\n"
                    f"Your current task: {task_text}\n\n"
                    f"Current game state:\n{state_text}\n\n"
                    f"Respond with your action in JSON format."
                )
            else:
                prompt = (
                    f"Your current task: {task_text}\n\n"
                    f"Current game state:\n{state_text}\n\n"
                    f"Respond with your action in JSON format."
                )

            messages = [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=prompt),
            ]

            # ONE-SHOT LLM INVOCATION: Single call with JSON action output
            if self.debug:
                print(f"  [{self.worker_id}] Invoking LLM (step {state.timestep})...")

            try:
                response = self._llm.invoke(messages)
                if self.debug:
                    print(
                        f"  [{self.worker_id}] LLM response: {str(response.content)[:100]}..."
                    )

                # Parse action from JSON response
                # Handle both string and list content types
                content = response.content
                if isinstance(content, list):
                    # If content is a list, join it into a string
                    content = " ".join(str(c) for c in content)

                chosen = self._parse_worker_action(str(content))
                if chosen is None:
                    if self.debug:
                        print(
                            f"  [{self.worker_id}] Failed to parse action, defaulting to STAY"
                        )
                    chosen = Action.STAY

            except Exception as e:
                if self.debug:
                    print(f"  [{self.worker_id}] LLM error: {e}")
                # On error, default to STAY
                chosen = Action.STAY

            # Record history
            player = state.players[self.agent_index]
            held = player.held_object.name if player.held_object else "nothing"
            self._add_to_history(
                timestep=state.timestep,
                position=player.position,
                action=chosen,
                held=held,
                task_description=task_text,
            )

            # Step 5: Track task progress
            if task:
                task.steps_active += 1

            if self.debug:
                action_name = Action.ACTION_TO_CHAR.get(chosen, str(chosen))
                player = state.players[self.agent_index]
                held = player.held_object.name if player.held_object else "nothing"
                print(
                    f"  [Step {state.timestep}] {self.worker_id} at {player.position} "
                    f"holding {held} → {action_name}"
                )
            action_name = Action.ACTION_TO_CHAR.get(chosen, str(chosen))
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
        """Reset worker state for a new episode."""
        super().reset()
        self._tool_state.reset()
        self._llm = None
        self._system_prompt = None
        self._history = []
