"""LLMAgent: an Agent that uses an LLM to decide actions each timestep."""

import numpy as np
from dotenv import load_dotenv

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.agents.llm.graph import build_graph
from overcooked_ai_py.agents.llm.state_serializer import build_system_prompt, serialize_state
from overcooked_ai_py.agents.llm.tools import (
    get_chosen_action,
    init_tools,
    set_state,
)
from overcooked_ai_py.mdp.actions import Action

# Load .env file for API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
load_dotenv()


class LLMAgent(Agent):
    """Agent that uses an LLM (via LiteLLM + LangGraph) to decide actions.

    Each timestep:
      1. Serialize the game state to text
      2. Run a LangGraph ReAct loop where the LLM can call observation tools
         and must call exactly one action tool
      3. Return the chosen action

    Args:
        model_name: any LiteLLM-compatible model string
            e.g. "gpt-4o", "anthropic/claude-sonnet-4-20250514", "openai/gpt-4o-mini"
        debug: if True, print LLM reasoning and actions each step
        horizon: episode length (for display in state serialization)
    """

    def __init__(self, model_name="gpt-4o", debug=False, horizon=None, history_size=10):
        self.model_name = model_name
        self.debug = debug
        self.horizon = horizon
        self.history_size = history_size
        self._history = []
        self._graph = None
        self._system_prompt = None
        super().__init__()

    def _format_history(self):
        """Format history entries for display to LLM."""
        if not self._history or self.history_size == 0:
            return ""

        lines = ["RECENT HISTORY:"]
        for entry in self._history:
            lines.append(
                f"- Step {entry['timestep']}: \"{entry['reasoning']}\" → {entry['action']}"
            )
        return "\n".join(lines)

    def _add_to_history(self, timestep, reasoning, action):
        """Add entry to history and maintain size limit."""
        from overcooked_ai_py.mdp.actions import Action

        action_name = Action.ACTION_TO_CHAR.get(action, str(action))

        self._history.append({
            "timestep": timestep,
            "reasoning": reasoning,
            "action": action_name,
        })

        # Trim to history_size
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]

    def _extract_reasoning(self, messages):
        """Extract reasoning text from LLM's final decision.

        Returns the content of the AIMessage that called an action tool,
        or a descriptive fallback if extraction fails.
        """
        from langchain_core.messages import AIMessage
        from overcooked_ai_py.agents.llm.tools import ACTION_TOOL_NAMES

        if not messages:
            return "(no messages returned)"

        try:
            # Look backwards for AIMessage with action tool call
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    # Check if this message called an action tool
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc["name"] in ACTION_TOOL_NAMES:
                                # Found action decision
                                return msg.content.strip() if msg.content else "(no reasoning provided)"
                    # Fallback to any AIMessage content
                    elif msg.content:
                        return msg.content.strip()

            return "(no reasoning found)"
        except Exception as e:
            if self.debug:
                print(f"  [LLMAgent] Reasoning extraction failed: {e}")
            return "(extraction failed)"

    def set_mdp(self, mdp):
        """Initialize serializer, tools, and build LangGraph.

        Called by AgentPair/AgentGroup before the episode starts.
        """
        super().set_mdp(mdp)

        # Build motion planner for tools (distance queries)
        from overcooked_ai_py.planning.planners import MotionPlanner

        mp = MotionPlanner(mdp)
        init_tools(mdp, mp)

        # Build system prompt (once per episode / layout)
        self._system_prompt = build_system_prompt(mdp, self.agent_index, self.horizon)

        # Build LangGraph
        self._graph = build_graph(
            model_name=self.model_name,
            system_prompt=self._system_prompt,
            debug=self.debug,
        )

    def action(self, state):
        """Decide an action for the current state.

        Args:
            state: OvercookedState

        Returns:
            (action, action_info): action is one of Action.ALL_ACTIONS,
                action_info is a dict with metadata
        """
        # Serialize state to text
        state_text = serialize_state(self.mdp, state, self.agent_index, self.horizon)

        # Update tool context
        set_state(state, self.agent_index)

        # Run LangGraph agent
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=f"Current game state:\n{state_text}\n\nDecide your action."),
        ]

        result = self._graph.invoke({"messages": messages})

        # Extract the action from the tool module
        chosen = get_chosen_action()
        if chosen is None:
            # LLM didn't call an action tool — default to STAY
            if self.debug:
                print(f"  [LLMAgent] No action tool called, defaulting to STAY")
            chosen = Action.STAY

        if self.debug:
            action_name = Action.ACTION_TO_CHAR.get(chosen, str(chosen))
            print(f"  [Step {state.timestep}] Player {self.agent_index} -> {action_name}")

        action_probs = self.a_probs_from_action(chosen)
        return chosen, {"action_probs": action_probs}

    def reset(self):
        """Reset agent state between episodes."""
        super().reset()
        self._graph = None
        self._system_prompt = None
