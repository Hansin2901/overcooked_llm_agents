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

    def __init__(self, model_name="gpt-4o", debug=False, horizon=None):
        self.model_name = model_name
        self.debug = debug
        self.horizon = horizon
        self._graph = None
        self._system_prompt = None
        super().__init__()

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
