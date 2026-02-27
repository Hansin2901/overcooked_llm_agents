"""Generic ReAct graph builder for LangGraph agents.

Shared by both planner and worker agents. The graph structure is:
  START -> llm_node -> route_after_llm -> {obs_tools, action_tools, end}
    obs_tools -> llm_node (loop)
    action_tools -> END
    end -> END
"""

from typing import Annotated, Callable, TypedDict

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    """State for the agent graph."""

    messages: Annotated[list, add_messages]


def build_react_graph(
    model_name: str,
    system_prompt: str,
    observation_tools: list,
    action_tools: list,
    action_tool_names: set[str],
    get_chosen_fn: Callable,
    debug: bool = False,
    debug_prefix: str = "[LLM]",
    api_base: str = None,
    api_key: str = None,
):
    """Build a ReAct LangGraph.

    Same structure as current graph.py:
    START → llm → route → {obs_tools → llm (loop), action_tools → END, end → END}

    Args:
        model_name: LiteLLM model string (e.g., "gpt-4o", "anthropic/claude-sonnet-4-20250514")
        system_prompt: One-time system prompt with game rules
        observation_tools: Tools that don't terminate the loop
        action_tools: Tools that terminate the loop
        action_tool_names: Set of action tool names for routing
        get_chosen_fn: Callable that returns truthy when graph should terminate.
            For workers: lambda: tool_state.chosen_action
            For planner: lambda: <check if tasks were assigned>
        debug: Print LLM reasoning
        debug_prefix: Prefix for debug output (e.g. "[Planner]" or "[worker_0]")
        api_base: Custom endpoint config
        api_key: Custom endpoint config

    Returns:
        Compiled LangGraph StateGraph
    """
    # Initialize LLM via LiteLLM
    llm_kwargs = {"model": model_name, "temperature": 0.2}
    if api_base:
        llm_kwargs["api_base"] = api_base
    if api_key:
        llm_kwargs["api_key"] = api_key

    llm = ChatLiteLLM(**llm_kwargs)
    all_tools = observation_tools + action_tools
    llm_with_tools = llm.bind_tools(all_tools)

    # Tool execution node
    tool_node = ToolNode(all_tools)

    def llm_node(state: AgentState) -> dict:
        """Call the LLM with current messages."""
        messages = state["messages"]

        # Prepend system prompt if not already there
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        response = llm_with_tools.invoke(messages)

        if debug and response.content:
            print(f"  {debug_prefix} {response.content[:200]}")

        return {"messages": [response]}

    def route_after_llm(state: AgentState) -> str:
        """Route based on the LLM's last message."""
        last_message = state["messages"][-1]

        # No tool calls -> default to wait (END)
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"

        # Check if any tool call is an action tool
        for tc in last_message.tool_calls:
            if tc["name"] in action_tool_names:
                return "action_tools"

        # Only observation tools
        return "obs_tools"

    # Build the graph
    graph = StateGraph(AgentState)

    graph.add_node("llm", llm_node)
    graph.add_node("obs_tools", tool_node)
    graph.add_node("action_tools", tool_node)

    graph.add_edge(START, "llm")

    graph.add_conditional_edges(
        "llm",
        route_after_llm,
        {
            "obs_tools": "obs_tools",
            "action_tools": "action_tools",
            "end": END,
        },
    )

    # After observation tools, go back to LLM
    graph.add_edge("obs_tools", "llm")

    # After action tools, end
    graph.add_edge("action_tools", END)

    return graph.compile()
