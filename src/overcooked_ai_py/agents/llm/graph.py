"""LangGraph ReAct agent for Overcooked.

Uses LangGraph StateGraph with a tool-calling loop:
  START -> llm_node -> {tool_calls?}
    yes, observation tool -> tool_node -> llm_node (loop)
    yes, action tool     -> tool_node -> END
    no tool call         -> default wait -> END
"""

from typing import Annotated, TypedDict

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from overcooked_ai_py.agents.llm.tools import (
    ACTION_TOOL_NAMES,
    ALL_TOOLS,
    get_chosen_action,
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(
    model_name: str,
    system_prompt: str,
    debug: bool = False,
    api_base: str = None,
    api_key: str = None,
    observability=None,
    role_name: str = "llm_agent",
):
    """Build and compile the LangGraph agent.

    Args:
        model_name: LiteLLM-compatible model string (e.g. "gpt-4o", "anthropic/claude-sonnet-4-20250514")
        system_prompt: the one-time system prompt with game rules
        debug: if True, print LLM reasoning
        api_base: optional custom API base URL for OpenAI-compatible endpoints
        api_key: optional API key for custom endpoints

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
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def _safe_emit(event_type: str, payload: dict):
        if observability is None:
            return
        try:
            observability.emit(
                event_type,
                payload,
                step=None,
                agent_role=role_name,
            )
        except Exception:
            pass

    # Tool execution node
    tool_node = ToolNode(ALL_TOOLS)

    def llm_node(state: AgentState) -> dict:
        """Call the LLM with current messages."""
        messages = state["messages"]

        # Prepend system prompt if not already there
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        try:
            response = llm_with_tools.invoke(messages)
            _safe_emit(
                "llm.generation",
                {
                    "content_preview": (response.content or "")[:200],
                    "tool_call_count": len(response.tool_calls or []),
                },
            )
            for tc in response.tool_calls or []:
                _safe_emit(
                    "tool.call",
                    {"tool_name": tc.get("name"), "args": tc.get("args", {})},
                )

            if debug and response.content:
                print(f"  [LLM] {response.content[:200]}")

            return {"messages": [response]}
        except Exception as exc:
            _safe_emit(
                "error",
                {"where": "graph.llm_node", "message": str(exc)},
            )
            return {"messages": [AIMessage(content=f"Error: {exc}")]}

    def route_after_llm(state: AgentState) -> str:
        """Route based on the LLM's last message."""
        last_message = state["messages"][-1]

        # No tool calls -> default to wait (END)
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"

        # Check if any tool call is an action tool
        for tc in last_message.tool_calls:
            if tc["name"] in ACTION_TOOL_NAMES:
                return "action_tools"

        # Only observation tools
        return "obs_tools"

    def route_after_tools(state: AgentState) -> str:
        """After executing tools, check if we should continue or stop."""
        # If an action was chosen (action tool was called), we're done
        if get_chosen_action() is not None:
            return "end"
        # Otherwise loop back to LLM
        return "continue"

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
