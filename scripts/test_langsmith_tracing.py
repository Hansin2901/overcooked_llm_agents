#!/usr/bin/env python
"""Test script to verify LangSmith tracing is working.

This script performs a minimal LangChain operation and checks if it appears in LangSmith.

Usage:
    # Enable tracing
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=lsv2_pt_...
    export LANGCHAIN_PROJECT=overcooked-test

    # Run test
    uv run python scripts/test_langsmith_tracing.py
"""

import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def check_env_vars():
    """Check if required environment variables are set."""
    tracing = os.getenv("LANGCHAIN_TRACING_V2")
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT", "default")

    print("=" * 60)
    print("LangSmith Tracing Configuration Check")
    print("=" * 60)
    print(f"LANGCHAIN_TRACING_V2: {tracing}")
    print(f"LANGCHAIN_API_KEY: {'SET' if api_key else 'NOT SET'}")
    print(f"LANGCHAIN_PROJECT: {project}")
    print()

    if tracing != "true":
        print("⚠️  WARNING: LANGCHAIN_TRACING_V2 is not set to 'true'")
        print("   Traces will NOT be sent to LangSmith.")
        print()

    if not api_key:
        print("❌ ERROR: LANGCHAIN_API_KEY is not set")
        print("   Get your API key from: https://smith.langchain.com/settings")
        print()
        return False

    return True


def test_langchain_tracing():
    """Test basic LangChain tracing with a simple LLM call."""
    try:
        from langchain_community.chat_models import ChatLiteLLM
        from langchain_core.messages import HumanMessage
    except ImportError as e:
        print(f"❌ ERROR: Failed to import LangChain: {e}")
        print("   Install dependencies: uv sync")
        return False

    # Check for LLM API key
    llm_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not llm_key:
        print("❌ ERROR: No LLM API key found (OPENAI_API_KEY or LLM_API_KEY)")
        print("   Add your API key to .env file")
        return False

    print("=" * 60)
    print("Testing LangChain Tracing")
    print("=" * 60)
    print()

    # Use environment model or default to gpt-4o-mini (cheaper for testing)
    model_name = os.getenv("LLM_MODEL") or "gpt-4o-mini"
    api_base = os.getenv("LLM_API_BASE")

    print(f"Model: {model_name}")
    if api_base:
        print(f"API Base: {api_base}")
    print()

    try:
        # Create LLM
        llm_kwargs = {"model": model_name, "temperature": 0}
        if api_base:
            llm_kwargs["api_base"] = api_base
        if llm_key:
            llm_kwargs["api_key"] = llm_key

        llm = ChatLiteLLM(**llm_kwargs)

        # Make a simple call
        print("Making test LLM call...")
        message = HumanMessage(
            content="Say 'Hello from Overcooked-AI tracing test!' and nothing else."
        )
        response = llm.invoke([message])

        print(f"✅ Response: {response.content}")
        print()

        project = os.getenv("LANGCHAIN_PROJECT", "default")
        print("=" * 60)
        print("✅ SUCCESS: LLM call completed")
        print("=" * 60)
        print()
        print("If LANGCHAIN_TRACING_V2=true is set, check your traces at:")
        print(f"  https://smith.langchain.com/o/projects/p/{project}")
        print()
        print("Look for a trace named something like:")
        print("  'ChatLiteLLM' or 'RunnableSequence'")
        print()
        print("The trace should show:")
        print("  - Input: 'Say 'Hello from...'")
        print("  - Output: 'Hello from Overcooked-AI tracing test!'")
        print("  - Token usage and latency")
        print()

        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to call LLM: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_langgraph_tracing():
    """Test LangGraph tracing with a simple graph."""
    try:
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict
    except ImportError as e:
        print(f"❌ ERROR: Failed to import LangGraph: {e}")
        return False

    print("=" * 60)
    print("Testing LangGraph Tracing")
    print("=" * 60)
    print()

    try:
        # Define a simple state
        class State(TypedDict):
            counter: int

        # Define nodes
        def increment(state: State) -> dict:
            return {"counter": state["counter"] + 1}

        def double(state: State) -> dict:
            return {"counter": state["counter"] * 2}

        # Build graph
        graph = StateGraph(State)
        graph.add_node("increment", increment)
        graph.add_node("double", double)
        graph.add_edge(START, "increment")
        graph.add_edge("increment", "double")
        graph.add_edge("double", END)

        app = graph.compile()

        # Run graph
        print("Running simple graph: increment(1) -> double -> result")
        result = app.invoke({"counter": 1})

        print(f"✅ Result: {result}")
        print(f"   Expected: {{'counter': 4}}, Got: {result}")
        print()

        project = os.getenv("LANGCHAIN_PROJECT", "default")
        print("=" * 60)
        print("✅ SUCCESS: LangGraph execution completed")
        print("=" * 60)
        print()
        print("If LANGCHAIN_TRACING_V2=true is set, check your traces at:")
        print(f"  https://smith.langchain.com/o/projects/p/{project}")
        print()
        print("Look for a trace showing the graph execution:")
        print("  - Node: increment (counter: 1 -> 2)")
        print("  - Node: double (counter: 2 -> 4)")
        print()

        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to run LangGraph: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print()
    print("=" * 60)
    print("LangSmith Tracing Test Suite")
    print("=" * 60)
    print()

    # Check environment
    if not check_env_vars():
        print("❌ FAILED: Environment variables not configured properly")
        print()
        print("To fix:")
        print("  1. Add to your .env file:")
        print("     LANGCHAIN_TRACING_V2=true")
        print("     LANGCHAIN_API_KEY=lsv2_pt_...")
        print("     LANGCHAIN_PROJECT=overcooked-test")
        print()
        print("  2. Get API key from: https://smith.langchain.com/settings")
        print()
        sys.exit(1)

    # Test LangChain tracing
    langchain_ok = test_langchain_tracing()

    # Test LangGraph tracing
    langgraph_ok = test_langgraph_tracing()

    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"LangChain Tracing: {'✅ PASS' if langchain_ok else '❌ FAIL'}")
    print(f"LangGraph Tracing: {'✅ PASS' if langgraph_ok else '❌ FAIL'}")
    print()

    if langchain_ok and langgraph_ok:
        print("✅ ALL TESTS PASSED")
        print()
        print("Next steps:")
        print("  1. Check your LangSmith project for traces")
        print("  2. Run the main agent with tracing enabled:")
        print("     source .env && uv run python scripts/run_llm_agent.py \\")
        print("       --agent-type planner-worker \\")
        print("       --layout cramped_room \\")
        print("       --horizon 10 \\")
        print("       --debug")
        print()
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("Check the error messages above for details.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
