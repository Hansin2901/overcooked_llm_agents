#!/usr/bin/env python
"""Diagnostic script to measure LLM API latency and calls per action.

This script tests:
1. Raw API latency - how long does a single LLM call take?
2. Number of LLM calls per worker action
3. Number of LLM calls per planner cycle
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from overcooked_ai_py.agents.llm.llm_agent import LLMAgent
from overcooked_ai_py.agents.llm.planner import Planner
from overcooked_ai_py.agents.llm.worker_agent import WorkerAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair


def test_raw_api_latency():
    """Test raw API latency with a simple completion."""
    print("=" * 60)
    print("TEST 1: Raw API Latency")
    print("=" * 60)

    load_dotenv()

    from litellm import completion

    model = os.getenv("LLM_MODEL", "gpt-4o")
    api_base = os.getenv("LLM_API_BASE")
    api_key = os.getenv("LLM_API_KEY")

    print(f"Model: {model}")
    print(f"API Base: {api_base}")
    print()

    # Test 5 simple completions
    latencies = []
    for i in range(5):
        start = time.time()
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "temperature": 0.2,
                "max_tokens": 10,
            }
            if api_base:
                kwargs["api_base"] = api_base
            if api_key:
                kwargs["api_key"] = api_key

            response = completion(**kwargs)
            duration = time.time() - start
            latencies.append(duration)
            print(f"  Call {i+1}: {duration*1000:.0f}ms - {response.choices[0].message.content}")
        except Exception as e:
            print(f"  Call {i+1}: FAILED - {e}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"\nAverage latency: {avg*1000:.0f}ms")
        print(f"Min: {min(latencies)*1000:.0f}ms, Max: {max(latencies)*1000:.0f}ms")
    print()


def test_worker_llm_calls():
    """Count LLM calls per worker action."""
    print("=" * 60)
    print("TEST 2: LLM Calls Per Worker Action")
    print("=" * 60)

    load_dotenv()

    model = os.getenv("LLM_MODEL", "gpt-4o")
    api_base = os.getenv("LLM_API_BASE")
    api_key = os.getenv("LLM_API_KEY")

    # Create simple environment
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OvercookedEnv.from_mdp(mdp, horizon=20)

    # Create planner-worker setup
    planner = Planner(
        model_name=model,
        replan_interval=3,
        debug=True,
        api_base=api_base,
        api_key=api_key,
    )

    worker_0 = WorkerAgent(
        planner=planner,
        worker_id="worker_0",
        agent_index=0,
        model_name=model,
        debug=True,
        api_base=api_base,
        api_key=api_key,
    )

    worker_1 = WorkerAgent(
        planner=planner,
        worker_id="worker_1",
        agent_index=1,
        model_name=model,
        debug=True,
        api_base=api_base,
        api_key=api_key,
    )

    agent_pair = AgentPair(worker_0, worker_1)

    # Initialize
    state = env.reset()[0]
    agent_pair.set_mdp(mdp)

    print(f"Running 3 timesteps to count LLM calls...")
    print()

    # Track LLM calls by counting "Calling LLM..." debug prints
    # This is a bit hacky but works for diagnostic purposes

    for step in range(3):
        print(f"\n--- Step {step} ---")
        step_start = time.time()

        joint_action, infos = agent_pair.joint_action(state)

        step_duration = time.time() - step_start
        print(f"Step {step} completed in {step_duration:.2f}s")

        state, reward, done, truncated, info = env.step(joint_action, infos)

        if done or truncated:
            break

    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    print("Look at the debug output above and count:")
    print("- How many '[worker_0] Calling LLM...' per step?")
    print("- How many '[worker_1] Calling LLM...' per step?")
    print("- How many '[Planner] Calling LLM...' per planning cycle?")
    print()
    print("Expected: 1-2 LLM calls per action (one to call tools, one to choose action)")
    print("If you see 3+ calls per action, the agent is making excessive observations.")
    print()


def main():
    """Run all diagnostic tests."""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "API LATENCY DIAGNOSTIC" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Test 1: Raw API latency
    test_raw_api_latency()

    # Test 2: LLM calls per action
    test_worker_llm_calls()

    print()
    print("=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    print()
    print("If average API latency is >2000ms: The API endpoint is slow")
    print("If LLM calls per action is >2: Agents are using too many observation tools")
    print()


if __name__ == "__main__":
    main()
