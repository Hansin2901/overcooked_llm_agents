#!/usr/bin/env python
"""Run an LLM agent paired with GreedyHumanModel on an Overcooked layout.

Usage:
    python scripts/run_llm_agent.py [--model MODEL] [--layout LAYOUT] [--horizon N] [--debug]

Requires a .env file in the project root with your API key(s):
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
"""

import argparse
import time

from dotenv import load_dotenv

# Load API keys from .env before any LLM imports
load_dotenv()

from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.llm import LLMAgent
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS


def make_greedy_partner(mdp):
    """Create a GreedyHumanModel partner agent."""
    from overcooked_ai_py.agents.agent import GreedyHumanModel

    mlam = MediumLevelActionManager.from_pickle_or_compute(
        mdp, NO_COUNTERS_PARAMS, force_compute=False
    )
    return GreedyHumanModel(mlam)


def main():
    parser = argparse.ArgumentParser(description="Run LLM agent on Overcooked")
    parser.add_argument("--model", default="gpt-4o", help="LiteLLM model name (default: gpt-4o)")
    parser.add_argument("--layout", default="cramped_room", help="Layout name (default: cramped_room)")
    parser.add_argument("--horizon", type=int, default=200, help="Episode length (default: 200)")
    parser.add_argument("--debug", action="store_true", help="Print LLM reasoning each step")
    args = parser.parse_args()

    print(f"Layout: {args.layout}")
    print(f"Model: {args.model}")
    print(f"Horizon: {args.horizon}")
    print()

    # Create MDP and environment
    mdp = OvercookedGridworld.from_layout_name(args.layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=args.horizon)

    # Create agents
    llm_agent = LLMAgent(model_name=args.model, debug=args.debug, horizon=args.horizon)
    partner = make_greedy_partner(mdp)

    agent_pair = AgentPair(llm_agent, partner)

    # Reset environment and agents
    env.reset()
    agent_pair.reset()
    agent_pair.set_mdp(mdp)

    print("Running episode...")
    start_time = time.time()

    total_reward = 0
    done = False
    step = 0

    while not done:
        state = env.state
        joint_action_and_infos = agent_pair.joint_action(state)
        actions = tuple(a for a, _ in joint_action_and_infos)
        infos = tuple(info for _, info in joint_action_and_infos)

        next_state, rewards, done, info = env.step(actions, infos)

        step_reward = sum(rewards)
        total_reward += step_reward

        if step_reward > 0 or args.debug:
            print(f"  Step {step}: actions={actions}, reward={step_reward}, total={total_reward}")

        step += 1

    elapsed = time.time() - start_time
    print()
    print(f"Episode finished in {step} steps ({elapsed:.1f}s)")
    print(f"Total reward: {total_reward}")
    print(f"Average time per step: {elapsed / max(step, 1):.2f}s")


if __name__ == "__main__":
    main()
