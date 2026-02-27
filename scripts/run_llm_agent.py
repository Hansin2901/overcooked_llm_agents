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
    import os

    parser = argparse.ArgumentParser(description="Run LLM agent on Overcooked")
    parser.add_argument("--model", default=None, help="LiteLLM model name (default: from LLM_MODEL env or gpt-4o)")
    parser.add_argument("--layout", default="cramped_room", help="Layout name (default: cramped_room)")
    parser.add_argument("--horizon", type=int, default=200, help="Episode length (default: 200)")
    parser.add_argument("--debug", action="store_true", help="Print LLM reasoning each step")
    parser.add_argument("--visualize", action="store_true", help="Show real-time pygame visualization")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for visualization (default: 2)")
    args = parser.parse_args()

    # Read model, API base, and API key from environment if not provided as arguments
    model = args.model or os.getenv("LLM_MODEL") or "gpt-4o"
    api_base = os.getenv("LLM_API_BASE")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    print(f"Layout: {args.layout}")
    print(f"Model: {model}")
    print(f"Horizon: {args.horizon}")
    if api_base:
        print(f"API Base: {api_base}")
    print()

    # Create MDP and environment
    mdp = OvercookedGridworld.from_layout_name(args.layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=args.horizon)

    # Create agents
    llm_agent = LLMAgent(model_name=model, debug=args.debug, horizon=args.horizon, api_base=api_base, api_key=api_key)
    partner = make_greedy_partner(mdp)

    agent_pair = AgentPair(llm_agent, partner)

    # Reset environment and agents
    env.reset()
    agent_pair.reset()
    agent_pair.set_mdp(mdp)

    # Initialize visualization if requested
    display = None
    clock = None
    if args.visualize:
        import pygame
        from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

        pygame.init()
        visualizer = StateVisualizer()

        # Create display window (800x600 default size)
        display = pygame.display.set_mode((800, 600))
        pygame.display.set_caption(f"Overcooked AI - {args.layout}")
        clock = pygame.time.Clock()

        print("Visualization window opened. Close window or press ESC to exit.")

    print("Running episode...")
    start_time = time.time()

    total_reward = 0
    done = False
    step = 0

    while not done:
        state = env.state

        # Render current state if visualizing
        if args.visualize:
            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    done = True
                    break

            if done:
                break

            # Render state to display
            surf = visualizer.render_state(state, grid=mdp.terrain_mtx)

            # Scale to fit window if needed
            if surf.get_width() != display.get_width() or surf.get_height() != display.get_height():
                surf = pygame.transform.scale(surf, (display.get_width(), display.get_height()))

            display.fill((0, 0, 0))  # Clear screen
            display.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(args.fps)

        joint_action_and_infos = agent_pair.joint_action(state)
        actions = tuple(a for a, _ in joint_action_and_infos)
        infos = tuple(info for _, info in joint_action_and_infos)

        next_state, rewards, done, info = env.step(actions, infos)

        step_reward = rewards if isinstance(rewards, (int, float)) else sum(rewards)
        total_reward += step_reward

        if step_reward > 0 or args.debug:
            print(f"  Step {step}: actions={actions}, reward={step_reward}, total={total_reward}")

        step += 1

    # Cleanup visualization
    if args.visualize:
        pygame.quit()

    elapsed = time.time() - start_time
    print()
    print(f"Episode finished in {step} steps ({elapsed:.1f}s)")
    print(f"Total reward: {total_reward}")
    print(f"Average time per step: {elapsed / max(step, 1):.2f}s")


if __name__ == "__main__":
    main()
