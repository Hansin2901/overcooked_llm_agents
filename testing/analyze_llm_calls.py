#!/usr/bin/env python
"""Analyze LLM call patterns from background task output."""

import re
import sys


def analyze_output(filepath):
    """Parse the background task output and count LLM calls per action."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Track LLM calls per step for each agent
    planner_calls = {}
    worker0_calls = {}
    worker1_calls = {}

    current_step = None

    for line in lines:
        # Detect step number
        step_match = re.search(r"\[Step (\d+)\]", line)
        if step_match:
            current_step = int(step_match.group(1))

        # Count planner LLM calls
        if "[Planner] Calling LLM..." in line:
            # Find the invoking step
            for prev_line in reversed(lines[:lines.index(line)]):
                if "Invoking graph (step" in prev_line:
                    step_match = re.search(r"step (\d+)", prev_line)
                    if step_match:
                        step = int(step_match.group(1))
                        planner_calls[step] = planner_calls.get(step, 0) + 1
                        break

        # Count worker_0 LLM calls
        if "[worker_0] Calling LLM..." in line:
            if current_step is not None:
                worker0_calls[current_step] = worker0_calls.get(current_step, 0) + 1

        # Count worker_1 LLM calls
        if "[worker_1] Calling LLM..." in line:
            if current_step is not None:
                worker1_calls[current_step] = worker1_calls.get(current_step, 0) + 1

    # Print results
    print("=" * 60)
    print("LLM CALL ANALYSIS")
    print("=" * 60)
    print()

    print("PLANNER LLM CALLS PER PLANNING CYCLE:")
    if planner_calls:
        for step in sorted(planner_calls.keys()):
            print(f"  Step {step}: {planner_calls[step]} LLM calls")
        avg_planner = sum(planner_calls.values()) / len(planner_calls)
        print(f"  Average: {avg_planner:.1f} calls per planning cycle")
    else:
        print("  No planner calls detected")
    print()

    print("WORKER_0 LLM CALLS PER ACTION:")
    if worker0_calls:
        for step in sorted(worker0_calls.keys())[:10]:  # Show first 10
            print(f"  Step {step}: {worker0_calls[step]} LLM calls")
        avg_w0 = sum(worker0_calls.values()) / len(worker0_calls)
        print(f"  Average: {avg_w0:.1f} calls per action")
    else:
        print("  No worker_0 calls detected")
    print()

    print("WORKER_1 LLM CALLS PER ACTION:")
    if worker1_calls:
        for step in sorted(worker1_calls.keys())[:10]:  # Show first 10
            print(f"  Step {step}: {worker1_calls[step]} LLM calls")
        avg_w1 = sum(worker1_calls.values()) / len(worker1_calls)
        print(f"  Average: {avg_w1:.1f} calls per action")
    else:
        print("  No worker_1 calls detected")
    print()

    print("=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)
    total_calls_per_step = (
        (avg_w0 if worker0_calls else 0) +
        (avg_w1 if worker1_calls else 0) +
        (avg_planner / 3 if planner_calls else 0)  # Planner runs every 3 steps
    )
    print(f"Total LLM calls per timestep: {total_calls_per_step:.1f}")
    print()

    if avg_w0 > 2 or avg_w1 > 2:
        print("⚠️  Workers are making excessive observation tool calls!")
        print("    Expected: 1-2 calls per action")
        print(f"    Actual: worker_0={avg_w0:.1f}, worker_1={avg_w1:.1f}")
        print()
        print("ROOT CAUSE: Workers are calling too many observation tools")
        print("before choosing an action. This is likely due to:")
        print("1. New observation tools added (get_nearby_interactables, validate_task_feasibility)")
        print("2. System prompts encouraging exploration")
        print()
        print("RECOMMENDATION: Review worker/planner system prompts to")
        print("encourage more direct action selection.")
    else:
        print("✓ Worker LLM call count looks normal (1-2 per action)")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_llm_calls.py <background_task_output_file>")
        sys.exit(1)

    analyze_output(sys.argv[1])
