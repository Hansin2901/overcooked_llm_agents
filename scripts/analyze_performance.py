#!/usr/bin/env python
"""Analyze performance of LLM agent runs from observability logs."""

import json
import sys
from collections import defaultdict
from pathlib import Path


def analyze_run(log_file: Path):
    """Analyze a single run log file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {log_file.name}")
    print('='*80)

    events = []
    with open(log_file) as f:
        for line in f:
            events.append(json.loads(line))

    # Extract run metadata
    if events:
        first = events[0]
        print(f"\nRun ID: {first['run_id']}")
        print(f"Mode: {first['mode']}")
        print(f"Layout: {first['layout']}")
        print(f"Model: {first['model']}")

    # Group by agent role and event type
    by_role = defaultdict(lambda: defaultdict(list))

    for event in events:
        role = event.get('agent_role', 'unknown')
        etype = event['event_type']
        by_role[role][etype].append(event)

    # Analyze LLM calls
    print(f"\n{'─'*80}")
    print("LLM Performance Analysis")
    print('─'*80)

    for role in sorted(by_role.keys()):
        llm_gens = by_role[role].get('llm.generation', [])
        if not llm_gens:
            continue

        print(f"\n{role.upper()}:")
        print(f"  LLM Calls: {len(llm_gens)}")

        total_latency = sum(g['payload'].get('latency_ms', 0) for g in llm_gens)
        avg_latency = total_latency / len(llm_gens) if llm_gens else 0

        print(f"  Total LLM Time: {total_latency/1000:.1f}s")
        print(f"  Avg Latency: {avg_latency/1000:.1f}s per call")

        total_tokens = sum(g['payload'].get('total_tokens', 0) for g in llm_gens)
        total_cost = sum(g['payload'].get('estimated_cost_usd', 0) for g in llm_gens)

        print(f"  Total Tokens: {total_tokens:,}")
        if total_cost:
            print(f"  Estimated Cost: ${total_cost:.4f}")

        # Tool calls
        tool_calls = by_role[role].get('tool.call', [])
        if tool_calls:
            print(f"  Tool Calls: {len(tool_calls)}")
            tool_counts = defaultdict(int)
            for tc in tool_calls:
                tool_name = tc['payload'].get('tool_name', 'unknown')
                tool_counts[tool_name] += 1
            for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
                print(f"    - {tool}: {count}x")

    # Step-by-step breakdown
    print(f"\n{'─'*80}")
    print("Step-by-Step Breakdown")
    print('─'*80)

    by_step = defaultdict(lambda: defaultdict(list))
    for event in events:
        step = event.get('step')
        if step is not None:
            role = event.get('agent_role', 'unknown')
            etype = event['event_type']
            by_step[step][(role, etype)].append(event)

    for step in sorted(by_step.keys()):
        step_events = by_step[step]
        print(f"\nStep {step}:")

        # Count LLM calls per role
        for (role, etype) in sorted(step_events.keys()):
            if etype == 'llm.generation':
                gens = step_events[(role, etype)]
                latency = sum(g['payload'].get('latency_ms', 0) for g in gens)
                print(f"  {role}: {len(gens)} LLM calls, {latency/1000:.1f}s")


def main():
    logs_dir = Path("logs/agent_runs")

    if not logs_dir.exists():
        print(f"Error: {logs_dir} not found")
        sys.exit(1)

    log_files = sorted(logs_dir.glob("*.jsonl"))

    if not log_files:
        print(f"No log files found in {logs_dir}")
        sys.exit(1)

    # Analyze most recent or specified file
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.exists():
            target = logs_dir / sys.argv[1]
        log_files = [target]
    else:
        log_files = [log_files[-1]]  # Most recent

    for log_file in log_files:
        analyze_run(log_file)


if __name__ == "__main__":
    main()
