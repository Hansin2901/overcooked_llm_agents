import yaml
import os
import subprocess
import sys
import csv
import re

def parse_metrics(output):
    total_reward = None
    steps = None
    elapsed = None
    time_per_step = None

    # Extract values from stdout
    reward_match = re.search(r"Total reward:\s*([-\d\.]+)", output)
    step_match = re.search(r"Episode finished in\s*(\d+)\s*steps\s*\(([\d\.]+)s\)", output)
    tps_match = re.search(r"Average time per step:\s*([\d\.]+)s", output)

    if reward_match:
        total_reward = float(reward_match.group(1))

    if step_match:
        steps = int(step_match.group(1))
        elapsed = float(step_match.group(2))

    if tps_match:
        time_per_step = float(tps_match.group(1))

    # LLM call and cost patterns (update if needed)
    llm_calls = len(re.findall(r"llm\.invoke|LLM call", output))
    cost_matches = re.findall(r"cost[:=]\s*\$?([\d\.]+)", output)
    total_cost = sum(float(c) for c in cost_matches) if cost_matches else 0.0

    return total_reward, steps, elapsed, time_per_step, llm_calls, total_cost


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    config_dir = os.path.join(project_root, 'configs')
    agent_script_path = os.path.join(script_dir, "run_llm_agent.py")

    output_csv = os.path.join(script_dir, "experiment_results.csv")

    if not os.path.isdir(config_dir):
        print(f"Error: Configs directory not found at {config_dir}")
        sys.exit(1)

    config_files = sorted([f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml'))])

    if not config_files:
        print(f"No YAML configuration files found in {config_dir}")
        sys.exit(0)

    print(f"Found {len(config_files)} configuration files to run.")

    # Create CSV and header
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "config",
            "layout",
            "model",
            "total_reward",
            "steps",
            "total_time_elapsed",
            "time_per_step",
            "total_llm_calls",
            "total_cost"
        ])

        for config_file in config_files:
            config_path = os.path.join(config_dir, config_file)
            print(f"\n----- Running configuration: {config_file} -----")

            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error reading YAML {config_file}: {e}")
                continue

            env = os.environ.copy()
            env["LAYOUT"] = config.get("layout_name", "")
            env["HORIZON"] = str(config.get("horizon", 200))
            env["DEBUG"] = str(config.get("debug", False)).lower()
            env["VISUALIZE"] = str(config.get("visualize", False)).lower()
            env["FPS"] = str(config.get("fps", 2))
            env["AGENT_TYPE"] = config.get("agent_type", "llm")
            env["REPLAN_INTERVAL"] = str(config.get("replan_interval", 5))
            env["LLM_MODEL"] = config.get("model_name", "")

            print(f"Running layout={env['LAYOUT']} model={env['LLM_MODEL']}")

            python_executable = sys.executable

            try:
                result = subprocess.run(
                    [python_executable, agent_script_path],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True
                )

                stdout = result.stdout

                # Parse metrics
                total_reward, steps, elapsed, time_per_step, llm_calls, total_cost = parse_metrics(stdout)

                # Write row
                writer.writerow([
                    config_file,
                    env["LAYOUT"],
                    env["LLM_MODEL"],
                    total_reward,
                    steps,
                    elapsed,
                    time_per_step,
                    llm_calls,
                    total_cost
                ])

                print(f"Finished {config_file}")
                print(f"Reward={total_reward}, Steps={steps}, Time={elapsed}s")

            except subprocess.CalledProcessError as e:
                print(f"Error running {config_file}: {e.returncode}")
            except Exception as e:
                print(f"Unexpected error with {config_file}: {e}")


if __name__ == "__main__":
    main()