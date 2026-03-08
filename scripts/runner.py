import yaml
import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run the LLM agent with a given configuration.")
    parser.add_argument("config_path", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print(f"Error: Configuration file not found at {args.config_path}")
        sys.exit(1)

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    agent_script_path = os.path.join(script_dir, "run_llm_agent.py")

    env = os.environ.copy()
    env["LAYOUT"] = config.get("layout_name", "")
    env["HORIZON"] = str(config.get("horizon", 200))
    env["DEBUG"] = str(config.get("debug", False)).lower()
    env["VISUALIZE"] = str(config.get("visualize", False)).lower()
    env["FPS"] = str(config.get("fps", 2))
    env["AGENT_TYPE"] = config.get("agent_type", "llm")
    env["REPLAN_INTERVAL"] = str(config.get("replan_interval", 5))
    env["LLM_MODEL"] = config.get("model_name", "")

    print(f"Running agent with layout: {env['LAYOUT']} and model: {env['LLM_MODEL']}")
    
    # Use python executable from the current environment
    python_executable = sys.executable
    subprocess.run([python_executable, agent_script_path], env=env)

if __name__ == "__main__":
    main()
