import yaml
import os
import subprocess
import sys

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    config_dir = os.path.join(project_root, 'configs')
    agent_script_path = os.path.join(script_dir, "run_llm_agent.py")

    if not os.path.isdir(config_dir):
        print(f"Error: Configs directory not found at {config_dir}")
        sys.exit(1)

    config_files = sorted([f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml'))])

    if not config_files:
        print(f"No YAML configuration files found in {config_dir}")
        sys.exit(0)

    print(f"Found {len(config_files)} configuration files to run.")

    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        print(f"\n----- Running configuration: {config_file} -----")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error reading or parsing YAML file {config_path}: {e}")
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

        print(f"Running agent with layout: '{env['LAYOUT']}' and model: '{env['LLM_MODEL']}'")
        
        python_executable = sys.executable
        try:
            subprocess.run([python_executable, agent_script_path], env=env, check=True)
            print(f"----- Finished configuration: {config_file} -----")
        except subprocess.CalledProcessError as e:
            print(f"Error running agent for configuration {config_file}. Exit code: {e.returncode}")
        except Exception as e:
            print(f"An unexpected error occurred while running agent for {config_file}: {e}")

if __name__ == "__main__":
    main()
