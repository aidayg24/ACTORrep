"""
Run all ACTOR active learning experiments sequentially.

This script runs each active learning runner as a subprocess.
For each method, it saves:
1. normal terminal output to logs/{method}.out
2. error output to logs/{method}.err
3. success/failure status to logs/run_summary.json

Run from project root:

    python scripts/run_all_active_learning.py
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime


EXPERIMENTS = {
    "random": "src/active_learning/active_learning_random.py",
    "individual_entropy": "src/active_learning/active_learning_individual_entropy.py",
    "group_entropy": "src/active_learning/active_learning_group_entropy_sampling.py",
    "vote_variance": "src/active_learning/active_learning_vote_variance_sampling.py",
    "mixed_entropy": "src/active_learning/active_learning_mixed_entropy_sampling.py",
}


def main():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    summary = {}

    for method_name, script_path in EXPERIMENTS.items():
        print(f"\n==============================")
        print(f"Running: {method_name}")
        print(f"Script: {script_path}")
        print(f"==============================\n")

        start_time = datetime.now().isoformat()

        out_path = log_dir / f"{method_name}.out"
        err_path = log_dir / f"{method_name}.err"

        with open(out_path, "w", encoding="utf-8") as out_file, \
             open(err_path, "w", encoding="utf-8") as err_file:

            result = subprocess.run(
                ["python", script_path],
                stdout=out_file,
                stderr=err_file,
                text=True,
            )

        end_time = datetime.now().isoformat()

        summary[method_name] = {
            "script": script_path,
            "return_code": result.returncode,
            "status": "success" if result.returncode == 0 else "failed",
            "started_at": start_time,
            "ended_at": end_time,
            "stdout_log": str(out_path),
            "stderr_log": str(err_path),
        }

        with open(log_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if result.returncode == 0:
            print(f"{method_name} finished successfully.")
        else:
            print(f"{method_name} failed. Check {err_path}")

    print("\nAll experiments finished.")
    print("Summary saved to logs/run_summary.json")


if __name__ == "__main__":
    main()