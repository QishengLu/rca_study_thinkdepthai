import os
import sys
import json
import re
import subprocess
from pathlib import Path

def main():
    # Setup paths
    base_dir = Path(__file__).parent.resolve()
    data_dir = base_dir / "data"
    experiments_dir = base_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    
    problem_file = data_dir / "problem.json"
    
    if not problem_file.exists():
        print(f"Error: {problem_file} not found.")
        sys.exit(1)

    # Read problem.json
    # The file content is a Python variable assignment
    with open(problem_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Extract text between triple quotes
    match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if match:
        task_description = match.group(1).strip()
    else:
        task_description = content.strip()
        
    # Create task.json for the runner
    task_file = base_dir / "task.json"
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump({"task_description": task_description}, f, indent=2)
        
    print(f"Generated task.json from {problem_file.name}")

    # Run the actual analysis using subprocess to avoid importing packages here
    # This keeps this script clean and independent of the project dependencies
    cmd = ["uv", "run", "python", "run_rca.py"]
    
    print("Starting RCA analysis...")
    try:
        subprocess.run(cmd, cwd=base_dir, check=True)
        print(f"Analysis complete. Results saved to {experiments_dir / 'output.json'}")
    except subprocess.CalledProcessError as e:
        print(f"Analysis failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
