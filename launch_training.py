"""
Self-contained training launcher - finds conda and runs training
"""
import subprocess
import sys
import os
from pathlib import Path

print("Searching for conda...")

# Initialize variables
conda_exe = None
conda_base = None

# Get CONDA_EXE from environment if available
conda_exe_str = os.environ.get("CONDA_EXE", "")
if conda_exe_str and Path(conda_exe_str).exists():
    conda_exe = Path(conda_exe_str)
    conda_base = conda_exe.parent.parent
    print(f"Found conda from env: {conda_exe}")
    print(f"Base: {conda_base}")
else:
    # Search for conda.bat or conda.exe in PATH
    try:
        result = subprocess.run(["where", "conda"], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            conda_path = result.stdout.strip().split('\n')[0]
            conda_exe = Path(conda_path)
            if "Scripts" in str(conda_exe):
                conda_base = conda_exe.parent.parent
            else:
                conda_base = conda_exe.parent
            print(f"Found conda from PATH: {conda_exe}")
    except:
        pass

if conda_exe is None or not conda_exe.exists():
    print("Cannot find conda. Please run manually:")
    print("  conda activate bdh-fest")
    print("  python train_minimal_working.py")
    sys.exit(1)

# Find bdh-fest environment
print("\nSearching for bdh-fest environment...")

# Method 1: Use conda env list
python_exe = None
try:
    result = subprocess.run(
        [str(conda_exe), "env", "list"],
        capture_output=True,
        text=True,
        shell=True
    )
    print("Conda envs:")
    for line in result.stdout.split('\n'):
        print(f"  {line}")
        if 'bdh-fest' in line:
            # Parse the path from conda output
            if '(' in line:
                env_path = line.split('(')[-1].rstrip(')')
                if env_path and Path(env_path).exists():
                    python_exe = Path(env_path) / "python.exe"
                    if python_exe.exists():
                        print(f"Found: {python_exe}")
                        break
except Exception as e:
    print(f"Error running conda env list: {e}")

# Method 2: Check standard locations
if not python_exe:
    locations = [
        conda_base / "envs" / "bdh-fest" / "python.exe",
        Path.home() / ".conda" / "envs" / "bdh-fest" / "python.exe",
        Path("C:/Users/parshu/.conda/envs/bdh-fest/python.exe"),
    ]
    for loc in locations:
        if loc.exists():
            python_exe = loc
            print(f"Found at: {python_exe}")
            break

if not python_exe:
    print("ERROR: bdh-fest environment not found!")
    sys.exit(1)

# Run training
print("\n" + "="*80)
print("Starting training...")
print("="*80)
print(f"Python: {python_exe}")
print()

training_script = Path(__file__).parent / "train_minimal_working.py"

result = subprocess.run(
    [str(python_exe), str(training_script)],
    cwd=Path(__file__).parent,
    env=os.environ.copy()
)

print("\nTraining completed with exit code:", result.returncode)
