import subprocess
import sys

print("Starting BDH training...")
print("=" * 60)

try:
    # Run the training script
    result = subprocess.run(
        [sys.executable, "train_bdh_reasoning.py"],
        cwd="D:/projects/BDH",
        capture_output=False,
        text=True
    )
    print("\nTraining completed!")
except Exception as e:
    print(f"Error: {e}")
