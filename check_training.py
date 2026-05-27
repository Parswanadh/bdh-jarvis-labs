"""
Quick training progress checker
"""

import time
import os
from pathlib import Path

checkpoint_dir = Path("checkpoints/multiscale_bdh_ollama")

print("="*60)
print("BDH Training Progress Check")
print("="*60)
print()

# Check for checkpoints
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("checkpoint_iter_*.pt"))

    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoint(s)")

        for ckpt in sorted(checkpoints, key=lambda x: int(x.stem.split('_')[-1])):
            try:
                import torch
                data = torch.load(ckpt, weights_only=False, map_location='cpu')
                iteration = data.get('iteration', 0)
                loss = data.get('loss', 0)
                is_best = data.get('is_best', False)

                status = " [BEST]" if is_best else ""
                print(f"  Iteration {iteration}: loss={loss:.4f}{status}")
            except:
                print(f"  {ckpt.name}")
    else:
        print("No checkpoints found yet")
else:
    print("Checkpoint directory not created yet - training may still be initializing")

print()
print("Checkpoints saved to:", checkpoint_dir)
print()

# Check if process is running
try:
    import subprocess
    result = subprocess.run(['tasklist'], capture_output=True, text=True)
    if 'python' in result.stdout:
        print("Python processes are running - training likely in progress")
except:
    pass
