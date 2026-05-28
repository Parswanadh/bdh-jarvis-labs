import torch
import os
from bdh.config import BDHv2Config
from bdh.model import BDHv2


def count_params(checkpoint_path):
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Attempt to infer config or use default
        # In a real scenario, we'd load the config from the checkpoint.
        # Here we'll use the default config as a baseline, but since we only
        # need the parameter count, we can just sum the elements in the state_dict.

        total_params = sum(p.numel() for p in state_dict.values() if p.dim() > 0)
        return total_params
    except Exception as e:
        return f"Error: {e}"


# Search for the latest checkpoint in potential folders
search_dirs = [
    "D:/Projects/BDH/checkpoints/final",
    "D:/Projects/BDH/checkpoints/sota",
    "D:/Projects/BDH/checkpoints/ultimate",
    "D:/Projects/BDH/checkpoints/true_distillation",
]

found_any = False
for d in search_dirs:
    if os.path.exists(d):
        files = [
            os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pt") or f.endswith(".bin")
        ]
        if files:
            latest_file = max(files, key=os.path.getmtime)
            params = count_params(latest_file)
            if isinstance(params, (int, float)):
                print(f"File: {latest_file} | Params: {params / 1e6:.2f}M")
            else:
                print(f"File: {latest_file} | {params}")
            found_any = True

if not found_any:
    print("No checkpoints found in specified directories.")
