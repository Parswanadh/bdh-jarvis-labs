import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

def run_surgery():
    device = 'cpu'
    checkpoint_path = Path("checkpoints/safe/laptop_safe.pt")
    out_path = Path("checkpoints/safe/laptop_safe_11L.pt")
    
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found.")
        return
        
    print("Loading 8-layer model for RYS Brain Surgery...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model']
    
    new_state_dict = {}
    
    print("Performing RYS Hack: Duplicating Layers 3, 4, 5...")
    for key, value in state_dict.items():
        if key.startswith('layers.'):
            parts = key.split('.')
            layer_idx = int(parts[1])
            suffix = '.'.join(parts[2:])
            
            if layer_idx < 6:
                # Keep 0 to 5 as they are
                new_state_dict[f'layers.{layer_idx}.{suffix}'] = value.clone()
            
            # Duplicate 3, 4, 5 to become 6, 7, 8
            if layer_idx == 3:
                new_state_dict[f'layers.6.{suffix}'] = value.clone()
            elif layer_idx == 4:
                new_state_dict[f'layers.7.{suffix}'] = value.clone()
            elif layer_idx == 5:
                new_state_dict[f'layers.8.{suffix}'] = value.clone()
                
            # Shift original 6, 7 to 9, 10
            if layer_idx >= 6:
                new_state_dict[f'layers.{layer_idx + 3}.{suffix}'] = value.clone()
        else:
            new_state_dict[key] = value.clone()
            
    ckpt['model'] = new_state_dict
    
    # Reset step counter to allow healing phase
    ckpt['step'] = 0 
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Surgery complete! Model expanded to 11 layers.")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    run_surgery()
