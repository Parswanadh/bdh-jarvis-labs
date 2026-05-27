"""
Test the quickly trained BDH model.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "implementation"))

from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig
from bbpe_tokenizer import BBPETokenizer

print("[LOAD] Loading model...")
checkpoint = torch.load("checkpoints/quick_training/final_model.pt", map_location='cuda', weights_only=False)

config = checkpoint['config']
model = MultiScaleBDH(config).to('cuda')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = BBPETokenizer(tokenizer_path="tokenizers/bbpe_tokenizer.json")

print(f"[OK] Model loaded! Loss: {checkpoint['loss']:.4f}")
print()

# Test generation
prompts = [
    "Once upon a time",
    "The little girl",
    "One day",
]

for prompt in prompts:
    print(f"[PROMPT] '{prompt}'")
    print("[GENERATING]", end=" ")

    # Encode
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to('cuda')

    # Generate
    generated = prompt
    with torch.no_grad():
        for i in range(30):
            logits, _ = model(input_ids)
            next_token = torch.argmax(logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode
            decoded = tokenizer.decode(input_ids[0].tolist())
            if decoded and len(decoded) > len(generated):
                new_text = decoded[len(generated):]
                try:
                    print(new_text, end="", flush=True)
                except:
                    print("?", end="", flush=True)
                generated = decoded

            if input_ids.size(1) >= 128:
                break

    print()
    print()
