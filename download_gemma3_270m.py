"""
Download Gemma 3 270M from HuggingFace
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def main():
    print("="*70)
    print("DOWNLOADING GEMMA 3 270M FROM HUGGINGFACE")
    print("="*70)
    print()

    print("[TEACHER] Loading Gemma 3 270M...")
    print("[NOTE] This will download ~550MB on first run")
    print()

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m-it",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print()
    print("[OK] Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocabulary size: {len(tokenizer):,}")
    print(f"   Device: {next(model.parameters()).device}")
    print()

    # Test generation
    print("[TEST] Running test generation...")

    # Fix: Set pad token for Gemma
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Simple test without chat template
    prompt = "Who are you?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,  # Use greedy decoding to avoid CUDA error
        pad_token_id=tokenizer.pad_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print()
    print("[GENERATED OUTPUT]")
    print(decoded)
    print()
    print("="*70)
    print("[SUCCESS] Gemma 3 270M is ready for distillation!")
    print("="*70)

if __name__ == "__main__":
    main()
