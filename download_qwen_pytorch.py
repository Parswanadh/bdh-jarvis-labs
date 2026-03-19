"""
Download Qwen 2.5/3.5 0.8B in PyTorch format for BDH distillation
NOT the GGUF format - we need PyTorch safetensors for training
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys

def main():
    print("="*70)
    print("DOWNLOADING QWEN 2.5/3.5 0.8B PYTORCH MODEL FOR DISTILLATION")
    print("="*70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    if torch.cuda.is_available():
        print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Try Qwen 2.5 0.8B (more likely to be available)
    model_names = [
        "Qwen/Qwen2.5-0.8B-Instruct",
        "Qwen/Qwen1.5-0.8B-Chat",
        "Qwen/Qwen2-0.5B-Instruct"
    ]

    for model_name in model_names:
        try:
            print(f"[TRY] Loading {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            model.eval()

            num_params = sum(p.numel() for p in model.parameters())
            vocab_size = len(tokenizer)

            print(f"[OK] Successfully loaded {model_name}")
            print(f"   Parameters: {num_params:,}")
            print(f"   Vocabulary: {vocab_size:,}")
            print(f"   Device: {next(model.parameters()).device}")
            print()

            # Test generation
            print("[TEST] Running test generation...")

            # Fix pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            prompt = "Once upon a time, there was a little"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print("[GENERATED]")
            print(generated)
            print()

            # Save model info
            info = {
                "model_name": model_name,
                "parameters": num_params,
                "vocab_size": vocab_size,
                "dtype": "float16",
                "device": str(device),
                "max_position_embeddings": getattr(model.config, "max_position_embeddings", "unknown")
            }

            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(info, "checkpoints/qwen_teacher_info.pt")
            print(f"[SAVED] Teacher info to checkpoints/qwen_teacher_info.pt")
            print()

            print("="*70)
            print(f"[SUCCESS] {model_name} ready for distillation!")
            print("="*70)

            return

        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")
            print()
            continue

    print("[FALLBACK] All models failed. Please check:")
    print("1. Internet connection")
    print("2. HuggingFace access (need to accept terms)")
    print("3. Available models at: https://huggingface.co/Qwen")

if __name__ == "__main__":
    main()
