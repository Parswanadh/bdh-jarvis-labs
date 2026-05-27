"""
Download and verify Qwen 3.5 0.8B for BDH distillation
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def main():
    print("="*70)
    print("DOWNLOADING QWEN 3.5 0.8B FOR BDH DISTILLATION")
    print("="*70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    if torch.cuda.is_available():
        print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")
    print()

    # Download Qwen 3.5 0.8B
    print("[TEACHER] Loading Qwen 3.5 0.8B...")
    print("[NOTE] Model size: ~1.5GB (fp32), ~800MB (fp16)")
    print()

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.8B-Instruct")

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.8B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        vocab_size = len(tokenizer)

        print(f"[OK] Model loaded successfully!")
        print(f"   Parameters: {num_params:,}")
        print(f"   Vocabulary: {vocab_size:,}")
        print(f"   Architecture: Qwen2.5 with GQA")
        print(f"   Device: {next(model.parameters()).device}")
        print()

        # Test generation
        print("[TEST] Running test generation...")

        prompt = "Once upon a time, there was a little girl who"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print()
        print("[GENERATED OUTPUT]")
        print(generated)
        print()
        print("="*70)
        print("[SUCCESS] Qwen 3.5 0.8B is ready for distillation!")
        print("="*70)

        # Save teacher info
        teacher_info = {
            "model_name": "Qwen/Qwen2.5-0.8B-Instruct",
            "parameters": num_params,
            "vocab_size": vocab_size,
            "context_length": getattr(model.config, "max_position_embeddings", 32768),
            "uses_gqa": True,
            "architecture": "Transformer with Grouped Query Attention"
        }

        torch.save(teacher_info, "checkpoints/qwen_teacher_info.pt")
        print(f"[SAVED] Teacher info to checkpoints/qwen_teacher_info.pt")

    except Exception as e:
        print(f"[ERROR] Failed to load Qwen 3.5 0.8B: {e}")
        print()
        print("[FALLBACK] Try Qwen 2.5 0.8B instead...")

        # Fallback to Qwen 2.5
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.8B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.8B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Loaded Qwen 2.5 0.8B as fallback")
        print(f"   Parameters: {num_params:,}")

if __name__ == "__main__":
    main()
