"""
Test the downloaded Qwen 3.5 0.8B model
Verifies it works correctly for distillation training
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def main():
    print("="*70)
    print("TESTING QWEN 3.5 0.8B PYTORCH MODEL")
    print("="*70)
    print()

    model_path = "Qwen3.5-0.8B"

    if not Path(model_path).exists():
        print(f"[ERROR] Model not found at {model_path}")
        print(f"[HINT] The files should be directly in D:/projects/BDH/{model_path}/")
        print()
        print("Looking for files...")
        import os
        if Path("Qwen3.5-0.8B").exists():
            print("Found: Qwen3.5-0.8B/ (directory)")
            files = list(Path("Qwen3.5-0.8B").rglob("*"))
            for f in files:
                print(f"  - {f.name}")
        return

    print(f"[LOAD] Loading model from {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    if torch.cuda.is_available():
        print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")
    print()

    try:
        # Load tokenizer
        print("[1/2] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"[OK] Tokenizer loaded")
        print(f"   Vocab size: {len(tokenizer):,}")

        # Load model
        print()
        print("[2/2] Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model loaded")
        print(f"   Parameters: {num_params:,}")
        print(f"   Hidden size: {model.config.hidden_size}")
        print(f"   Num layers: {model.config.num_hidden_layers}")
        print(f"   Num attention heads: {model.config.num_attention_heads}")
        print(f"   Context length: {model.config.max_position_embeddings}")
        print()

        # Test generation
        print("[TEST] Running test generation...")

        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        prompt = "Once upon a time, there was a little"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print(f"   Input: '{prompt}'")
        print(f"   Input shape: {inputs['input_ids'].shape}")

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
        print("[GENERATED]")
        print(generated)
        print()

        # Test logits access (CRITICAL for distillation!)
        print("[LOGIT TEST] Verifying logits access...")

        with torch.no_grad():
            logits = model(**inputs)
            print(f"   Logits shape: {logits.logits.shape}")
            print(f"   Vocab size: {logits.logits.shape[-1]}")

            # Convert to probabilities
            probs = torch.softmax(logits.logits[0, -1, :], dim=-1)
            top_5 = torch.topk(probs, 5)

            print()
            print(f"   [Top 5 tokens for next prediction]")
            for i, (prob, idx) in enumerate(zip(top_5.values.tolist(), top_5.indices.tolist())):
                token = tokenizer.decode([idx])[0]
                print(f"      {i+1}. {token}: {prob:.4f}")

        print()
        print("="*70)
        print("[SUCCESS] Qwen 3.5 0.8B is ready for distillation!")
        print("="*70)

        # Save teacher info for training script
        checkpoints_dir = Path("checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)

        teacher_info = {
            "model_path": model_path,
            "parameters": num_params,
            "vocab_size": len(tokenizer),
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_hidden_layers,
            "num_heads": model.config.num_attention_heads,
            "max_position_embeddings": model.config.max_position_embeddings,
            "device": str(device),
            "dtype": "float16"
        }

        torch.save(teacher_info, checkpoints_dir / "qwen35_teacher_info.pt")
        print(f"[SAVED] Teacher info to {checkpoints_dir / 'qwen35_teacher_info.pt'}")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print()
        print("[DEBUG] Full error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
