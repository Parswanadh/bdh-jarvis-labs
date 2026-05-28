import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


def measure_metrics(model, tokenizer, prompt, device):
    # Warmup
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    _ = model.generate(inputs.input_ids, max_new_tokens=1)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()

    # Generation
    if hasattr(model, "generate"):
        if "MultiScaleBDH" in str(type(model)):
            out = model.generate(inputs.input_ids, max_new_tokens=50, temperature=0.2)
        else:
            out = model.generate(inputs.input_ids, max_new_tokens=50, do_sample=False)
    else:
        raise AttributeError("Model has no generate method")

    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.time()

    tokens_generated = out.shape[1] - inputs.input_ids.shape[1]
    tokens_per_sec = tokens_generated / (t1 - t0)

    # VRAM (approximate for CUDA)
    vram = 0
    if device.type == "cuda":
        vram = torch.cuda.max_memory_allocated() / (1024**2)

    return tokens_per_sec, vram, tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = "Once upon a time, in a land far away, there was a small"

    print(f"Device: {device}")

    # --- DistilGPT-2 ---
    print("\nEvaluating DistilGPT-2...")
    gpt2_tok = AutoTokenizer.from_pretrained("distilgpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tps_gpt2, vram_gpt2, out_gpt2 = measure_metrics(gpt2_model, gpt2_tok, prompt, device)

    # --- BDH ---
    print("\nEvaluating BDH...")
    bdh_tok = AutoTokenizer.from_pretrained("Qwen3.5-0.8B", trust_remote_code=True)
    cfg = MultiScaleBDHConfig(
        vocab_size=248320,
        n_embd=256,
        n_layer=8,
        n_head=8,
        ffn_dim=1024,
        max_seq_len=192,
    )
    bdh_model = MultiScaleBDH(cfg).to(device)
    ckpt = torch.load("checkpoints/safe/laptop_safe.pt", map_location=device)
    bdh_model.load_state_dict(ckpt["model"])
    bdh_model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tps_bdh, vram_bdh, out_bdh = measure_metrics(bdh_model, bdh_tok, prompt, device)

    print("\n" + "=" * 30)
    print("Comparison Results")
    print("=" * 30)
    print(f"Metric            | DistilGPT-2  | BDH")
    print(f"------------------|--------------|-------")
    print(f"Tokens/Sec        | {tps_gpt2:12.2f} | {tps_bdh:6.2f}")
    print(f"VRAM (MB)         | {vram_gpt2:12.2f} | {vram_bdh:6.2f}")
    print("\nQualitative Output:")
    print(f"DistilGPT-2: {out_gpt2}")
    print(f"BDH: {out_bdh}")


if __name__ == "__main__":
    main()
