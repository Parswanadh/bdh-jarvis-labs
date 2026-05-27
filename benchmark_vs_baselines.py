import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_eval_stories(path: Path, count: int = 100, seed: int = 42) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [x.strip() for x in f if x.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    return lines[:count]


def perplexity_hf(model, tokenizer, stories: List[str], device: torch.device, max_len: int = 192) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for s in stories:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=max_len)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] <= 1:
                continue
            out = model(input_ids=input_ids, labels=input_ids)
            token_count = int(input_ids.shape[1] - 1)
            total_loss += float(out.loss.item()) * token_count
            total_tokens += token_count
    avg = total_loss / max(total_tokens, 1)
    return math.exp(avg), avg


def perplexity_bdh(model, tokenizer, stories: List[str], device: torch.device, max_len: int = 192) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for s in stories:
            enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] <= 1:
                continue
            logits, _ = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += int(shift_labels.numel())
    avg = total_loss / max(total_tokens, 1)
    return math.exp(avg), avg


def logic_score_hf(model, tokenizer, device: torch.device) -> int:
    tests = [
        ("Timmy has a blue ball. He gave the ball to Sarah. Who has the ball now? ", ["sarah"]),
        ("The sun is very hot. If you touch it, you will feel ", ["hot", "pain", "hurt"]),
    ]
    score = 0
    with torch.no_grad():
        for prompt, targets in tests:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            out = model.generate(ids, max_new_tokens=8, temperature=0.2, do_sample=False)
            txt = tokenizer.decode(out[0], skip_special_tokens=True).lower()
            if any(t in txt for t in targets):
                score += 10
    return score


def logic_score_bdh(model, tokenizer, device: torch.device) -> int:
    tests = [
        ("Timmy has a blue ball. He gave the ball to Sarah. Who has the ball now? ", ["sarah"]),
        ("The sun is very hot. If you touch it, you will feel ", ["hot", "pain", "hurt"]),
    ]
    score = 0
    with torch.no_grad():
        for prompt, targets in tests:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            out = model.generate(ids, max_new_tokens=8, temperature=0.2)
            txt = tokenizer.decode(out[0], skip_special_tokens=True).lower()
            if any(t in txt for t in targets):
                score += 10
    return score


def load_bdh(device: torch.device, checkpoint: Path):
    qtok = AutoTokenizer.from_pretrained("Qwen3.5-0.8B", trust_remote_code=True)
    cfg = MultiScaleBDHConfig(
        vocab_size=248320,
        n_embd=256,
        n_layer=8,
        n_head=8,
        ffn_dim=1024,
        max_seq_len=192,
    )
    model = MultiScaleBDH(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model, qtok, int(ckpt.get("step", 0))


def run_model(name: str, model_id: str, stories: List[str], device: torch.device, max_len: int, force_fp16: bool) -> Dict:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": "auto" if device.type == "cuda" else None}
    if force_fp16 and device.type == "cuda":
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    ppl, loss = perplexity_hf(model, tok, stories, device, max_len=max_len)
    logic = logic_score_hf(model, tok, device)
    dt = time.time() - t0
    return {"name": name, "model_id": model_id, "perplexity": ppl, "loss": loss, "logic_score": logic, "elapsed_sec": dt}


def _fetch_model_card(model_id: str) -> str:
    import urllib.request

    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def model_card_notes(model_id: str) -> Dict[str, object]:
    notes: Dict[str, object] = {"model_id": model_id}
    try:
        card = _fetch_model_card(model_id)
        l = card.lower()
        weak_markers = []
        if "82 million" in l or "82m" in l:
            weak_markers.append("small-82m")
        if "70m" in l or "70 million" in l:
            weak_markers.append("very-small-70m")
        if "not intended for deployment" in l:
            weak_markers.append("research-only")
        if "perplexity" in l and "21.1" in l:
            weak_markers.append("reported-wikitext103-ppl-21.1")
        notes["weak_markers"] = weak_markers
        notes["card_fetched"] = True
    except Exception as e:
        notes["card_fetched"] = False
        notes["card_error"] = str(e)
    return notes


def main():
    parser = argparse.ArgumentParser(description="Benchmark BDH against weaker baselines")
    parser.add_argument("--checkpoint", default="checkpoints/safe/laptop_safe.pt")
    parser.add_argument("--stories", default="data/tinystories.txt")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baselines",
        default="distilbert/distilgpt2,EleutherAI/pythia-70m-deduped,sshleifer/tiny-gpt2",
        help="Comma-separated baseline HF model IDs",
    )
    parser.add_argument("--output", default="benchmarking/results/bdh_vs_baselines.json")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip-card-fetch", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stories = load_eval_stories(Path(args.stories), count=args.samples, seed=args.seed)

    bdh_model, bdh_tok, bdh_step = load_bdh(device, Path(args.checkpoint))
    bdh_ppl, bdh_loss = perplexity_bdh(bdh_model, bdh_tok, stories, device, max_len=args.max_len)
    bdh_logic = logic_score_bdh(bdh_model, bdh_tok, device)
    results: Dict[str, object] = {
        "device": str(device),
        "samples": args.samples,
        "max_len": args.max_len,
        "bdh": {
            "checkpoint": args.checkpoint,
            "step": bdh_step,
            "perplexity": bdh_ppl,
            "loss": bdh_loss,
            "logic_score": bdh_logic,
        },
        "baselines": [],
    }

    for bid in [x.strip() for x in args.baselines.split(",") if x.strip()]:
        try:
            bres = run_model(bid.split("/")[-1], bid, stories, device, args.max_len, args.fp16)
            if not args.skip_card_fetch:
                bres["model_card_notes"] = model_card_notes(bid)
            bres["bdh_beats_on_ppl"] = bool(bdh_ppl < bres["perplexity"])
            bres["bdh_beats_on_logic"] = bool(bdh_logic >= bres["logic_score"])
            results["baselines"].append(bres)
            print(
                f"{bid}: ppl={bres['perplexity']:.2f}, logic={bres['logic_score']} | "
                f"BDH ppl={bdh_ppl:.2f}, logic={bdh_logic}"
            )
        except Exception as e:
            results["baselines"].append({"name": bid, "error": str(e)})
            print(f"[WARN] baseline failed: {bid} -> {e}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

