"""
BDH Web comparison app (backend + static file server).

Features:
- Prompt-based side-by-side generation: BDH vs DistilGPT2.
- Real comparison metrics (latency, tokens/sec, prompt perplexity).
- Quick benchmark endpoint using TinyStories samples.
- Serves static frontend from webapp/static.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import random
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = WEBAPP_ROOT / "static"
VISUALIZATION_DIR = REPO_ROOT / "visualization"
BENCHMARK_RESULTS_DIR = REPO_ROOT / "benchmarking" / "results"

sys.path.insert(0, str(REPO_ROOT))
from implementation.multiscale_bdh import MultiScaleBDH, MultiScaleBDHConfig  # noqa: E402
import implementation.multiscale_bdh as _multiscale_bdh_module  # noqa: E402

# Some older checkpoints reference module path "multiscale_bdh" during unpickling.
sys.modules.setdefault("multiscale_bdh", _multiscale_bdh_module)


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _read_lines_sample(path: Path, sample_count: int, seed: int) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        rows = [x.strip() for x in f if x.strip()]
    if not rows:
        raise RuntimeError(f"No non-empty lines found in data file: {path}")
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: min(sample_count, len(rows))]


def _json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class AppConfig:
    host: str
    port: int
    checkpoint_path: Path
    data_path: Path
    teacher_model_id: str
    distil_model_id: str
    bdh_n_embd: int
    bdh_n_layer: int
    bdh_n_head: int
    bdh_ffn_dim: int
    bdh_max_seq_len: int
    bdh_hebbian_lr: float
    default_top_k: int
    eager_load: bool


class ByteLevelTokenizer:
    """Minimal byte tokenizer fallback for small-vocabulary checkpoints."""

    def __init__(self, vocab_size: int):
        self.vocab_size = max(1, int(vocab_size))
        self.eos_token_id = None
        self.pad_token_id = 0
        self.eos_token = None
        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"

    def __len__(self) -> int:
        return self.vocab_size

    def _clip(self, token_id: int) -> int:
        if token_id < 0 or token_id >= self.vocab_size:
            return self.pad_token_id
        return token_id

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ):
        ids = [self._clip(b) for b in text.encode("utf-8", errors="ignore")]
        if truncation and max_length is not None and max_length > 0:
            ids = ids[:max_length]
        if not ids:
            ids = [self.pad_token_id]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = False,
        max_length: Optional[int] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        input_ids = self.encode(
            text,
            return_tensors="pt",
            truncation=truncation,
            max_length=max_length,
        )
        attention_mask = torch.ones_like(input_ids)
        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
        }

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        byte_vals = []
        for token_id in token_ids:
            tid = int(token_id)
            if 0 <= tid < 256:
                byte_vals.append(tid)

        if not byte_vals:
            return ""
        return bytes(byte_vals).decode("utf-8", errors="ignore")


class ModelService:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._runtime_warnings: List[str] = []
        self._bdh_model: Optional[MultiScaleBDH] = None
        self._bdh_tokenizer: Optional[AutoTokenizer] = None
        self._bdh_step: int = 0
        self._bdh_runtime_config: Dict[str, Any] = {}
        self._distil_model: Optional[AutoModelForCausalLM] = None
        self._distil_tokenizer: Optional[AutoTokenizer] = None

    @staticmethod
    def _param_count_millions(model: torch.nn.Module) -> float:
        return round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    @staticmethod
    def _sanitize_bpe_artifacts(text: str) -> str:
        if not text:
            return text
        if "Ġ" not in text and "Ċ" not in text and "ĉ" not in text:
            return text
        cleaned = text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
        cleaned = re.sub(r" {2,}", " ", cleaned)
        cleaned = re.sub(r" *\n *", "\n", cleaned)
        return cleaned

    @staticmethod
    def _tokenizer_vs_model_vocab(tokenizer: Any, model_vocab_size: int) -> Dict[str, int]:
        tok_vocab = int(len(tokenizer))
        return {
            "tokenizer_vocab_size": tok_vocab,
            "model_vocab_size": int(model_vocab_size),
            "vocab_padding_tokens": int(max(model_vocab_size - tok_vocab, 0)),
        }

    @staticmethod
    def _is_cuda_runtime_error(exc: Exception) -> bool:
        if not isinstance(exc, RuntimeError):
            return False
        msg = str(exc).lower()
        return (
            "cuda error" in msg
            or "cudnn" in msg
            or "cublas" in msg
            or "device-side assert" in msg
        )

    def _switch_to_cpu_and_reload(self, exc: Exception) -> bool:
        if self.device.type != "cuda":
            return False

        summary = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        note = f"CUDA failure detected; switching inference to CPU ({summary})."
        self._runtime_warnings.append(note)
        print(f"[webapp] {note}")

        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        with self._lock:
            self.device = torch.device("cpu")
            self._bdh_model = None
            self._distil_model = None

        self.ensure_models_loaded()
        return True

    def status(self) -> Dict[str, Any]:
        return {
            "time_utc": utc_now(),
            "device": str(self.device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "runtime_warnings": self._runtime_warnings[-5:],
            "models_loaded": {
                "bdh": self._bdh_model is not None,
                "distilgpt2": self._distil_model is not None,
            },
            "checkpoint_path": str(self.cfg.checkpoint_path),
            "data_path": str(self.cfg.data_path),
        }

    def ensure_models_loaded(self) -> None:
        with self._lock:
            if self._bdh_model is None:
                self._load_bdh()
            if self._distil_model is None:
                self._load_distil()

    def model_cards(self) -> Dict[str, Any]:
        self.ensure_models_loaded()

        bdh_cfg = getattr(self._bdh_model, "config", None)

        bdh_card = {
            "name": "BDH (MultiScale)",
            "n_layer": getattr(bdh_cfg, "n_layer", self.cfg.bdh_n_layer),
            "n_head": getattr(bdh_cfg, "n_head", self.cfg.bdh_n_head),
            "n_embd": getattr(bdh_cfg, "n_embd", self.cfg.bdh_n_embd),
            "ffn_dim": getattr(bdh_cfg, "ffn_dim", self.cfg.bdh_ffn_dim),
            "max_seq_len": getattr(bdh_cfg, "max_seq_len", self.cfg.bdh_max_seq_len),
            "hebbian_lr": getattr(bdh_cfg, "hebbian_lr", self.cfg.bdh_hebbian_lr),
            "logical_layers": len(getattr(bdh_cfg, "decay_rates", []) or []),
            "model_vocab_size": self._bdh_runtime_config.get("model_vocab_size"),
            "tokenizer_vocab_size": self._bdh_runtime_config.get("tokenizer_vocab_size"),
            "vocab_padding_tokens": self._bdh_runtime_config.get("vocab_padding_tokens"),
            "teacher_model": self._bdh_runtime_config.get("teacher_model"),
            "step": self._bdh_step,
            "params_m": self._param_count_millions(self._bdh_model),
            "features": [
                "Multi-scale Hebbian traces",
                "Teacher-guided distillation",
                "Top-k constrained sampling",
                "Short-context efficiency focus",
            ],
        }

        distil_cfg = self._distil_model.config if self._distil_model is not None else None
        distil_card = {
            "name": self.cfg.distil_model_id,
            "n_layer": int(getattr(distil_cfg, "n_layer", 0) or 0) or None,
            "n_head": int(getattr(distil_cfg, "n_head", 0) or 0) or None,
            "n_embd": int(getattr(distil_cfg, "n_embd", 0) or 0) or None,
            "ffn_dim": int(getattr(distil_cfg, "n_inner", 0) or 0) or None,
            "max_seq_len": int(getattr(distil_cfg, "n_positions", 0) or 0) or None,
            "hebbian_lr": None,
            "logical_layers": None,
            "params_m": self._param_count_millions(self._distil_model) if self._distil_model else None,
            "features": [
                "Transformer decoder",
                "No Hebbian adapters",
                "Greedy positional attention",
            ],
        }

        return {"bdh": bdh_card, "distilgpt2": distil_card}

    def _load_bdh(self) -> None:
        if not self.cfg.checkpoint_path.exists():
            raise FileNotFoundError(f"BDH checkpoint not found: {self.cfg.checkpoint_path}")

        checkpoint = torch.load(self.cfg.checkpoint_path, map_location=self.device, weights_only=False)

        # Accept both new-style ("model") and training-style ("model_state_dict") checkpoints.
        model_state = None
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint:
                model_state = checkpoint[key]
                break

        if model_state is None:
            raise KeyError("BDH checkpoint missing model parameters (expected 'model' or 'model_state_dict')")

        if "token_embedding.weight" not in model_state:
            raise KeyError("BDH checkpoint model state is missing token_embedding.weight")

        runtime_cfg = checkpoint.get("config", checkpoint.get("cfg", {})) or {}

        def _cfg_get(name: str, default: Any) -> Any:
            if isinstance(runtime_cfg, dict):
                return runtime_cfg.get(name, default)
            return getattr(runtime_cfg, name, default)

        seq_len = int(_cfg_get("seq_len", _cfg_get("max_seq_len", self.cfg.bdh_max_seq_len)))
        vocab_size = int(model_state["token_embedding.weight"].shape[0])
        teacher_model = str(_cfg_get("teacher_model", self.cfg.teacher_model_id))

        tokenizer: Optional[AutoTokenizer] = None
        tokenizer_path = _cfg_get("tokenizer_path", None)
        candidates = []
        if tokenizer_path:
            tp = Path(tokenizer_path)
            candidates.append(tp if tp.is_absolute() else (REPO_ROOT / tp))
        candidates.append(REPO_ROOT / "tokenizer-model")
        candidates.append(REPO_ROOT / "tokenizers")
        candidates.append(REPO_ROOT / "tokenizers" / "bbpe_tokenizer.json")

        def _load_local_tokenizer(path: Path) -> Optional[PreTrainedTokenizerFast]:
            if path.is_file() and path.suffix == ".json":
                tok_file = path
            elif path.is_dir() and (path / "tokenizer.json").exists():
                tok_file = path / "tokenizer.json"
            else:
                return None
            tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_file))
            # Set reasonable specials
            vocab = tok.get_vocab()
            # Some local BBPE tokenizer exports in this repo have decoder=null,
            # which leaks raw token markers like "Ġ" in generated text.
            try:
                backend_decoder = getattr(tok.backend_tokenizer, "decoder", None)
                if backend_decoder is None and "Ġ" in vocab:
                    from tokenizers import decoders as token_decoders

                    tok.backend_tokenizer.decoder = token_decoders.ByteLevel()
            except Exception:
                # Keep tokenizer load resilient; fallback decode still works.
                pass
            if tok.eos_token is None:
                if "<|endoftext|>" in vocab:
                    tok.eos_token = "<|endoftext|>"
                elif "[SEP]" in vocab:
                    tok.eos_token = "[SEP]"
            if tok.unk_token is None:
                tok.unk_token = tok.eos_token or "[UNK]"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token or tok.unk_token
            return tok

        for cand in candidates:
            tok = _load_local_tokenizer(cand)
            if tok is None:
                continue
            if len(tok) != vocab_size:
                continue
            tokenizer = tok
            self._bdh_runtime_config["tokenizer_path"] = str(cand)
            break

        if tokenizer is None:
            # Small-vocab checkpoints (e.g., byte-level 256) can run without
            # external tokenizer artifacts using a simple byte fallback.
            if vocab_size <= 256:
                tokenizer = ByteLevelTokenizer(vocab_size)
                self._bdh_runtime_config["tokenizer_path"] = "byte-level-fallback"
            else:
                tokenizer = AutoTokenizer.from_pretrained(teacher_model, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                tok_vocab = int(len(tokenizer))
                # Allow padded model vocab (common in Qwen checkpoints) where model vocab >= tokenizer vocab.
                if tok_vocab > vocab_size:
                    raise ValueError(
                        f"Tokenizer vocab ({tok_vocab}) is larger than checkpoint vocab ({vocab_size}). "
                        "Please set a compatible tokenizer_path."
                    )
                if (vocab_size - tok_vocab) > 2048:
                    raise ValueError(
                        f"Tokenizer vocab ({tok_vocab}) too small for checkpoint vocab ({vocab_size}). "
                        "Please provide a tokenizer_path with closer vocab size."
                    )

        model_cfg = MultiScaleBDHConfig(
            vocab_size=vocab_size,
            n_embd=int(_cfg_get("n_embd", self.cfg.bdh_n_embd)),
            n_layer=int(_cfg_get("n_layer", self.cfg.bdh_n_layer)),
            n_head=int(_cfg_get("n_head", self.cfg.bdh_n_head)),
            ffn_dim=int(_cfg_get("ffn_dim", self.cfg.bdh_ffn_dim)),
            dropout=float(_cfg_get("dropout", 0.1)),
            max_seq_len=int(_cfg_get("max_seq_len", seq_len)),
            decay_rates=list(_cfg_get("decay_rates", [0.95, 0.99, 0.995])),
            num_scales=int(_cfg_get("num_scales", 3)),
            scale_weights=list(_cfg_get("scale_weights", [0.2, 0.3, 0.5])),
            hebbian_lr=float(_cfg_get("hebbian_lr", self.cfg.bdh_hebbian_lr)),
            init_states=str(_cfg_get("init_states", "zeros")),
        )

        model = MultiScaleBDH(model_cfg).to(self.device)
        model.load_state_dict(model_state, strict=True)
        model.eval()

        self._bdh_model = model
        self._bdh_tokenizer = tokenizer
        self._bdh_step = int(
            checkpoint.get("step")
            or checkpoint.get("global_step")
            or checkpoint.get("batch")
            or 0
        )
        self._bdh_runtime_config = {
            "teacher_model": teacher_model,
            "seq_len": seq_len,
            "top_k": int(_cfg_get("top_k", self.cfg.default_top_k)),
            "teacher_replicas": _cfg_get("teacher_replicas", None),
            "batch_size": _cfg_get("batch_size", None),
            **self._tokenizer_vs_model_vocab(tokenizer, vocab_size),
        }

    def _load_distil(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.distil_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.distil_model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        model = model.to(self.device)
        model.eval()
        self._distil_tokenizer = tokenizer
        self._distil_model = model

    def _generate_bdh(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> Tuple[str, Dict[str, Any]]:
        assert self._bdh_model is not None and self._bdh_tokenizer is not None
        tokenizer = self._bdh_tokenizer
        model = self._bdh_model

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids.clone()
        start = time.time()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                window = generated[:, -model.config.max_seq_len :]
                logits, _ = model(window)
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-6)

                tokenizer_vocab_size = int(self._bdh_runtime_config.get("tokenizer_vocab_size", len(tokenizer)))
                effective_vocab = min(int(next_token_logits.shape[-1]), tokenizer_vocab_size)
                if effective_vocab <= 0:
                    raise RuntimeError("Invalid tokenizer vocabulary size for generation.")
                next_token_logits = next_token_logits[:, :effective_vocab]

                k = min(int(top_k), int(next_token_logits.shape[-1]))
                if k > 0:
                    values, _ = torch.topk(next_token_logits, k=k)
                    kth = values[:, [-1]]
                    next_token_logits = torch.where(
                        next_token_logits < kth,
                        torch.full_like(next_token_logits, -float("inf")),
                        next_token_logits,
                    )

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
                    break

        elapsed = time.time() - start
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        output_text = self._sanitize_bpe_artifacts(output_text)
        generated_tokens = int(generated.shape[1] - input_ids.shape[1])
        stats = {
            "elapsed_sec": elapsed,
            "generated_tokens": generated_tokens,
            "tokens_per_sec": generated_tokens / elapsed if elapsed > 0 else None,
            "input_tokens": int(input_ids.shape[1]),
            "total_tokens": int(generated.shape[1]),
        }
        return output_text, stats

    def _generate_distil(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> Tuple[str, Dict[str, Any]]:
        assert self._distil_model is not None and self._distil_tokenizer is not None
        tokenizer = self._distil_tokenizer
        model = self._distil_model

        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)

        start = time.time()
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-6),
                top_k=max(1, int(top_k)),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.time() - start
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_tokens = int(output.shape[1] - input_ids.shape[1])
        stats = {
            "elapsed_sec": elapsed,
            "generated_tokens": generated_tokens,
            "tokens_per_sec": generated_tokens / elapsed if elapsed > 0 else None,
            "input_tokens": int(input_ids.shape[1]),
            "total_tokens": int(output.shape[1]),
        }
        return output_text, stats

    def _prompt_perplexity_bdh(self, prompt: str) -> Optional[float]:
        assert self._bdh_model is not None and self._bdh_tokenizer is not None
        ids = self._bdh_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        if ids.shape[1] <= 1:
            return None
        ids = ids[:, -self._bdh_model.config.max_seq_len :]
        with torch.no_grad():
            logits, _ = self._bdh_model(ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
        return float(math.exp(float(loss.item())))

    def _prompt_perplexity_distil(self, prompt: str) -> Optional[float]:
        assert self._distil_model is not None and self._distil_tokenizer is not None
        enc = self._distil_tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        if ids.shape[1] <= 1:
            return None
        with torch.no_grad():
            out = self._distil_model(input_ids=ids, labels=ids)
            loss = float(out.loss.item())
        return float(math.exp(loss))

    def generate_compare(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> Dict[str, Any]:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        self.ensure_models_loaded()
        fallback_used = False
        try:
            with self._lock:
                bdh_text, bdh_stats = self._generate_bdh(prompt, max_new_tokens, temperature, top_k)
                distil_text, distil_stats = self._generate_distil(prompt, max_new_tokens, temperature, top_k)
                bdh_ppl = self._prompt_perplexity_bdh(prompt)
                distil_ppl = self._prompt_perplexity_distil(prompt)
        except RuntimeError as e:
            if not self._is_cuda_runtime_error(e):
                raise
            if not self._switch_to_cpu_and_reload(e):
                raise

            fallback_used = True
            with self._lock:
                bdh_text, bdh_stats = self._generate_bdh(prompt, max_new_tokens, temperature, top_k)
                distil_text, distil_stats = self._generate_distil(prompt, max_new_tokens, temperature, top_k)
                bdh_ppl = self._prompt_perplexity_bdh(prompt)
                distil_ppl = self._prompt_perplexity_distil(prompt)

        return {
            "prompt": prompt,
            "settings": {
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "top_k": int(top_k),
            },
            "runtime": {
                "device": str(self.device),
                "cuda_fallback_used": fallback_used,
                "warnings": self._runtime_warnings[-5:],
            },
            "bdh": {
                "model": "BDH (MultiScale)",
                "checkpoint": str(self.cfg.checkpoint_path),
                "step": self._bdh_step,
                "text": bdh_text,
                "stats": bdh_stats,
                "prompt_perplexity": bdh_ppl,
                "runtime_config": self._bdh_runtime_config,
            },
            "distilgpt2": {
                "model": self.cfg.distil_model_id,
                "text": distil_text,
                "stats": distil_stats,
                "prompt_perplexity": distil_ppl,
            },
            "comparison": {
                "faster_model": (
                    "bdh"
                    if (bdh_stats["tokens_per_sec"] or 0.0) >= (distil_stats["tokens_per_sec"] or 0.0)
                    else "distilgpt2"
                ),
                "lower_prompt_perplexity_model": (
                    "bdh"
                    if (bdh_ppl is not None and distil_ppl is not None and bdh_ppl <= distil_ppl)
                    else "distilgpt2"
                )
                if (bdh_ppl is not None and distil_ppl is not None)
                else None,
            },
            "time_utc": utc_now(),
        }

    def _perplexity_hf(self, stories: List[str], max_len: int) -> Tuple[float, float]:
        assert self._distil_model is not None and self._distil_tokenizer is not None
        model = self._distil_model
        tokenizer = self._distil_tokenizer
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for s in stories:
                enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=max_len)
                ids = enc["input_ids"].to(self.device)
                if ids.shape[1] <= 1:
                    continue
                out = model(input_ids=ids, labels=ids)
                tok_count = int(ids.shape[1] - 1)
                total_loss += float(out.loss.item()) * tok_count
                total_tokens += tok_count
        avg_loss = total_loss / max(total_tokens, 1)
        return float(math.exp(avg_loss)), float(avg_loss)

    def _perplexity_bdh(self, stories: List[str], max_len: int) -> Tuple[float, float]:
        assert self._bdh_model is not None and self._bdh_tokenizer is not None
        model = self._bdh_model
        tokenizer = self._bdh_tokenizer
        total_loss = 0.0
        total_tokens = 0
        max_len = min(max_len, model.config.max_seq_len)
        with torch.no_grad():
            for s in stories:
                enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=max_len)
                ids = enc["input_ids"].to(self.device)
                if ids.shape[1] <= 1:
                    continue
                logits, _ = model(ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )
                total_loss += float(loss.item())
                total_tokens += int(shift_labels.numel())
        avg_loss = total_loss / max(total_tokens, 1)
        return float(math.exp(avg_loss)), float(avg_loss)

    def _logic_score_hf(self) -> int:
        assert self._distil_model is not None and self._distil_tokenizer is not None
        tests = [
            ("Timmy has a blue ball. He gave the ball to Sarah. Who has the ball now? ", ["sarah"]),
            ("The sun is very hot. If you touch it, you will feel ", ["hot", "pain", "hurt"]),
        ]
        score = 0
        with torch.no_grad():
            for prompt, targets in tests:
                ids = self._distil_tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
                out = self._distil_model.generate(
                    ids,
                    max_new_tokens=8,
                    temperature=0.2,
                    do_sample=False,
                    pad_token_id=self._distil_tokenizer.eos_token_id,
                    eos_token_id=self._distil_tokenizer.eos_token_id,
                )
                continuation_ids = out[0, ids.shape[1] :]
                txt = self._distil_tokenizer.decode(continuation_ids, skip_special_tokens=True).lower()
                if any(t in txt for t in targets):
                    score += 10
        return score

    def _logic_score_bdh(self) -> int:
        tests = [
            ("Timmy has a blue ball. He gave the ball to Sarah. Who has the ball now? ", ["sarah"]),
            ("The sun is very hot. If you touch it, you will feel ", ["hot", "pain", "hurt"]),
        ]
        score = 0
        for prompt, targets in tests:
            generated = self._generate_bdh(prompt, max_new_tokens=8, temperature=0.2, top_k=40)[0]
            gl = generated.lower()
            pl = prompt.lower()
            continuation = gl[len(pl) :] if gl.startswith(pl) else gl
            if any(t in continuation for t in targets):
                score += 10
        return score

    def quick_benchmark(self, sample_count: int, max_len: int, seed: int) -> Dict[str, Any]:
        if sample_count < 1:
            raise ValueError("sample_count must be >= 1")
        if max_len < 8:
            raise ValueError("max_len must be >= 8")

        self.ensure_models_loaded()
        stories = _read_lines_sample(self.cfg.data_path, sample_count=sample_count, seed=seed)

        with self._lock:
            bdh_ppl, bdh_loss = self._perplexity_bdh(stories, max_len=max_len)
            distil_ppl, distil_loss = self._perplexity_hf(stories, max_len=max_len)
            bdh_logic = self._logic_score_bdh()
            distil_logic = self._logic_score_hf()

        result = {
            "time_utc": utc_now(),
            "sample_count": int(sample_count),
            "max_len": int(max_len),
            "seed": int(seed),
            "bdh": {
                "checkpoint": str(self.cfg.checkpoint_path),
                "step": self._bdh_step,
                "perplexity": bdh_ppl,
                "loss": bdh_loss,
                "logic_score": bdh_logic,
            },
            "distilgpt2": {
                "model_id": self.cfg.distil_model_id,
                "perplexity": distil_ppl,
                "loss": distil_loss,
                "logic_score": distil_logic,
            },
            "comparison": {
                "bdh_beats_distilgpt2_on_perplexity": bool(bdh_ppl < distil_ppl),
                "bdh_beats_distilgpt2_on_logic": bool(bdh_logic >= distil_logic),
            },
        }

        BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = BENCHMARK_RESULTS_DIR / "webapp_bdh_vs_distilgpt2.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def list_artifacts(self) -> Dict[str, Any]:
        artifacts: List[Dict[str, Any]] = []
        if BENCHMARK_RESULTS_DIR.exists():
            for path in sorted(BENCHMARK_RESULTS_DIR.glob("*.json")):
                item: Dict[str, Any] = {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
                try:
                    data = _json_load(path)
                    if isinstance(data, dict):
                        item["top_level_keys"] = list(data.keys())
                    artifacts.append(item)
                except json.JSONDecodeError:
                    item["error"] = "invalid_json"
                    artifacts.append(item)

        visuals = []
        if VISUALIZATION_DIR.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
                for p in sorted(VISUALIZATION_DIR.glob(ext)):
                    visuals.append({"name": p.name, "url": f"/viz/{p.name}"})

        return {"artifacts": artifacts, "visualizations": visuals, "time_utc": utc_now()}

    def latest_benchmark(self) -> Optional[Dict[str, Any]]:
        preferred = [
            BENCHMARK_RESULTS_DIR / "webapp_bdh_vs_distilgpt2.json",
            BENCHMARK_RESULTS_DIR / "bdh_vs_distilgpt2.json",
            BENCHMARK_RESULTS_DIR / "benchmark_results.json",
            BENCHMARK_RESULTS_DIR / "test_results.json",
        ]
        for p in preferred:
            if p.exists():
                payload = _json_load(p)
                return {"source": str(p), "payload": payload}
        return None


class RequestHandler(BaseHTTPRequestHandler):
    service: ModelService

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[webapp] {self.address_string()} - {fmt % args}")

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        content_len = int(self.headers.get("Content-Length", "0"))
        if content_len <= 0:
            return {}
        body = self.rfile.read(content_len)
        return json.loads(body.decode("utf-8"))

    def _serve_file(self, path: Path, status: HTTPStatus = HTTPStatus.OK) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": f"File not found: {path.name}"}, status=HTTPStatus.NOT_FOUND)
            return

        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = unquote(parsed.path)

        if route == "/":
            self._serve_file(STATIC_DIR / "index.html")
            return
        if route.startswith("/static/"):
            rel = route[len("/static/") :].lstrip("/")
            target = (STATIC_DIR / rel).resolve()
            if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
                self._send_json({"error": "Invalid static path"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._serve_file(target)
            return
        if route.startswith("/viz/"):
            rel = route[len("/viz/") :].lstrip("/")
            target = (VISUALIZATION_DIR / rel).resolve()
            if VISUALIZATION_DIR.resolve() not in target.parents and target != VISUALIZATION_DIR.resolve():
                self._send_json({"error": "Invalid visualization path"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._serve_file(target)
            return
        if route == "/api/health":
            self._send_json(self.service.status())
            return
        if route == "/api/config":
            self.service.ensure_models_loaded()
            if self.service._bdh_model is None:
                self._send_json({"error": "BDH model failed to load"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            bdh_cfg = getattr(self.service._bdh_model, "config", None)
            self._send_json(
                {
                    "checkpoint_path": str(self.service.cfg.checkpoint_path),
                    "data_path": str(self.service.cfg.data_path),
                    "teacher_model_id": self.service.cfg.teacher_model_id,
                    "distil_model_id": self.service.cfg.distil_model_id,
                    "bdh_architecture": {
                        "n_embd": getattr(bdh_cfg, "n_embd", self.service.cfg.bdh_n_embd),
                        "n_layer": getattr(bdh_cfg, "n_layer", self.service.cfg.bdh_n_layer),
                        "n_head": getattr(bdh_cfg, "n_head", self.service.cfg.bdh_n_head),
                        "ffn_dim": getattr(bdh_cfg, "ffn_dim", self.service.cfg.bdh_ffn_dim),
                        "max_seq_len": getattr(bdh_cfg, "max_seq_len", self.service.cfg.bdh_max_seq_len),
                        "hebbian_lr": getattr(bdh_cfg, "hebbian_lr", self.service.cfg.bdh_hebbian_lr),
                    },
                    "bdh_runtime": self.service._bdh_runtime_config,
                    "default_top_k": self.service.cfg.default_top_k,
                    "model_cards": self.service.model_cards(),
                }
            )
            return
        if route == "/api/artifacts":
            self._send_json(self.service.list_artifacts())
            return
        if route == "/api/benchmark/latest":
            data = self.service.latest_benchmark()
            if data is None:
                self._send_json({"error": "No benchmark result found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(data)
            return

        self._send_json({"error": f"Unknown route: {route}"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = unquote(parsed.path)

        try:
            payload = self._read_json_body()
        except json.JSONDecodeError as e:
            self._send_json({"error": f"Invalid JSON body: {e}"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            if route == "/api/generate":
                prompt = str(payload.get("prompt", ""))
                max_new_tokens = int(payload.get("max_new_tokens", 100))
                temperature = float(payload.get("temperature", 0.8))
                top_k = int(payload.get("top_k", self.service.cfg.default_top_k))
                result = self.service.generate_compare(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
                self._send_json(result)
                return

            if route == "/api/benchmark/quick":
                sample_count = int(payload.get("sample_count", 30))
                max_len = int(payload.get("max_len", 192))
                seed = int(payload.get("seed", 42))
                result = self.service.quick_benchmark(sample_count=sample_count, max_len=max_len, seed=seed)
                self._send_json(result)
                return

            self._send_json({"error": f"Unknown route: {route}"}, status=HTTPStatus.NOT_FOUND)
        except (ValueError, FileNotFoundError, RuntimeError, KeyError) as e:
            self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as e:
            traceback.print_exc()
            self._send_json(
                {"error": f"Unhandled server error: {type(e).__name__}: {e}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def parse_args() -> AppConfig:
    p = argparse.ArgumentParser(description="BDH web interface server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--checkpoint", default="checkpoints/jarvis_l4_super/latest.pt")
    p.add_argument("--data", default="data/tinystories.txt")
    p.add_argument("--teacher-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--distil-model", default="distilbert/distilgpt2")
    p.add_argument("--bdh-n-embd", type=int, default=512)
    p.add_argument("--bdh-n-layer", type=int, default=10)
    p.add_argument("--bdh-n-head", type=int, default=8)
    p.add_argument("--bdh-ffn-dim", type=int, default=2048)
    p.add_argument("--bdh-max-seq-len", type=int, default=192)
    p.add_argument("--bdh-hebbian-lr", type=float, default=0.0006)
    p.add_argument("--default-top-k", type=int, default=40)
    p.add_argument("--eager-load", action="store_true")
    args = p.parse_args()

    cfg = AppConfig(
        host=args.host,
        port=args.port,
        checkpoint_path=(REPO_ROOT / args.checkpoint).resolve()
        if not Path(args.checkpoint).is_absolute()
        else Path(args.checkpoint),
        data_path=(REPO_ROOT / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data),
        teacher_model_id=args.teacher_model,
        distil_model_id=args.distil_model,
        bdh_n_embd=args.bdh_n_embd,
        bdh_n_layer=args.bdh_n_layer,
        bdh_n_head=args.bdh_n_head,
        bdh_ffn_dim=args.bdh_ffn_dim,
        bdh_max_seq_len=args.bdh_max_seq_len,
        bdh_hebbian_lr=args.bdh_hebbian_lr,
        default_top_k=args.default_top_k,
        eager_load=bool(args.eager_load),
    )
    return cfg


def main() -> int:
    if not STATIC_DIR.exists():
        raise FileNotFoundError(f"Static directory missing: {STATIC_DIR}")

    cfg = parse_args()
    service = ModelService(cfg)
    if cfg.eager_load:
        service.ensure_models_loaded()

    RequestHandler.service = service
    server = ThreadingHTTPServer((cfg.host, cfg.port), RequestHandler)
    print(f"[webapp] Serving on http://{cfg.host}:{cfg.port}")
    print(f"[webapp] BDH checkpoint: {cfg.checkpoint_path}")
    print(f"[webapp] Distil baseline: {cfg.distil_model_id}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[webapp] Stopping server...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

