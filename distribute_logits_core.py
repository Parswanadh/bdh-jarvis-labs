"""
Core utilities for deterministic 3-node logits generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class DistLogitsConfig:
    total_stories: int = 100000
    num_nodes: int = 3
    chunk_max_bytes: int = 220 * 1024 * 1024
    seq_len: int = 192
    batch_size: int = 8
    top_k: int = 256
    teacher_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    data_file: str = "data/tinystories.txt"
    output_root: str = "data/teacher_logits_shards"


def get_node_shard_range(node_id: int, total_stories: int = 100000, num_nodes: int = 3) -> Tuple[int, int]:
    if node_id < 1 or node_id > num_nodes:
        raise ValueError(f"node_id must be in [1, {num_nodes}], got {node_id}")
    per = total_stories // num_nodes
    start = (node_id - 1) * per
    end = node_id * per if node_id < num_nodes else total_stories
    return start, end


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_chunk_integrity(chunk_path: Path, expected_sha256: str) -> bool:
    return compute_sha256(chunk_path) == expected_sha256.strip().lower()


class ProgressTracker:
    def __init__(self, progress_file: Path, node_id: int):
        self.progress_file = progress_file
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.node_id = node_id
        self.state = self._load_or_init()

    def _load_or_init(self) -> Dict:
        if self.progress_file.exists():
            with self.progress_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "node_id": self.node_id,
            "status": "initializing",
            "started_at": _utc_ts(),
            "last_updated_at": _utc_ts(),
            "stories_processed": 0,
            "batches_processed": 0,
            "chunks_written": 0,
            "current_chunk_idx": 0,
            "last_story_offset": 0,
            "error": None,
        }

    def save(self) -> None:
        self.state["last_updated_at"] = _utc_ts()
        tmp = self.progress_file.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        tmp.replace(self.progress_file)

    def mark_batch(self, stories_inc: int, batch_inc: int = 1) -> None:
        self.state["status"] = "running"
        self.state["stories_processed"] += stories_inc
        self.state["batches_processed"] += batch_inc
        self.state["last_story_offset"] += stories_inc
        self.save()

    def mark_chunk(self, chunk_idx: int) -> None:
        self.state["chunks_written"] += 1
        self.state["current_chunk_idx"] = chunk_idx
        self.save()

    def mark_completed(self) -> None:
        self.state["status"] = "completed"
        self.save()

    def mark_error(self, msg: str) -> None:
        self.state["status"] = "error"
        self.state["error"] = msg
        self.save()


class ShardRegistry:
    def __init__(self, metadata_file: Path):
        self.metadata_file = metadata_file
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self, cfg: DistLogitsConfig) -> None:
        if self.metadata_file.exists():
            return
        data = {
            "version": 1,
            "created_at": _utc_ts(),
            "total_stories": cfg.total_stories,
            "seq_len": cfg.seq_len,
            "top_k": cfg.top_k,
            "teacher_model": cfg.teacher_model,
            "nodes": {
                "node1": {"status": "pending", "chunks": []},
                "node2": {"status": "pending", "chunks": []},
                "node3": {"status": "pending", "chunks": []},
            },
            "story_ranges": {
                "node1": list(get_node_shard_range(1, cfg.total_stories, cfg.num_nodes)),
                "node2": list(get_node_shard_range(2, cfg.total_stories, cfg.num_nodes)),
                "node3": list(get_node_shard_range(3, cfg.total_stories, cfg.num_nodes)),
            },
        }
        tmp = self.metadata_file.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.metadata_file)

    def load(self) -> Dict:
        with self.metadata_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: Dict) -> None:
        tmp = self.metadata_file.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.metadata_file)

    def register_chunk(self, node_id: int, chunk_idx: int, chunk_path: Path, sha256: str, rows: int) -> None:
        data = self.load()
        nk = f"node{node_id}"
        data["nodes"][nk]["chunks"].append(
            {
                "chunk_idx": chunk_idx,
                "path": str(chunk_path),
                "sha256": sha256,
                "rows": rows,
                "written_at": _utc_ts(),
            }
        )
        self._save(data)

    def mark_node_complete(self, node_id: int) -> None:
        data = self.load()
        nk = f"node{node_id}"
        data["nodes"][nk]["status"] = "completed"
        data["nodes"][nk]["completed_at"] = _utc_ts()
        self._save(data)


class ChunkWriter:
    """
    Writes compact top-k logits to chunked NPZ files.
    """

    def __init__(self, output_dir: Path, node_id: int, max_bytes: int, registry: ShardRegistry):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.node_id = node_id
        self.max_bytes = max_bytes
        self.registry = registry
        self.chunk_idx = 0
        self._reset()

    def _reset(self) -> None:
        self.buf_ids: List[np.ndarray] = []
        self.buf_topk_idx: List[np.ndarray] = []
        self.buf_topk_logits: List[np.ndarray] = []
        self.buf_rows = 0
        self.buf_bytes = 0

    def add_batch(self, input_ids: np.ndarray, topk_indices: np.ndarray, topk_logits: np.ndarray) -> Optional[Tuple[int, Path]]:
        est = input_ids.nbytes + topk_indices.nbytes + topk_logits.nbytes
        if self.buf_bytes > 0 and self.buf_bytes + est > self.max_bytes:
            flushed = self.flush()
            self._append(input_ids, topk_indices, topk_logits, est)
            return flushed
        self._append(input_ids, topk_indices, topk_logits, est)
        return None

    def _append(self, input_ids: np.ndarray, topk_indices: np.ndarray, topk_logits: np.ndarray, est: int) -> None:
        self.buf_ids.append(input_ids.astype(np.int32, copy=False))
        self.buf_topk_idx.append(topk_indices.astype(np.int32, copy=False))
        self.buf_topk_logits.append(topk_logits.astype(np.float16, copy=False))
        self.buf_rows += int(input_ids.shape[0])
        self.buf_bytes += est

    def flush(self) -> Optional[Tuple[int, Path]]:
        if self.buf_rows == 0:
            return None
        ids = np.concatenate(self.buf_ids, axis=0)
        idx = np.concatenate(self.buf_topk_idx, axis=0)
        vals = np.concatenate(self.buf_topk_logits, axis=0)

        chunk_path = self.output_dir / f"shard_node{self.node_id}_{self.chunk_idx:04d}.npz"
        tmp_path = chunk_path.with_suffix(".npz.tmp")
        with tmp_path.open("wb") as f:
            np.savez_compressed(
                f,
                input_ids=ids,
                topk_indices=idx,
                topk_logits=vals,
                rows=np.array([self.buf_rows], dtype=np.int32),
            )
        tmp_path.replace(chunk_path)

        sha = compute_sha256(chunk_path)
        sha_path = chunk_path.with_suffix(".sha256")
        with sha_path.open("w", encoding="utf-8") as f:
            f.write(sha + "\n")

        self.registry.register_chunk(self.node_id, self.chunk_idx, chunk_path, sha, self.buf_rows)
        out = (self.chunk_idx, chunk_path)
        self.chunk_idx += 1
        self._reset()
        return out

    def finalize(self) -> Optional[Tuple[int, Path]]:
        return self.flush()


class GitOperations:
    def __init__(self, repo_root: Path, node_id: int):
        self.repo_root = repo_root
        self.node_id = node_id

    def _run(self, args: List[str]) -> Tuple[int, str]:
        p = subprocess.run(args, cwd=str(self.repo_root), capture_output=True, text=True)
        return p.returncode, (p.stdout + "\n" + p.stderr).strip()

    def _current_branch(self) -> str:
        code, out = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        if code != 0:
            return "main"
        branch = out.strip().splitlines()[0] if out.strip() else "main"
        return branch or "main"

    def configure(self) -> None:
        code, out = self._run(["git", "rev-parse", "--is-inside-work-tree"])
        if code != 0:
            raise RuntimeError(f"Not a git repo: {out}")
        self._run(["git", "config", "--local", "user.name", f"BDH-Node{self.node_id}"])
        self._run(["git", "config", "--local", "user.email", f"node{self.node_id}@bdh.local"])

    def stage_commit_push(self, rel_path: str, max_retries: int = 3) -> None:
        self.configure()
        token = os.environ.get("GITHUB_TOKEN_BDH", "").strip()
        if not token:
            raise RuntimeError("GITHUB_TOKEN_BDH is required for --git-push")

        # Stage + commit (commit can be no-op)
        code, out = self._run(["git", "add", rel_path])
        if code != 0:
            raise RuntimeError(f"git add failed: {out}")
        msg = f"[logits] node{self.node_id} shards {_utc_ts()}"
        code, out = self._run(["git", "commit", "-m", msg])
        if code != 0 and "nothing to commit" not in out.lower():
            raise RuntimeError(f"git commit failed: {out}")

        # Tokenized remote url for this process only
        code, remote_url = self._run(["git", "remote", "get-url", "origin"])
        if code != 0:
            raise RuntimeError(f"git remote get-url failed: {remote_url}")
        remote_url = remote_url.strip()
        if remote_url.startswith("https://"):
            auth_url = remote_url.replace("https://", f"https://x-access-token:{token}@")
        else:
            raise RuntimeError("Only HTTPS remotes are supported for automated push")

        # Pull rebase before push (retry)
        branch = self._current_branch()
        for i in range(max_retries):
            code, out = self._run(["git", "pull", "--rebase", auth_url, branch])
            if code == 0:
                break
            if i == max_retries - 1:
                raise RuntimeError(f"git pull --rebase failed: {out}")
            time.sleep(2 ** i)

        for i in range(max_retries):
            code, out = self._run(["git", "push", auth_url, f"HEAD:{branch}"])
            if code == 0:
                return
            if i == max_retries - 1:
                raise RuntimeError(f"git push failed: {out}")
            time.sleep(2 ** i)

