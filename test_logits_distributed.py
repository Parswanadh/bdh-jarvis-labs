import json
from pathlib import Path

from distribute_logits_core import (
    DistLogitsConfig,
    ProgressTracker,
    ShardRegistry,
    get_node_shard_range,
    verify_chunk_integrity,
)


def test_shard_ranges():
    ranges = [get_node_shard_range(i, 100000, 3) for i in (1, 2, 3)]
    assert ranges[0] == (0, 33333)
    assert ranges[1] == (33333, 66666)
    assert ranges[2] == (66666, 100000)
    assert sum(e - s for s, e in ranges) == 100000


def test_registry_init():
    root = Path("data/teacher_logits_shards_test")
    meta = root / "metadata.json"
    if meta.exists():
        meta.unlink()
    reg = ShardRegistry(meta)
    reg.initialize(DistLogitsConfig(output_root=str(root).replace("\\", "/")))
    data = reg.load()
    assert data["version"] == 1
    assert "node1" in data["nodes"]


def test_progress_roundtrip():
    p = Path("data/teacher_logits_shards_test/node1/progress.json")
    tr = ProgressTracker(p, 1)
    tr.mark_batch(8)
    tr2 = ProgressTracker(p, 1)
    assert tr2.state["stories_processed"] >= 8


def test_checksum_helpers():
    t = Path("data/teacher_logits_shards_test/check.txt")
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text("abc", encoding="utf-8")
    import hashlib
    expected = hashlib.sha256(b"abc").hexdigest()
    assert verify_chunk_integrity(t, expected)


if __name__ == "__main__":
    test_shard_ranges()
    test_registry_init()
    test_progress_roundtrip()
    test_checksum_helpers()
    print("distributed logits tests: PASS")

