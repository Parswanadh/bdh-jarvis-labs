import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, IterableDataset
from pathlib import Path
import sys
import time
import os
import argparse
import gc

# --- SETTINGS ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B" 
LOCAL_MODEL_PATH = Path("./Qwen3.5-0.8B")
DATA_PATH = Path("./data/tinystories.txt")
SEQ_LEN = 192
TOP_K = 512
BATCH_SIZE = 1 # Safety first

class DistributedStoryDataset(IterableDataset):
    """Splits dataset deterministically across N nodes"""
    def __init__(self, file_path, node_id, total_nodes, resume_from=0):
        self.file_path = file_path
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.resume_from = resume_from
        
    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip() and (i % self.total_nodes == self.node_id):
                    # Skip lines already processed
                    if i < self.resume_from:
                        continue
                    yield i, line.strip()

def save_logits_binary(indices, values, path):
    """Saves Top-K indices (int32) and values (float16)"""
    with open(path, 'ab') as f:
        f.write(indices.cpu().to(torch.int32).numpy().tobytes())
        f.write(values.cpu().to(torch.float16).numpy().tobytes())

def run_node(node_id, total_nodes, max_hours):
    device = torch.device('cuda')
    start_time = time.time()
    
    print(f"HYDRA NODE {node_id}/{total_nodes} STARTING...")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")

    # 1. Load Model
    model_path = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()

    # 2. Setup Paths & Checkpointing
    out_dir = Path(f"./logits_cache/node_{node_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "logits.bin"
    progress_file = out_dir / "progress.log"
    
    start_line = 0
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            try:
                start_line = int(f.read().strip())
                print(f"Resuming from story index {start_line}...")
            except:
                pass # Corrupted file, start fresh

    # 3. Data Processing Loop
    dataset = DistributedStoryDataset(DATA_PATH, node_id, total_nodes, resume_from=start_line)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    print(f"Node {node_id} processing its shard. Max hours: {max_hours or 'unlimited'}")
    
    last_story_idx = start_line
    try:
        with torch.no_grad():
            for i, (story_idx, story_text) in enumerate(dataloader):
                story_idx = story_idx.item()
                
                # Timed stop for Colab
                if max_hours and (time.time() - start_time) / 3600 >= max_hours:
                    print(f"[TIMED STOP] Reached {max_hours} hours. Shutting down cleanly.")
                    break
                
                ids = tokenizer(story_text, truncation=True, max_length=SEQ_LEN, padding='max_length', return_tensors='pt')['input_ids'].to(device)
                logits = teacher(ids).logits[0]
                
                vals, idxs = torch.topk(logits, TOP_K, dim=-1)
                save_logits_binary(idxs, vals, out_file)
                last_story_idx = story_idx

                if i % 100 == 0:
                    print(f"Processed story {i} (Index: {story_idx})")
                    
    except KeyboardInterrupt:
        print("
Interrupted. Saving progress...")
    finally:
        # Save last processed line number
        with open(progress_file, 'w') as f:
            f.write(str(last_story_idx))
        print(f"
Node {node_id} work session complete. Progress saved to line {last_story_idx}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, required=True, help="Node ID (e.g., 0 for local, 1 for Colab)")
    parser.add_argument("--total-nodes", type=int, default=2, help="Total number of nodes in the cluster")
    parser.add_argument("--max-hours", type=float, default=None, help="Optional max runtime in hours (for Colab)")
    args = parser.parse_args()
    
    if not DATA_PATH.exists():
        print(f"ERROR: Place tinystories.txt in {DATA_PATH.parent}")
        sys.exit(1)
        
    run_node(args.node, args.total_nodes, args.max_hours)
