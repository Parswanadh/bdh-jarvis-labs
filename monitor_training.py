"""
BDH Training Monitor - Real-time Training Progress Monitor
======================================================

This script monitors the training progress and provides real-time updates.
Run this in a separate terminal while training is running.
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

def check_progress(checkpoint_dir="checkpoints/multiscale_bdh_reasoning"):
    """Check training progress from checkpoints."""

    print("="*60)
    print("BDH Training Monitor")
    print("="*60)
    print()

    if not os.path.exists(checkpoint_dir):
        print(f"⏳ Waiting for training to start...")
        print(f"   Looking for checkpoints in: {checkpoint_dir}")
        return

    # List all checkpoints
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])

    if len(checkpoints) == 0:
        print("⏳ No checkpoints found yet. Training may still be starting...")
        return

    print(f"📊 Found {len(checkpoints)} checkpoint(s)")
    print()

    # Check latest checkpoint
    latest = "checkpoint_latest.pt"
    latest_path = os.path.join(checkpoint_dir, latest)

    if os.path.exists(latest_path):
        try:
            checkpoint = torch.load(latest_path)
            iteration = checkpoint.get('iteration', 0)
            loss = checkpoint.get('loss', 0)
            timestamp = checkpoint.get('timestamp', 'Unknown')
            is_best = checkpoint.get('is_best', False)

            print(f"📈 LATEST CHECKPOINT:")
            print(f"   Iteration: {iteration}")
            print(f"   Loss: {loss:.4f}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Is Best: {'✅ YES' if is_best else '❌ NO'}")
            print()

            # Check if training is complete
            if iteration >= 1000:
                print("✅ TRAINING COMPLETE!")
            else:
                progress = (iteration / 1000) * 100
                print(f"⏳ Training progress: {progress:.1f}%")

        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")

    # Check best checkpoint
    best = "checkpoint_best.pt"
    best_path = os.path.join(checkpoint_dir, best)

    if os.path.exists(best_path):
        try:
            checkpoint = torch.load(best_path)
            best_loss = checkpoint.get('loss', 0)
            best_iter = checkpoint.get('iteration', 0)

            print(f"🏆 BEST CHECKPOINT:")
            print(f"   Iteration: {best_iter}")
            print(f"   Loss: {best_loss:.4f}")
            print()
        except Exception as e:
            print(f"⚠️  Error loading best checkpoint: {e}")

    # Check training config
    config_path = os.path.join(checkpoint_dir, "training_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            task = config.get('training_config', {}).get('task', 'Unknown')
            max_iters = config.get('training_config', {}).get('max_iters', 'Unknown')

            print(f"📋 TRAINING CONFIG:")
            print(f"   Task: {task}")
            print(f"   Max iterations: {max_iters}")
            print()
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")

    print("="*60)


def monitor_training(checkpoint_dir="checkpoints/multiscale_bdh_reasoning", interval=30):
    """Continuously monitor training progress."""

    print(f"🔄 Starting training monitor...")
    print(f"   Checking every {interval} seconds")
    print(f"   Press Ctrl+C to stop")
    print()

    iteration_count = 0
    last_iteration = 0

    try:
        while True:
            check_progress(checkpoint_dir)

            # Check if training is complete
            latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
            if os.path.exists(latest_path):
                try:
                    checkpoint = torch.load(latest_path)
                    current_iteration = checkpoint.get('iteration', 0)

                    if current_iteration >= 1000:
                        print("\n" + "="*60)
                        print("🎉 TRAINING COMPLETE!")
                        print("="*60)
                        print()
                        print("✅ Model is ready for science fair demo!")
                        print("✅ Checkpoints saved to:", checkpoint_dir)
                        print()
                        break

                    # Check if progress is being made
                    if current_iteration == last_iteration:
                        print(f"⏸️  Waiting for training progress... (iteration {current_iteration})")
                    else:
                        last_iteration = current_iteration
                        iteration_count += 1

                except Exception as e:
                    pass

            print()
            print(f"⏰ Next check in {interval} seconds...")
            print("-" * 40)
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("🛑 Monitoring stopped by user")
        print("="*60)


if __name__ == "__main__":
    import torch

    print("\n" + "="*60)
    print("BDH TRAINING MONITOR")
    print("="*60)
    print()
    print("This script will:")
    print("  1. Check for training checkpoints")
    print("  2. Display training progress")
    print("  3. Show loss and iteration numbers")
    print("  4. Alert when training is complete")
    print()
    print("Usage:")
    print("  python monitor_training.py")
    print()
    print("Monitoring starts in 5 seconds...")
    print()

    time.sleep(5)

    # Single check mode
    check_progress()

    # Uncomment below for continuous monitoring:
    # monitor_training(interval=30)
