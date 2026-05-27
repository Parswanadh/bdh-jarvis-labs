#!/bin/bash
# ============================================================================
# DOWNLOAD TRAINED MODEL FROM GITHUB
# ============================================================================
# Run this script locally to download the trained model from GitHub
# Use after uploading from Jarvis Labs

set -e

echo "============================================================"
echo "DOWNLOAD TRAINED MODEL FROM GITHUB"
echo "============================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "upload_to_github.sh" ]; then
    echo "[ERROR] Please run this script from the bdh-jarvis-labs directory"
    exit 1
fi

echo "[OK] In bdh-jarvis-labs directory"
echo ""

echo "============================================================"
echo "STEP 1: Pulling latest from GitHub (including checkpoints)"
echo "============================================================"
echo ""

# Pull from GitHub (including large files)
echo "Downloading from GitHub..."
git pull origin master

echo ""
echo "[OK] Repository updated"
echo ""

# Check if checkpoints exist
CHECKPOINT_DIR="checkpoints/bdh_a100"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[WARN] Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "[WARN] Model may not have been uploaded yet from Jarvis Labs"
    echo ""
    echo "To upload from Jarvis Labs, run:"
    echo "  ssh into Jarvis Labs"
    echo "  cd bdh-jarvis-labs"
    echo "  ./upload_to_github.sh"
    exit 1
fi

echo ""
echo "============================================================"
echo "STEP 2: Verifying downloaded checkpoints"
echo "============================================================"
echo ""

# Show checkpoint info
echo "Downloaded checkpoints:"
ls -lh "$CHECKPOINT_DIR"
echo ""

# Count total files
TOTAL_FILES=$(find "$CHECKPOINT_DIR" -type f | wc -l)
TOTAL_SIZE=$(du -sh "$CHECKPOINT_DIR" | cut -f1)

echo "Total files: $TOTAL_FILES"
echo "Total size: $TOTAL_SIZE"
echo ""

echo "============================================================"
echo "DOWNLOAD COMPLETE!"
echo "============================================================"
echo ""
echo "Your trained model is ready to use!"
echo ""
echo "Quick test:"
echo "  python -c \"import torch; ckpt = torch.load('$CHECKPOINT_DIR/checkpoint_best.pt', map_location='cpu'); print(f'Loaded checkpoint: epoch {ckpt[\"iteration\"]}, loss {ckpt[\"loss\"]:.4f}')\""
echo ""
echo "Use in your code:"
echo "  from implementation.multiscale_bdh import MultiScaleBDH"
echo "  model = MultiScaleBDH.from_pretrained('$CHECKPOINT_DIR/checkpoint_best.pt')"
echo ""
echo "============================================================"
