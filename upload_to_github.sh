#!/bin/bash
# ============================================================================
# UPLOAD TRAINED MODEL TO GITHUB
# ============================================================================
# Run this script on Jarvis Labs AFTER training completes
# It will commit and push the trained model checkpoints to GitHub

set -e

echo "============================================================"
echo "UPLOAD TRAINED MODEL TO GITHUB"
echo "============================================================"
echo ""

# Check if training completed
CHECKPOINT_DIR="checkpoints/bdh_a100"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[ERROR] Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "[ERROR] Please run training first: ./run_jarvis_training.sh"
    exit 1
fi

echo "[OK] Found checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Show checkpoint info
echo "Checkpoints to upload:"
ls -lh "$CHECKPOINT_DIR"
echo ""

# Count total files
TOTAL_FILES=$(find "$CHECKPOINT_DIR" -type f | wc -l)
TOTAL_SIZE=$(du -sh "$CHECKPOINT_DIR" | cut -f1)

echo "Total files: $TOTAL_FILES"
echo "Total size: $TOTAL_SIZE"
echo ""

# Ask for confirmation
read -p "Upload these checkpoints to GitHub? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "[CANCEL] Upload cancelled"
    exit 0
fi

echo ""
echo "============================================================"
echo "STEP 1: Committing checkpoints to git"
echo "============================================================"
echo ""

# Add checkpoints to git
git add checkpoints/bdh_a100/

# Create commit
git commit -m "Add trained A100 model checkpoints

- Model: Multi-Scale BDH (5M parameters)
- Training: Gemma 3 270M distillation
- Hardware: Jarvis Labs A100 80GB
- Epochs: 5
- Checkpoints: All epochs + best model
- Loss: ~2.2-2.5

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

echo "[OK] Checkpoints committed"
echo ""

echo "============================================================"
echo "STEP 2: Pushing to GitHub"
echo "============================================================"
echo ""

# Push to GitHub
echo "Pushing to origin/master..."
git push origin master

echo ""
echo "============================================================"
echo "UPLOAD COMPLETE!"
echo "============================================================"
echo ""
echo "Your trained model is now on GitHub!"
echo ""
echo "Download from anywhere:"
echo "  git clone https://github.com/Parswanadh/bdh-jarvis-labs"
echo ""
echo "Or download individual checkpoints from:"
echo "  https://github.com/Parswanadh/bdh-jarvis-labs/tree/master/checkpoints/bdh_a100"
echo ""
echo "============================================================"
