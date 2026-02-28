#!/bin/bash
# ============================================================================
# MASTER SCRIPT - Complete BDH Distillation on Jarvis Labs
# ============================================================================
# This script automates the entire process on Jarvis Labs A100

set -e  # Exit on error

START_TIME=$(date +%s)

echo "============================================================"
echo "BDH DISTILLATION - JARVIS LABS A100 80GB"
echo "============================================================"
echo ""
echo "Start time: $(date)"
echo ""

# Phase 1: Setup
echo "============================================================"
echo "PHASE 1: SETUP (5 minutes)"
echo "============================================================"
echo ""

bash jarvis_setup.sh

echo ""
echo "============================================================"
echo "PHASE 2: GENERATE DATA (20-30 minutes)"
echo "============================================================"
echo ""

python generate_teacher_data.py

echo ""
echo "============================================================"
echo "PHASE 3: TRAIN BDH (30-40 minutes)"
echo "============================================================"
echo ""

python train_a100.py

# Calculate total time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))

echo ""
echo "============================================================"
echo "✅ TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Total time: ${MINUTES} minutes"
echo ""

# Check if under 1 hour
if [ $MINUTES -lt 60 ]; then
    echo "✅ SUCCESS: Under 1 hour!"
    echo "   Cost efficient!"
else
    echo "⚠️  WARNING: Over 1 hour"
    echo "   Consider reducing epochs or batch size"
fi

echo ""
echo "Checkpoints saved to: checkpoints/bdh_a100/"
echo "Best model: checkpoints/bdh_a100/checkpoint_best.pt"
echo ""
echo "Ready for science fair demo!"
echo "============================================================"
