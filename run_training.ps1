# BDH Training with Qwen 3.5 0.8B
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "BDH TRAINING - QWEN 3.5 0.8B TEACHER" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Activating bdh-fest environment..." -ForegroundColor Yellow
conda activate bdh-fest

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate bdh-fest environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Training will run for 2 hours with checkpoints every 30 minutes" -ForegroundColor Green
Write-Host "Both models will be on GPU - check Task Manager for VRAM usage!" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop training" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

python train_minimal_working.py

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "TRAINING COMPLETE!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Checkpoints saved to: checkpoints\minimal\" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
