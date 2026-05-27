@echo off
REM Start BDH Training with Qwen 3.5 0.8B
echo ========================================================================
echo BDH TRAINING - QWEN 3.5 0.8B TEACHER
echo ========================================================================
echo.

echo Activating bdh-fest environment...
call conda activate bdh-fest

if errorlevel 1 (
    echo ERROR: Failed to activate bdh-fest environment
    pause
    exit /b 1
)

echo.
echo Training will run for 2 hours with checkpoints every 30 minutes
echo Both models will be on GPU - you should see VRAM usage!
echo Press Ctrl+C to stop training
echo.
echo Starting training...
echo.

python train_minimal_working.py

echo.
echo ========================================================================
echo TRAINING COMPLETE!
echo ========================================================================
echo.
echo Checkpoints saved to: checkpoints\minimal\
echo.
pause
