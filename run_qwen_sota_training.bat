@echo off
REM SOTA BDH Training with Qwen 3.5 0.8B Teacher
REM 2-Hour Session with Resume Capability
echo ================================================================================
echo SOTA BDH TRAINING - QWEN 3.5 0.8B DISTILLATION
echo ================================================================================
echo.
echo Configuration:
echo   Teacher: Qwen 3.5 0.8B (800M params)
echo   Student: BDH 41M params
echo   Session: 2 hours (checkpoint every 30 min)
echo   Temperature: 3.0 (softer teaching)
echo   Distillation: 70%% KL + 30%% CE
echo   Target Loss: 1.40-1.60 (after 13-16h total)
echo.
echo ================================================================================
echo.

echo Activating bdh-fest environment...
call conda activate bdh-fest

if errorlevel 1 (
    echo ERROR: Failed to activate bdh-fest environment
    pause
    exit /b 1
)

echo.
echo [1/2] Testing Qwen 3.5 0.8B model...
python test_qwen3.5_model.py

if errorlevel 1 (
    echo ERROR: Qwen model test failed
    pause
    exit /b 1
)

echo.
echo [2/2] Starting SOTA training...
echo.
python train_qwen_sota_2hr.py

echo.
echo ================================================================================
echo TRAINING SESSION COMPLETE!
echo ================================================================================
echo.
echo Checkpoints saved to: checkpoints/qwen35_sota/
echo.
echo To resume training, run this script again.
echo.
pause
