@echo off
REM Download Qwen 3.5 0.8B for BDH SOTA Training
echo ========================================================================
echo DOWNLOADING QWEN 3.5 0.8B FOR BDH SOTA DISTILLATION
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
echo Model size: ~1.5GB (fp32) or ~800MB (fp16)
echo This may take a few minutes on first download...
echo.

python download_qwen3_0.8B.py

echo.
echo ========================================================================
echo DOWNLOAD COMPLETE!
echo ========================================================================
echo.
echo Next step: Run SOTA training script
pause
