@echo off
echo =================================
echo HYDRA SETUP & LAUNCHER
echo =================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b
)

echo [1/3] Setting up Python environment...
python setup_distillation_env.py

echo [2/3] Checking for dataset...
if not exist "data	inystories.txt" (
    echo ERROR: tinystories.txt not found in data folder.
    pause
    exit /b
)

echo [3/3] Starting Logit Generation for Node 2...
echo This will take several hours. You can close this window when it finishes.

python generate_logits.py --node 2

echo.
echo =================================
echo NODE 2 FINISHED!
echo =================================
pause
