@echo off
REM BDH Science Fest - PyTorch Quick Setup
REM Double-click this file to run

echo ========================================
echo BDH Science Fest - PyTorch Setup
echo ========================================
echo.

echo [1/4] Creating conda environment...
conda create -n bdh-fest python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create environment
    pause
    exit /b 1
)
echo.
echo ✅ Environment created!
echo.

echo [2/4] Installing PyTorch (this takes 5-10 minutes)...
conda run -n bdh-fest pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)
echo.
echo ✅ PyTorch installed!
echo.

echo [3/4] Installing dependencies...
conda run -n bdh-fest pip install tokenizers matplotlib seaborn plotly numpy jupyter
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some dependencies may have failed
)
echo.
echo ✅ Dependencies installed!
echo.

echo [4/4] Verifying installation...
conda run -n bdh-fest python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo.

echo ========================================
echo ✅ SETUP COMPLETE!
echo ========================================
echo.
echo To activate environment, run:
echo   conda activate bdh-fest
echo.
echo Then run:
echo   python build_and_test.py
echo.
pause
