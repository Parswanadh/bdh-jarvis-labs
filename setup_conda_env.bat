@echo off
REM BDH Science Fest - PyTorch Installation Script
REM This script creates a conda environment and installs PyTorch for BDH development

echo ========================================
echo BDH Science Fest - Environment Setup
echo ========================================
echo.

echo Step 1: Creating conda environment 'bdh-fest'...
call conda create -n bdh-fest python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)
echo.
echo ✅ Environment 'bdh-fest' created successfully!
echo.

echo Step 2: Activating environment...
call conda activate bdh-fest
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate environment
    pause
    exit /b 1
)
echo.
echo ✅ Environment activated!
echo.

echo Step 3: Installing PyTorch with CUDA 12.4 support...
echo (This may take a few minutes...)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)
echo.
echo ✅ PyTorch installed successfully!
echo.

echo Step 4: Installing additional dependencies...
pip install tokenizers matplotlib seaborn plotly numpy jupyter
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.
echo ✅ All dependencies installed!
echo.

echo Step 5: Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Could not verify PyTorch installation
)
echo.

echo ========================================
echo ✅ SETUP COMPLETE!
echo ========================================
echo.
echo To activate the environment, run:
echo   conda activate bdh-fest
echo.
echo PyTorch is ready for BDH development!
echo.
pause
