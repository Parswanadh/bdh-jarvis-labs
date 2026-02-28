# BDH Science Fest - PyTorch Setup Script
# Run this in PowerShell as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BDH Science Fest - PyTorch Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create conda environment
Write-Host "[Step 1/5] Creating conda environment 'bdh-fest'..." -ForegroundColor Yellow
conda create -n bdh-fest python=3.10 -y
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Environment created successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to create environment" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Install PyTorch
Write-Host "[Step 2/5] Installing PyTorch with CUDA 12.4..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Cyan
conda run -n bdh-fest pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ PyTorch installed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install PyTorch" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Install dependencies
Write-Host "[Step 3/5] Installing additional dependencies..." -ForegroundColor Yellow
conda run -n bdh-fest pip install tokenizers matplotlib seaborn plotly numpy jupyter
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 4: Verify installation
Write-Host "[Step 4/5] Verifying PyTorch installation..." -ForegroundColor Yellow
conda run -n bdh-fest python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
Write-Host ""

# Step 5: Complete
Write-Host "========================================" -ForegroundColor Green
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Cyan
Write-Host "  conda activate bdh-fest" -ForegroundColor White
Write-Host ""
Write-Host "Then run the build script:" -ForegroundColor Cyan
Write-Host "  python build_and_test.py" -ForegroundColor White
Write-Host ""
