# PowerShell script to download Gemma 3 270M
Write-Host "=" * 70
Write-Host "DOWNLOADING GEMMA 3 270M FOR DISTILLATION"
Write-Host "=" * 70
Write-Host ""

# Initialize conda
Write-Host "[INIT] Initializing conda..."
& conda init powershell | Out-Null

# Activate bdh environment
Write-Host "[ENV] Activating bdh environment..."
conda activate bdh

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate bdh environment" -ForegroundColor Red
    Write-Host "Please install conda and create bdh environment first"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[OK] Environment activated"
Write-Host ""

# Run download script
Write-Host "[DOWNLOAD] Running download script..."
Write-Host ""
python download_gemma3_270m.py

Write-Host ""
Write-Host "=" * 70
Write-Host "DOWNLOAD COMPLETE!"
Write-Host "=" * 70
Write-Host ""
Read-Host "Press Enter to exit"
