[CmdletBinding()]
param(
    [ValidateSet("latest", "safe8", "ultimate", "stable")]
    [string]$Profile = "latest",
    [int]$Port = 8000,
    [switch]$EagerLoad
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$profiles = @{
    latest = @{
        Name = "Latest checkpoint (safe_11L)"
        Checkpoint = "checkpoints/safe/laptop_safe_11L.pt"
        TeacherModel = "Qwen3.5-0.8B"
        NEmbd = 256
        NLayer = 11
        NHead = 8
        FfnDim = 1024
        MaxSeq = 192
        HebbianLr = "0.0006"
    }
    safe8 = @{
        Name = "Safe 8-layer baseline"
        Checkpoint = "checkpoints/safe/laptop_safe.pt"
        TeacherModel = "Qwen3.5-0.8B"
        NEmbd = 256
        NLayer = 8
        NHead = 8
        FfnDim = 1024
        MaxSeq = 192
        HebbianLr = "0.0006"
    }
    ultimate = @{
        Name = "Ultimate 12-layer checkpoint"
        Checkpoint = "checkpoints/ultimate/hatchling_latest.pt"
        TeacherModel = "Qwen3.5-0.8B"
        NEmbd = 256
        NLayer = 12
        NHead = 8
        FfnDim = 1024
        MaxSeq = 224
        HebbianLr = "0.0006"
    }
    stable = @{
        Name = "Stable BBPE checkpoint"
        Checkpoint = "checkpoints/1hour_training/final_model.pt"
        TeacherModel = "Qwen3.5-0.8B"
        NEmbd = 512
        NLayer = 8
        NHead = 8
        FfnDim = 2048
        MaxSeq = 512
        HebbianLr = "0.0006"
    }
}

$selected = $profiles[$Profile]
$selectedPath = Join-Path $repoRoot $selected.Checkpoint

if (-not (Test-Path $selectedPath)) {
    if ($Profile -eq "latest") {
        Write-Warning "Latest checkpoint not found. Falling back to stable profile."
        $selected = $profiles["stable"]
        $selectedPath = Join-Path $repoRoot $selected.Checkpoint
    }
}

if (-not (Test-Path $selectedPath)) {
    throw "Checkpoint file not found: $selectedPath"
}

Write-Host "Launching BDH professor demo" -ForegroundColor Cyan
Write-Host "Profile: $($selected.Name)" -ForegroundColor Cyan
Write-Host "Checkpoint: $selectedPath" -ForegroundColor Cyan
Write-Host "Open UI at: http://127.0.0.1:$Port" -ForegroundColor Cyan

Push-Location $repoRoot
try {
    $args = @(
        "webapp/server.py",
        "--host", "127.0.0.1",
        "--port", $Port.ToString(),
        "--checkpoint", $selected.Checkpoint,
        "--data", "data/tinystories.txt",
        "--teacher-model", $selected.TeacherModel,
        "--distil-model", "distilbert/distilgpt2",
        "--bdh-n-embd", $selected.NEmbd.ToString(),
        "--bdh-n-layer", $selected.NLayer.ToString(),
        "--bdh-n-head", $selected.NHead.ToString(),
        "--bdh-ffn-dim", $selected.FfnDim.ToString(),
        "--bdh-max-seq-len", $selected.MaxSeq.ToString(),
        "--bdh-hebbian-lr", $selected.HebbianLr,
        "--default-top-k", "40"
    )

    if ($EagerLoad) {
        $args += "--eager-load"
    }

    python @args
}
finally {
    Pop-Location
}
