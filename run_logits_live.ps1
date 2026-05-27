param(
    [string]$PythonExe = "D:\.conda\envs\bdh-fest\python.exe",
    [string]$Data = "data\chat_reasoning_new_v2_45k.jsonl",
    [string]$TeacherModel = "D:\projects\BDH\Qwen3.5-4B",
    [string]$OutputRoot = "data\teacher_logits_new",
    [int]$SeqLen = 128,
    [int]$TopK = 128,
    [int]$BatchSize = 5,
    [int]$NumWorkers = 2,
    [int]$ChunkRows = 1024,
    [double]$RunHours = 4,
    [int]$ProgressEvery = 20,
    [double]$HeartbeatEverySec = 10,
    [int]$MonitorIntervalSec = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python not found: $PythonExe"
}
if (-not (Test-Path ".\generate_logits_new_data.py")) {
    throw "Missing script: .\generate_logits_new_data.py"
}
if (-not (Test-Path ".\monitor_logits_live.ps1")) {
    throw "Missing script: .\monitor_logits_live.ps1"
}

$env:DISABLE_ADDMM_CUDA_LT = "1"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $env:TEMP ("logits-live-" + $ts + ".log")

$args = @(
    "generate_logits_new_data.py",
    "--data", $Data,
    "--teacher-model", $TeacherModel,
    "--output-root", $OutputRoot,
    "--seq-len", "$SeqLen",
    "--top-k", "$TopK",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--chunk-rows", "$ChunkRows",
    "--run-hours", "$RunHours",
    "--allow-reuse",
    "--gpu-dtype", "fp16",
    "--progress-every", "$ProgressEvery",
    "--heartbeat-every-sec", "$HeartbeatEverySec"
)

Write-Host "Starting logits generation..."
Write-Host "Log file: $logFile"
$proc = Start-Process -FilePath $PythonExe -ArgumentList $args -RedirectStandardOutput $logFile -PassThru

$runDir = ""
for ($i = 0; $i -lt 60; $i++) {
    if (Test-Path $logFile) {
        $line = Select-String -Path $logFile -Pattern "^Run dir:\s*(.+)$" | Select-Object -Last 1
        if ($line) {
            $runDir = $line.Matches[0].Groups[1].Value.Trim()
            break
        }
    }
    Start-Sleep -Seconds 1
}

if ([string]::IsNullOrWhiteSpace($runDir)) {
    $root = Join-Path (Get-Location) "data\teacher_logits_new"
    if (Test-Path $root) {
        $latest = Get-ChildItem -Path $root -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) { $runDir = $latest.FullName }
    }
}

Write-Host "PID: $($proc.Id)"
Write-Host "RunDir: $runDir"

powershell -NoProfile -ExecutionPolicy Bypass -File .\monitor_logits_live.ps1 `
    -RunDir $runDir `
    -LogFile $logFile `
    -IntervalSec $MonitorIntervalSec `
    -ProcessId $proc.Id `
    -StopWhenDone

Write-Host ""
Write-Host "Generation process finished."
Write-Host "RunDir: $runDir"
Write-Host "Log:    $logFile"
