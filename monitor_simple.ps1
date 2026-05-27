#!/usr/bin/env pwsh
# Simple one-liner monitor for current logits generation run on D: drive.
# Run this to see live progress without any setup.

param(
    [string]$Root = "D:\logits_production_extended",
    [int]$IntervalSec = 5
)

while ($true) {
    Clear-Host
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "=== LOGITS GENERATION MONITOR [$ts] ===" -ForegroundColor Cyan
    
    # Find latest run
    $runDir = (Get-ChildItem -Path "$Root\run_*" -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    if (-not $runDir) {
        Write-Host "No run found in $Root"
        Start-Sleep -Seconds $IntervalSec
        continue
    }
    
    Write-Host "Run: $(Split-Path $runDir -Leaf)"
    
    # Check process
    $proc = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -match "generate_logits"
    } | Select-Object -First 1
    if ($proc) {
        $p = Get-Process -Id $proc.ProcessId -ErrorAction SilentlyContinue
        if ($p) {
            Write-Host "Process: RUNNING (PID=$($p.Id), CPU=$([math]::Round($p.CPU, 1))s, RAM=$([math]::Round($p.WorkingSet64/1MB, 0))MB)"
        } else {
            Write-Host "Process: RUNNING (PID=$($proc.ProcessId))"
        }
    } else {
        Write-Host "Process: NOT RUNNING" -ForegroundColor Red
    }
    
    # GPU stats
    try {
        $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ($gpu) {
            $parts = $gpu -split ",\s*"
            Write-Host "GPU: util=$($parts[0])% mem=$($parts[1])/$($parts[2])MB temp=$($parts[3])C power=$($parts[4])W"
        }
    } catch {}
    
    # Logits stats
    if (Test-Path $runDir) {
        $chunks = @(Get-ChildItem -Path "$runDir\logits_chunk_*.npz" -File -ErrorAction SilentlyContinue).Count
        $heartbeat = "$runDir\heartbeat.json"
        
        if (Test-Path $heartbeat) {
            $hb = Get-Content $heartbeat -Raw | ConvertFrom-Json
            Write-Host "Progress: $($hb.rows_done)/$($hb.rows_total) rows | $([math]::Round($hb.rows_per_sec, 2)) rows/sec | ETA $([math]::Round($hb.eta_sec / 60, 1))m | $chunks chunks"
        }
    }
    
    Write-Host ""
    Start-Sleep -Seconds $IntervalSec
}
