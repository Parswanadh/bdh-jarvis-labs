param(
    [string]$RunDir = "",
    [string]$LogFile = "",
    [int]$IntervalSec = 5,
    [int]$ProcessId = 0,
    [switch]$StopWhenDone
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "SilentlyContinue"

function Get-LatestRunDir {
    $root = Join-Path (Get-Location) "data\teacher_logits_new"
    if (-not (Test-Path $root)) { return "" }
    $dirs = Get-ChildItem -Path $root -Directory | Sort-Object LastWriteTime -Descending
    if ($dirs.Count -eq 0) { return "" }
    return $dirs[0].FullName
}

function Get-LatestDetachedLog {
    $tmp = [System.IO.Path]::GetTempPath()
    $logs = Get-ChildItem -Path $tmp -Filter "copilot-detached-*.log" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($logs.Count -eq 0) { return "" }
    return $logs[0].FullName
}

function Get-GpuLine {
    $line = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if (-not $line) { return "GPU: nvidia-smi unavailable" }
    $parts = $line -split ",\s*"
    if ($parts.Count -lt 5) { return "GPU: $line" }
    return ("GPU util={0}% mem={1}/{2} MiB temp={3}C power={4}W" -f $parts[0], $parts[1], $parts[2], $parts[3], $parts[4])
}

if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $RunDir = Get-LatestRunDir
}

if ([string]::IsNullOrWhiteSpace($LogFile)) {
    $LogFile = Get-LatestDetachedLog
}

if ([string]::IsNullOrWhiteSpace($RunDir)) {
    Write-Host "No run dir found under data\teacher_logits_new"
    exit 1
}

while ($true) {
    Clear-Host
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "=== LOGITS LIVE MONITOR [$ts] ==="
    Write-Host "RunDir: $RunDir"

    $proc = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and (
            ($ProcessId -gt 0 -and $_.ProcessId -eq $ProcessId) -or
            ($ProcessId -le 0 -and $_.CommandLine -match "generate_logits_new_data.py")
        )
    } | Select-Object -First 1

    if ($proc) {
        $p = Get-Process -Id $proc.ProcessId
        $rssGb = [math]::Round($p.WorkingSet64 / 1GB, 2)
        Write-Host ("Process: RUNNING pid={0} cpu={1}s rss={2}GB" -f $p.Id, [math]::Round($p.CPU, 2), $rssGb)
    } else {
        Write-Host "Process: NOT RUNNING"
        if ($StopWhenDone) {
            Write-Host "Process completed. Exiting monitor."
            break
        }
    }

    Write-Host (Get-GpuLine)

    if (Test-Path $RunDir) {
        $chunks = Get-ChildItem -Path $RunDir -Filter "logits_chunk_*.npz" -File
        $chunkCount = @($chunks).Count
        $chunkGb = [math]::Round((($chunks | Measure-Object Length -Sum).Sum / 1GB), 3)
        Write-Host ("Chunks: {0}  Size: {1} GB" -f $chunkCount, $chunkGb)

        $metaPath = Join-Path $RunDir "metadata.json"
        $hbPath = Join-Path $RunDir "heartbeat.json"
        if (Test-Path $metaPath) {
            $meta = Get-Content -Path $metaPath -Raw | ConvertFrom-Json
            Write-Host ("Rows generated: {0} / {1}  Seconds: {2}" -f $meta.rows_generated, $meta.rows_total_loaded, $meta.seconds)
        } else {
            Write-Host "Rows generated: metadata.json not yet written (run still active)."
        }
        if (Test-Path $hbPath) {
            $hb = Get-Content -Path $hbPath -Raw | ConvertFrom-Json
            Write-Host ("Heartbeat: status={0} rows={1}/{2} rps={3} eta_s={4}" -f $hb.status, $hb.rows_done, $hb.rows_total, $hb.rows_per_sec, $hb.eta_sec)
        } else {
            Write-Host "Heartbeat: heartbeat.json not found yet."
        }
    } else {
        Write-Host "RunDir missing."
    }

    if (-not [string]::IsNullOrWhiteSpace($LogFile) -and (Test-Path $LogFile)) {
        Write-Host ""
        Write-Host "--- Log tail ---"
        Get-Content -Path $LogFile -Tail 12
    }

    Start-Sleep -Seconds ([Math]::Max(1, $IntervalSec))
}
