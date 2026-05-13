# NASDAQ Agent — background process manager for Windows
# Usage (open PowerShell in the nasdaq_agent folder):
#   .\start.ps1            - start agent in background
#   .\start.ps1 stop       - stop agent
#   .\start.ps1 status     - check if running
#   .\start.ps1 logs       - tail live log output
#   .\start.ps1 install    - register Task Scheduler task (auto-start on login)
#   .\start.ps1 uninstall  - remove the scheduled task

param([string]$Action = "start")

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir      = Join-Path $ScriptDir "logs"
$StdoutLog   = Join-Path $LogDir "agent.log"
$StderrLog   = Join-Path $LogDir "agent.err"
$PidFile     = Join-Path $LogDir "agent.pid"
$TaskName    = "NasdaqScalpingAgent"
$EnvFile     = Join-Path $ScriptDir ".env"

# Load .env into the current process — child processes inherit these vars
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | Where-Object { $_ -match "^\s*[^#]" } | ForEach-Object {
        $parts = $_ -split "=", 2
        if ($parts.Count -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
        }
    }
}

function Find-Python {
    foreach ($name in @("python", "python3")) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if ($found) { return $found.Source }
    }
    Write-Host "ERROR: python not found in PATH. Install Python and retry." -ForegroundColor Red
    exit 1
}

function Get-RunningPid {
    if (Test-Path $PidFile) {
        $stored = [int](Get-Content $PidFile -ErrorAction SilentlyContinue)
        if ($stored -and (Get-Process -Id $stored -ErrorAction SilentlyContinue)) {
            return $stored
        }
    }
    # Fallback: search running processes for our uvicorn command
    $match = Get-WmiObject Win32_Process |
             Where-Object { $_.CommandLine -like "*uvicorn*main:app*" } |
             Select-Object -First 1
    if ($match) { return $match.ProcessId }
    return $null
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$cmd = $Action.ToLower()

# ── start ──────────────────────────────────────────────────────────────────────
if ($cmd -eq "start") {
    $running = Get-RunningPid
    if ($running) {
        Write-Host "Already running (PID $running)" -ForegroundColor Green
        exit 0
    }

    $python    = Find-Python
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content $StdoutLog "[$timestamp] === Agent starting ==="

    # Use separate stdout/stderr files — PowerShell requires distinct paths.
    # 'logs' command merges them for display.
    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList "-m uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log" `
        -WorkingDirectory $ScriptDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError  $StderrLog `
        -PassThru

    Start-Sleep -Milliseconds 1500   # give it a moment to either bind or crash
    if ($proc.HasExited) {
        Write-Host "ERROR: Agent exited immediately (code $($proc.ExitCode))." -ForegroundColor Red
        Write-Host "--- stdout ---"
        if (Test-Path $StdoutLog) { Get-Content $StdoutLog | Select-Object -Last 20 }
        Write-Host "--- stderr ---"
        if (Test-Path $StderrLog) { Get-Content $StderrLog | Select-Object -Last 20 }
        exit 1
    }

    $proc.Id | Set-Content $PidFile
    Write-Host "Started  PID $($proc.Id)" -ForegroundColor Green
    Write-Host "Logs:    $StdoutLog  (errors: $StderrLog)"
    Write-Host "Stop:    .\start.ps1 stop"
    Write-Host "Dashboard: http://localhost:8000"

# ── stop ───────────────────────────────────────────────────────────────────────
} elseif ($cmd -eq "stop") {
    $running = Get-RunningPid
    if ($running) {
        Stop-Process -Id $running -Force
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        Write-Host "Stopped (PID $running)" -ForegroundColor Yellow
    } else {
        Write-Host "Not running." -ForegroundColor Gray
    }

# ── status ─────────────────────────────────────────────────────────────────────
} elseif ($cmd -eq "status") {
    $running = Get-RunningPid
    if ($running) {
        Write-Host "Running OK  (PID $running)  http://localhost:8000" -ForegroundColor Green
    } else {
        Write-Host "Stopped" -ForegroundColor Red
        if (Test-Path $StderrLog) {
            $errs = Get-Content $StderrLog -ErrorAction SilentlyContinue
            if ($errs) {
                Write-Host "Last errors:" -ForegroundColor Yellow
                $errs | Select-Object -Last 10 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkYellow }
            }
        }
    }

# ── logs ───────────────────────────────────────────────────────────────────────
} elseif ($cmd -eq "logs") {
    if (-not (Test-Path $StdoutLog)) {
        Write-Host "No log file yet. Start the agent first." -ForegroundColor Gray
        exit 0
    }
    Write-Host "(Ctrl+C to stop tailing)" -ForegroundColor DarkGray
    # Interleave stdout + stderr by watching both files
    $jobs = @()
    $jobs += Start-Job { Get-Content $using:StdoutLog -Wait -Tail 30 }
    if (Test-Path $StderrLog) {
        $jobs += Start-Job { Get-Content $using:StderrLog -Wait -Tail 5 }
    }
    try {
        while ($true) {
            $jobs | Receive-Job
            Start-Sleep -Milliseconds 500
        }
    } finally {
        $jobs | Stop-Job
        $jobs | Remove-Job
    }

# ── install ────────────────────────────────────────────────────────────────────
} elseif ($cmd -eq "install") {
    $python    = Find-Python
    $ta        = New-ScheduledTaskAction `
                    -Execute $python `
                    -Argument "-m uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log" `
                    -WorkingDirectory $ScriptDir
    $trigger   = New-ScheduledTaskTrigger -AtLogOn
    $settings  = New-ScheduledTaskSettingsSet `
                    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
                    -RestartCount 99 `
                    -RestartInterval (New-TimeSpan -Minutes 1) `
                    -StartWhenAvailable
    $principal = New-ScheduledTaskPrincipal `
                    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
                    -LogonType Interactive `
                    -RunLevel Highest
    Register-ScheduledTask -TaskName $TaskName -Action $ta `
        -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    Write-Host "Task '$TaskName' registered. Agent will start automatically at next login." -ForegroundColor Green
    Write-Host "Start it now:  Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "Remove it:     .\start.ps1 uninstall"

# ── uninstall ──────────────────────────────────────────────────────────────────
} elseif ($cmd -eq "uninstall") {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Task '$TaskName' removed." -ForegroundColor Yellow

} else {
    Write-Host "Usage: .\start.ps1  start | stop | status | logs | install | uninstall"
}
