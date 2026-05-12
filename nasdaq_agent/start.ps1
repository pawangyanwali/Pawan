# NASDAQ Agent — background process manager for Windows
# Usage (open PowerShell in the nasdaq_agent folder):
#   .\start.ps1            - start agent in background
#   .\start.ps1 stop       - stop agent
#   .\start.ps1 status     - check if running
#   .\start.ps1 logs       - tail live log output
#   .\start.ps1 install    - register Task Scheduler task (auto-start on login)
#   .\start.ps1 uninstall  - remove the scheduled task

param([string]$Action = "start")

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir    = Join-Path $ScriptDir "logs"
$LogFile   = Join-Path $LogDir "agent.log"
$PidFile   = Join-Path $LogDir "agent.pid"
$TaskName  = "NasdaqScalpingAgent"
$EnvFile   = Join-Path $ScriptDir ".env"

# Load .env file into current process so API key is passed to uvicorn
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | Where-Object { $_ -match "^\s*[^#]" } | ForEach-Object {
        $parts = $_ -split "=", 2
        if ($parts.Count -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
        }
    }
}

function Find-Python {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }
    $py3 = Get-Command python3 -ErrorAction SilentlyContinue
    if ($py3) { return $py3.Source }
    Write-Host "ERROR: python not found in PATH." -ForegroundColor Red
    exit 1
}

function Find-Uvicorn {
    # Prefer the standalone uvicorn.exe if it is on PATH
    $uv = Get-Command uvicorn -ErrorAction SilentlyContinue
    if ($uv) { return $uv.Source }

    # Always-works fallback: run uvicorn as a Python module.
    # Wrap it as a tiny launcher script so Start-Process can use it.
    $python = Find-Python
    $launcher = Join-Path $LogDir "run_agent.bat"
    @"
@echo off
cd /d "$ScriptDir"
"$python" -m uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log
"@ | Set-Content $launcher
    return $launcher
}

function Get-RunningPid {
    if (Test-Path $PidFile) {
        $stored = [int](Get-Content $PidFile -ErrorAction SilentlyContinue)
        if ($stored -and (Get-Process -Id $stored -ErrorAction SilentlyContinue)) {
            return $stored
        }
    }
    $proc = Get-WmiObject Win32_Process |
            Where-Object { $_.CommandLine -like "*uvicorn*main:app*" } |
            Select-Object -First 1
    if ($proc) { return $proc.ProcessId }
    return $null
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$cmd = $Action.ToLower()

if ($cmd -eq "start") {
    $running = Get-RunningPid
    if ($running) {
        Write-Host "Already running (PID $running)" -ForegroundColor Green
        exit 0
    }
    $python    = Find-Python
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content $LogFile "[$timestamp] === Agent starting ==="
    # cmd /c merges stderr into stdout (2>&1) so both go to the same log file
    $proc = Start-Process `
        -FilePath "cmd.exe" `
        -ArgumentList "/c `"$python`" -m uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log >> `"$LogFile`" 2>&1" `
        -WorkingDirectory $ScriptDir `
        -WindowStyle Hidden `
        -PassThru
    $proc.Id | Set-Content $PidFile
    Write-Host "Started  PID $($proc.Id)" -ForegroundColor Green
    Write-Host "Logs:    $LogFile"
    Write-Host "Stop:    .\start.ps1 stop"
    Write-Host "Dashboard: http://localhost:8000"

} elseif ($cmd -eq "stop") {
    $running = Get-RunningPid
    if ($running) {
        Stop-Process -Id $running -Force
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        Write-Host "Stopped (PID $running)" -ForegroundColor Yellow
    } else {
        Write-Host "Not running." -ForegroundColor Gray
    }

} elseif ($cmd -eq "status") {
    $running = Get-RunningPid
    if ($running) {
        Write-Host "Running OK  (PID $running)  http://localhost:8000" -ForegroundColor Green
    } else {
        Write-Host "Stopped" -ForegroundColor Red
    }

} elseif ($cmd -eq "logs") {
    if (Test-Path $LogFile) {
        Get-Content $LogFile -Wait -Tail 40
    } else {
        Write-Host "No log file yet. Start the agent first." -ForegroundColor Gray
    }

} elseif ($cmd -eq "install") {
    # Registers a Task Scheduler task that starts the agent at every login.
    # Run this once from an elevated (Administrator) PowerShell prompt.
    $python  = Find-Python
    $ta      = New-ScheduledTaskAction `
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

} elseif ($cmd -eq "uninstall") {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Task '$TaskName' removed." -ForegroundColor Yellow

} else {
    Write-Host "Usage: .\start.ps1  start | stop | status | logs | install | uninstall"
}
