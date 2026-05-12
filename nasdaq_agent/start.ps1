# NASDAQ Agent — background process manager for Windows
# Usage (run in PowerShell as Administrator for install-task):
#   .\start.ps1            — start agent in background
#   .\start.ps1 stop       — stop agent
#   .\start.ps1 status     — check if running
#   .\start.ps1 logs       — tail live log output
#   .\start.ps1 install    — register Windows Task Scheduler task (auto-start on boot)
#   .\start.ps1 uninstall  — remove the scheduled task

param([string]$Action = "start")

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir     = Join-Path $ScriptDir "logs"
$LogFile    = Join-Path $LogDir "agent.log"
$PidFile    = Join-Path $LogDir "agent.pid"
$TaskName   = "NasdaqScalpingAgent"
$EnvFile    = Join-Path $ScriptDir ".env"

# Load .env into current process environment (keeps API key out of the script)
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | Where-Object { $_ -match "^\s*[^#]" } | ForEach-Object {
        $parts = $_ -split "=", 2
        if ($parts.Count -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
        }
    }
}

function Get-Uvicorn {
    $uv = Get-Command uvicorn -ErrorAction SilentlyContinue
    if ($uv) { return $uv.Source }
    # Fallback: look in common Python Scripts dirs
    foreach ($p in @("$env:LOCALAPPDATA\Programs\Python\Python3*\Scripts\uvicorn.exe",
                     "$env:APPDATA\Python\Python3*\Scripts\uvicorn.exe",
                     "C:\Python3*\Scripts\uvicorn.exe")) {
        $found = Resolve-Path $p -ErrorAction SilentlyContinue | Select-Object -Last 1
        if ($found) { return $found.Path }
    }
    throw "uvicorn not found. Run: pip install uvicorn"
}

function Get-RunningPid {
    if (Test-Path $PidFile) {
        $pid = [int](Get-Content $PidFile -ErrorAction SilentlyContinue)
        if ($pid -and (Get-Process -Id $pid -ErrorAction SilentlyContinue)) { return $pid }
    }
    # Fallback: find by command line
    $proc = Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*uvicorn*main:app*" } | Select-Object -First 1
    if ($proc) { return $proc.ProcessId }
    return $null
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

switch ($Action.ToLower()) {

    "start" {
        $running = Get-RunningPid
        if ($running) {
            Write-Host "Already running (PID $running)" -ForegroundColor Green
            exit 0
        }
        $uvicorn = Get-Uvicorn
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content $LogFile "[$timestamp] === Agent starting ==="
        $proc = Start-Process -FilePath $uvicorn `
            -ArgumentList "main:app --host 0.0.0.0 --port 8000 --no-access-log" `
            -WorkingDirectory $ScriptDir `
            -WindowStyle Hidden `
            -RedirectStandardOutput $LogFile `
            -RedirectStandardError  $LogFile `
            -PassThru
        $proc.Id | Set-Content $PidFile
        Write-Host "Started — PID $($proc.Id)" -ForegroundColor Green
        Write-Host "Logs:   $LogFile"
        Write-Host "Stop:   .\start.ps1 stop"
        Write-Host "Dashboard: http://localhost:8000"
    }

    "stop" {
        $running = Get-RunningPid
        if ($running) {
            Stop-Process -Id $running -Force
            Remove-Item $PidFile -ErrorAction SilentlyContinue
            Write-Host "Stopped (PID $running)" -ForegroundColor Yellow
        } else {
            Write-Host "Not running." -ForegroundColor Gray
        }
    }

    "status" {
        $running = Get-RunningPid
        if ($running) {
            Write-Host "Running ✓  (PID $running)  http://localhost:8000" -ForegroundColor Green
        } else {
            Write-Host "Stopped ✗" -ForegroundColor Red
        }
    }

    "logs" {
        if (Test-Path $LogFile) {
            Get-Content $LogFile -Wait -Tail 40
        } else {
            Write-Host "No log file yet. Start the agent first." -ForegroundColor Gray
        }
    }

    "install" {
        # Registers a Task Scheduler task that starts the agent at every logon
        # and restarts it if it crashes.  Run as Administrator.
        $uvicorn   = Get-Uvicorn
        $action    = New-ScheduledTaskAction `
            -Execute $uvicorn `
            -Argument "main:app --host 0.0.0.0 --port 8000 --no-access-log" `
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
        Register-ScheduledTask -TaskName $TaskName -Action $action `
            -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
        Write-Host "Task '$TaskName' registered — agent will start automatically at logon." -ForegroundColor Green
        Write-Host "Start now:   Start-ScheduledTask -TaskName '$TaskName'"
        Write-Host "Uninstall:   .\start.ps1 uninstall"
    }

    "uninstall" {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Task '$TaskName' removed." -ForegroundColor Yellow
    }

    default {
        Write-Host "Usage: .\start.ps1 {start|stop|status|logs|install|uninstall}"
    }
}
