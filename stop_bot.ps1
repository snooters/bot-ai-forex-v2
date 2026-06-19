# Stop bot by saved PID (safe, no "Access denied")
$pidFile = Join-Path $PSScriptRoot "bot.pid"

if (!(Test-Path $pidFile)) {
    Write-Host "No bot.pid found. Bot may not be running."
    exit
}

$pid = Get-Content $pidFile -Raw -ErrorAction SilentlyContinue | ForEach-Object { $_.Trim() }

if (!($pid -match '^\d+$')) {
    Write-Host "Invalid PID in bot.pid"
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    exit
}

# Check if process exists
$proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
if (!$proc) {
    Write-Host "Bot (PID $pid) not running. Cleaning up."
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    exit
}

# Only kill this specific PID
Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Verify
$still = Get-Process -Id $pid -ErrorAction SilentlyContinue
if ($still) {
    Write-Host "Failed to stop bot (PID $pid)"
} else {
    Write-Host "Bot stopped (PID $pid)"
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}
