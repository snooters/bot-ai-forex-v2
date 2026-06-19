# Stop bot by port 9090 (most reliable)
$port = 9090
$conn = netstat -ano | Select-String ":$port\s+.*LISTENING"
if (!$conn) {
    Write-Host "No bot found on port $port"
    exit
}

$pid = ($conn -replace '.*\s+(\d+)$', '$1').Trim()
if (!($pid -match '^\d+$')) {
    Write-Host "Could not parse PID from netstat"
    exit
}

$proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
if (!$proc) {
    Write-Host "Process (PID $pid) not found"
    exit
}

Write-Host "Stopping bot (PID $pid, Name: $($proc.ProcessName))..."
Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

$still = Get-Process -Id $pid -ErrorAction SilentlyContinue
if ($still) {
    Write-Host "Failed to stop bot (PID $pid)"
} else {
    Write-Host "Bot stopped (PID $pid)"
}
