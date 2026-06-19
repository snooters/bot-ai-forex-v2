# Start bot in background, save PID
$log = Join-Path $PSScriptRoot "bot.log"
$pidFile = Join-Path $PSScriptRoot "bot.pid"

$proc = Start-Process -FilePath "python" -ArgumentList "main.py", "live" `
    -WorkingDirectory $PSScriptRoot `
    -WindowStyle Hidden -PassThru

$proc.Id | Out-File -FilePath $pidFile -Encoding ascii
Write-Host "Bot started (PID: $($proc.Id))"
