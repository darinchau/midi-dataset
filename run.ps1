# A little script to run a python module in a loop, restarting it if it segfaultss or crashes.
# Useful for long training runs that might be interrupted
# Path to your venv
$VENV_PATH = ".venv"
$PYTHON = "python"
if ($args.Length -eq 0) {
    Write-Host "Usage: package.ps1 <module_name>"
    exit 1
}
$BASE_MODULE = $args[0]
$ARGS_LIST = $args[1..($args.Length - 1)]

# Gracefully handle Ctrl+C
$stopLoop = $false
$null = Register-EngineEvent ConsoleCancel -Action {
    Write-Host "Stopping driver..."
    $global:stopLoop = $true
}

# Activate virtual environment
$activatePath1 = Join-Path $VENV_PATH "Scripts\Activate.ps1"
$activatePath2 = Join-Path $VENV_PATH "bin\Activate.ps1"

if (Test-Path $activatePath1) {
    . $activatePath1
} elseif (Test-Path $activatePath2) {
    . $activatePath2
} else {
    Write-Host "Could not find virtual environment activation script."
    exit 1
}

while (-not $stopLoop) {
    $argsList = @("-m", $BASE_MODULE) + $ARGS_LIST

    Write-Host "Running: $PYTHON $($argsList -join ' ')"

    & $PYTHON @argsList
    $EXIT_CODE = $LASTEXITCODE

    Write-Host "Training process exited with code $EXIT_CODE."
    if ($stopLoop) { break }

    Write-Host "Restarting in 5 seconds... (Press Ctrl+C to stop)"
    Start-Sleep -Seconds 5
}