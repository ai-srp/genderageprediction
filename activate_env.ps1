# PowerShell script to activate the virtual environment and return to base folder
# This needs to be 'sourced' to affect the current session
# Usage: . .\activate_env.ps1

$scriptPath = $PSScriptRoot
if (-not $scriptPath) {
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$envPath = Join-Path -Path $scriptPath -ChildPath "age_gender_env"
$activateScript = Join-Path -Path $envPath -ChildPath "Scripts\Activate.ps1"

Write-Host "Activating virtual environment from: $envPath"
# Source the activation script to affect the current session
. $activateScript

# Return to the base application folder
Set-Location $scriptPath
Write-Host "Returned to base application folder: $scriptPath"
Write-Host "Virtual environment is now active. You can run your Python scripts."
Write-Host "NOTE: This script must be 'sourced' using: . .\activate_env.ps1"
