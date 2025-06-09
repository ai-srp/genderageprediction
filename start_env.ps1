# Wrapper script to properly activate the virtual environment
# This script creates a temporary script that gets executed by the shell

$tempScriptPath = [System.IO.Path]::GetTempFileName() + ".ps1"

@"
# Get path to the virtual environment
`$scriptPath = "$(Get-Location)"
`$envPath = Join-Path -Path `$scriptPath -ChildPath "age_gender_env"
`$activateScript = Join-Path -Path `$envPath -ChildPath "Scripts\Activate.ps1"

# Activate the virtual environment
Write-Host "Activating virtual environment from: `$envPath"
. `$activateScript

# Set location back to the project directory
Set-Location "`$scriptPath"
Write-Host "Returned to base application folder: `$scriptPath"
Write-Host "Virtual environment is now active. You can run your Python scripts."

# Create a new PowerShell session with the virtual environment activated
`$env:VIRTUAL_ENV_PROMPT = "(age_gender_env) "
"@ | Out-File -FilePath $tempScriptPath

Write-Host "To activate the virtual environment, copy and run this command:"
Write-Host ""
Write-Host "powershell -NoExit -File $tempScriptPath" -ForegroundColor Green
Write-Host ""
Write-Host "This will open a new PowerShell window with the virtual environment activated."
