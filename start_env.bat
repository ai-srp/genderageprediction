@echo off
echo This command will create a new Command Prompt with the virtual environment activated.
echo.
echo Starting new Command Prompt with activated environment...

start cmd.exe /k "cd /d "%~dp0" && call age_gender_env\Scripts\activate.bat && echo Virtual environment activated && echo Current directory: %~dp0"

echo.
echo A new Command Prompt window should open with the activated environment.
echo If a new window didn't open, please run this command manually:
echo start cmd.exe /k "cd /d "%~dp0" && call age_gender_env\Scripts\activate.bat"
