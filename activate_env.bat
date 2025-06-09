@echo off
:: Batch script to activate the virtual environment and return to base folder
echo Activating virtual environment...
set BASE_DIR=%~dp0
set ENV_DIR=%BASE_DIR%age_gender_env
set ACTIVATE_SCRIPT=%ENV_DIR%\Scripts\activate.bat

:: Activate the virtual environment
call "%ACTIVATE_SCRIPT%"

:: Return to the base application folder
cd /d "%BASE_DIR%"
echo Returned to base application folder: %BASE_DIR%
echo Virtual environment is now active. You can run your Python scripts.
