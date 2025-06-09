@echo off
echo Running Age & Gender AI setup script...
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in the PATH. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Execute the setup script
python setup_env.py

if %errorlevel% neq 0 (
    echo.
    echo Setup failed! Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo Setup completed! To activate the environment, run:
echo    age_gender_env\Scripts\activate.bat
echo.
echo After activation, you can run the training with:
echo    python train_model.py
echo.
echo Or run the web app with:
echo    python web_app.py
echo.

pause