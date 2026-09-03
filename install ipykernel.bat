@echo off
cd /d "%~dp0"

echo Checking environment...
if not exist "env\Scripts\python.exe" (
    echo ERROR: 'env' folder not found!
    echo Please run 'INDAH-installer.bat' first to install the environment.
    pause
    exit /b 1
)

echo Installing ipykernel...
.\env\Scripts\pip install ipykernel --quiet

echo.
echo Done! You can now use this environment in VS Code notebooks.
echo.
echo To select the kernel:
echo   1. Open analisis.ipynb in VS Code
echo   2. Click "Select Kernel" (top-right)
echo   3. Choose "Python Environments" ^> "INDAH-repo\env"
echo.
pause