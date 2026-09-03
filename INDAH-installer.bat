@echo off
setlocal enabledelayedexpansion
title INDAH Installer v1.0

echo =====================================================
echo   INDAH Installer v1.0
echo   Instrumental Detail Amplifier and Harmonizer
echo =====================================================
echo.
echo This is going to take a while. Go make yourself some coffee... :)
echo.

set "INSTALL_DIR=%cd%"
set "MINICONDA_DIR=%UserProfile%\Miniconda3"
set "ENV_DIR=%INSTALL_DIR%\env"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Windows-x86_64.exe"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"

call :privilege_checking
if errorlevel 1 exit /b 1

call :install_miniconda
if errorlevel 1 goto :error

call :create_conda_env
if errorlevel 1 goto :error

call :install_dependencies
if errorlevel 1 goto :error

call :verify_cuda
call :create_folders
call :create_launchers

echo.
echo =====================================================
echo   INDAH has been installed successfully!
echo =====================================================
pause
exit /b 0

:privilege_checking
NET SESSION >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ERROR: This script must NOT be run as Administrator!
    echo    Please close this window and run it as a normal user.
    pause
    exit /b 1
)
echo Running without Administrator privileges...
echo.
exit /b 0

:install_miniconda
if exist "%MINICONDA_DIR%\Scripts\conda.exe" (
    echo Miniconda already installed at %MINICONDA_DIR%
    exit /b 0
)
echo Downloading Miniconda installer...
powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile 'miniconda.exe' -UseBasicParsing}"
if not exist "miniconda.exe" (
    echo Download failed. Check internet connection.
    goto :error
)
echo Installing Miniconda (JustMe, no admin)...
start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_DIR%
if errorlevel 1 (
    echo Miniconda installation failed.
    goto :error
)
del miniconda.exe
echo Miniconda installed successfully
echo.
exit /b 0

:create_conda_env
echo Creating Conda environment (Python 3.11)...
call "%MINICONDA_DIR%\_conda.exe" create --no-shortcuts -y -k --prefix "%ENV_DIR%" python=3.11 >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to create conda environment
    goto :error
)
echo Conda environment created
echo.
exit /b 0

:install_dependencies
echo Installing Python dependencies...

echo Installing PyTorch 2.5.1 + CUDA 12.1...
"%ENV_DIR%\python.exe" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "%TORCH_INDEX%" --quiet --no-warn-script-location
if errorlevel 1 goto :error

echo Installing project dependencies...
"%ENV_DIR%\python.exe" -m pip install -r "%INSTALL_DIR%\requirements.txt" --quiet --no-warn-script-location
if errorlevel 1 goto :error

echo Dependencies installed successfully
echo.
exit /b 0

:verify_cuda
echo Verifying CUDA installation...
"%ENV_DIR%\python.exe" -c "import torch; assert torch.cuda.is_available(); name = torch.cuda.get_device_name(0); print('CUDA Verified: ' + name)"
if errorlevel 1 (
    echo    CUDA verification failed! Training will run on CPU.
    echo    Please check NVIDIA drivers or reinstall PyTorch.
) else (
    echo CUDA verification passed!
)
echo.
exit /b 0

:create_folders
echo Creating default folder structure...
if not exist "%INSTALL_DIR%\train\input" (
    mkdir "%INSTALL_DIR%\train\input"
    echo Created: train\input\
)
if not exist "%INSTALL_DIR%\train\target" (
    mkdir "%INSTALL_DIR%\train\target"
    echo Created: train\target\
)
echo Folder structure ready.
echo.
exit /b 0

:create_launchers
echo Creating launcher scripts...
(
echo @echo off
echo set "SCRIPT_DIR=%%~dp0"
echo call "%MINICONDA_DIR%\condabin\conda.bat" activate "%%SCRIPT_DIR%%env"
echo python train.py %%*
echo call "%MINICONDA_DIR%\condabin\conda.bat" deactivate
) > "%INSTALL_DIR%\run-INDAH.bat"

(
echo @echo off
echo set "SCRIPT_DIR=%%~dp0"
echo call "%MINICONDA_DIR%\condabin\conda.bat" activate "%%SCRIPT_DIR%%env"
echo echo Updating yt-dlp...
echo python -m pip install --upgrade yt-dlp --quiet
echo echo Starting Gradio app...
echo python inference.py
echo call "%MINICONDA_DIR%\condabin\conda.bat" deactivate
) > "%INSTALL_DIR%\run-inference.bat"

echo Launchers created: run-INDAH.bat, run-inference.bat
exit /b 0

:error
echo.
echo Installation failed. Check output above for details.
pause
exit /b 1