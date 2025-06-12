@echo off
setlocal enabledelayedexpansion

:: Set working directory to script location
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

:: Define Python installation target
set "PYTHON_DIR=%ROOT_DIR%Python312"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"

:: === Check for existing Python 3.12+ ===
echo Sprawdzanie instalacji Python

set FOUND_PYTHON=0

:: Check with existing py/command-line Python
for %%I in (py python) do (
    for /f "tokens=2 delims= " %%V in ('%%I --version 2^>nul') do (
        for /f "tokens=1,2 delims=." %%a in ("%%V") do (
            set MAJOR=%%a
            set MINOR=%%b
            if !MAJOR! GEQ 4 (
                set FOUND_PYTHON=1
            ) else if !MAJOR!==3 (
                if !MINOR! GEQ 12 (
                    set FOUND_PYTHON=1
                )
            )
        )
    )
)

if !FOUND_PYTHON! EQU 1 (
    echo Python 3.12+ zaisntalowany.
    goto SetupProject
)

:: === Python Not Found - Install 3.12.8 Locally ===
echo Instalowanie Python3.12.8

set "PYTHON_VERSION=3.12.8"
set "INSTALLER_NAME=python-%PYTHON_VERSION%-amd64.exe"
set "INSTALLER_PATH=%TEMP%\%INSTALLER_NAME%"

:: Download Python installer
echo Downloading Python installer...
curl -o "%INSTALLER_PATH%" "https://www.python.org/ftp/python/%PYTHON_VERSION%/%INSTALLER_NAME%"

if not exist "%INSTALLER_PATH%" (
    echo ERROR: Failed to download Python installer.
    exit /b 1
)

:: Install Python silently into local folder
echo Instalowanie...
"%INSTALLER_PATH%" /quiet InstallAllUsers=0 PrependPath=0 Include_test=0 TargetDir="%PYTHON_DIR%"

:: Cleanup
del "%INSTALLER_PATH%"

:: Confirm installation
if not exist "%PYTHON_EXE%" (
    echo Nie udalo sie zainstalowac automatycznie, zainstaluj Python3.12.8 z linku: 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe' 
    exit /b 1
)

echo Zainstalowano Python do: %PYTHON_DIR%.

:SetupProject
echo === Konfiguracja projektu ===

:: Update PATH to include local Python and Scripts
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%"

:: Install Poetry locally (in %ROOT_DIR%\poetry)
if not exist "%ROOT_DIR%poetry" (
    echo Installing Poetry locally...
    curl -sSL https://install.python-poetry.org -o install-poetry.py
    "%PYTHON_EXE%" install-poetry.py --install-dir "%ROOT_DIR%poetry" --no-modify-path
    del install-poetry.py
)

:: Add Poetry to PATH
set "PATH=%ROOT_DIR%poetry\bin;%PATH%"

:: Clone the project repo if not already present
if not exist sketchy_recogniser (
    echo Cloning project repository...
    git clone https://github.com/maksblink/sketchy_recogniser.git
)

cd sketchy_recogniser || exit /b 1

:: Create virtual environment if not exists
if not exist .venv (
    "%PYTHON_EXE%" -m venv .venv
)

:: Activate venv
call .venv\Scripts\activate.bat

:: Tell Poetry to use the venv Python
poetry env use "%CD%\.venv\Scripts\python.exe"

:: Install dependencies
poetry install --no-root

:: Run the app
poetry run python main.py

endlocal
pause
