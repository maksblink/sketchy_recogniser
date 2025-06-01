@echo off

:: Clone the repository
git clone https://github.com/maksblink/sketchy_recogniser.git

:: Change directory to project
cd sketchy_recogniser || exit /b 1

:: Install Poetry (assumes Python is already installed)
curl -sSL https://install.python-poetry.org/ | python

:: Create a new conda environment with Python 3.12.8 at the current directory
set "VENV_PATH=%cd%.venv"
conda create --yes --prefix "%VENV_PATH%" python=3.12.8

:: Get the path to the new Python interpreter
for /f "delims=" %%i in ('conda run --prefix "%VENV_PATH%" where python') do set "PYTHON_PATH=%%i"

:: Set Poetry to use the new Python interpreter
poetry env use "%PYTHON_PATH%"

:: Install dependencies
poetry install

:: Run the main script
poetry run python main.py