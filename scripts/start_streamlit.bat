@echo off
REM Quick Start Script for healthdq-ai Streamlit UI (Windows)
REM Author: Agate Jarmakoviča

echo ==============================================
echo   healthdq-ai Streamlit Quick Start
echo ==============================================
echo.

REM Check if in correct directory
if not exist "src\healthdq\ui\streamlit_app.py" (
    echo [ERROR] streamlit_app.py not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

echo [OK] Project directory: OK
echo.

REM Check Python version
python --version
echo.

REM Check if Streamlit is installed
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Streamlit not installed
    echo Installing minimal dependencies...
    echo.

    pip install streamlit pandas pyyaml numpy

    echo [OK] Minimal dependencies installed
    echo.
) else (
    for /f "delims=" %%i in ('python -c "import streamlit; print(streamlit.__version__)"') do set STREAMLIT_VERSION=%%i
    echo [OK] Streamlit already installed: v%STREAMLIT_VERSION%
    echo.
)

REM Add src to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%cd%\src

echo Starting Streamlit application...
echo.
echo   Local URL: http://localhost:8501
echo   To stop: Press Ctrl+C
echo.
echo ==============================================
echo.

REM Launch Streamlit
streamlit run src\healthdq\ui\streamlit_app.py

pause
