@echo off
cd /d "%~dp0"
set "LOCAL_VENV=%USERPROFILE%\.venv-rootmeasure"
if exist "%LOCAL_VENV%\Scripts\python.exe" (
    "%LOCAL_VENV%\Scripts\python.exe" gui\app.py
    if errorlevel 1 (
        echo.
        echo === App exited with an error ===
        pause
    )
) else (
    echo No virtual environment found at %LOCAL_VENV%
    echo.
    echo Run these commands to set it up:
    echo   python -m venv "%LOCAL_VENV%"
    echo   "%LOCAL_VENV%\Scripts\pip" install -r install\requirements.txt
    pause
)
