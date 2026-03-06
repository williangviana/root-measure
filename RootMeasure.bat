@echo off
cd /d "%~dp0"
set "LOCAL_VENV=%USERPROFILE%\.venv-rootmeasure"
if exist "%LOCAL_VENV%\Scripts\pythonw.exe" (
    start "" "%LOCAL_VENV%\Scripts\pythonw.exe" gui\app.py
) else (
    echo No virtual environment found at %LOCAL_VENV%
    echo.
    echo Run these commands to set it up:
    echo   python -m venv "%LOCAL_VENV%"
    echo   "%LOCAL_VENV%\Scripts\pip" install -r install\requirements.txt
    pause
)
