@echo off
cd /d "%~dp0"
if exist .venv-win\Scripts\pythonw.exe (
    start "" .venv-win\Scripts\pythonw.exe gui\app.py
) else if exist .venv\Scripts\pythonw.exe (
    start "" .venv\Scripts\pythonw.exe gui\app.py
) else (
    echo No Windows virtual environment found.
    echo Run: python -m venv .venv-win
    echo Then: .venv-win\Scripts\pip install -r install\requirements.txt
    pause
)
