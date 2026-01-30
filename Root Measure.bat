@echo off
cd /d "%~dp0scripts"
set PYTHONDONTWRITEBYTECODE=1
python measure_roots.py %*
pause
