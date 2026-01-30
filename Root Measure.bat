@echo off
cd /d "%~dp0"
pip install -q -r requirements.txt
cd scripts
set PYTHONDONTWRITEBYTECODE=1
python measure_roots.py %*
pause
