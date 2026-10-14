@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Check Python and network.
  pause
  exit /b 1
)
echo Installation completed.
pause
