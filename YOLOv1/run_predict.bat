@echo off
cd /d "%~dp0"
if not exist "checkpoints\raw.pt" (
  echo Missing checkpoints\raw.pt. Copy the final model weights here first.
  pause
  exit /b 1
)
python predict.py %*
if errorlevel 1 echo Prediction failed. See the error above.
pause
