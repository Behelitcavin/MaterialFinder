@echo off
cd /d "%~dp0"
echo ===== Sozai Sagashi setup =====

set "PY="
for %%v in (3.12 3.11 3.13 3.10) do (
  if not defined PY (
    py -%%v -c "pass" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python 3.10 - 3.13 was not found.
  echo Install Python from https://www.python.org/downloads/ and run setup.bat again.
  pause
  exit /b 1
)
echo Python: %PY%

rem Create .venv here. If .venv was copied from another PC it does not run, so make it again.
".venv\Scripts\python.exe" -c "pass" >nul 2>&1 || (
  if exist ".venv" rmdir /s /q ".venv"
  %PY% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"

set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
nvidia-smi >nul 2>&1 && set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
echo PyTorch: %TORCH_INDEX%
"%VPY%" -m pip install --upgrade pip || goto :fail
"%VPY%" -m pip install torch --index-url %TORCH_INDEX% || goto :fail
"%VPY%" -m pip install -r requirements.txt || goto :fail
"%VPY%" sozai_sagashi.py --setup || goto :fail

echo.
echo ===== Done! =====
echo Start: run.bat  or  Start menu - Sozai Sagashi
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup failed. Please check the messages above.
pause
exit /b 1
