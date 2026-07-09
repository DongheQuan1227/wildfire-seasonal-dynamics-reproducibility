@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found on PATH.
  echo Install Python 3.11 and the packages in requirements.txt.
  pause
  exit /b 2
)
python run_all.py --profile full --resume --verify
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" echo The workflow stopped with exit code %EXITCODE%. See run_logs for details.
pause
exit /b %EXITCODE%
