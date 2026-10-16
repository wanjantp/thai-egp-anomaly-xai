@echo off
REM Check: specific/selection-method contracts above THB 500,000 after the 2017 Act (exceptions, circular ว 123)
REM Run from Anaconda Prompt:  C:\thai-egp-anomaly-xai\run_check_v123.bat     Output: reports\check_specific_above500k.md
setlocal
cd /d C:\thai-egp-anomaly-xai\src
if not exist ..\logs mkdir ..\logs
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
call conda activate egp-xai
if errorlevel 1 (
  echo ERROR: activate egp-xai failed
  exit /b 1
)
python -u check_specific_above500k.py > ..\logs\check_v123.log 2>&1
type ..\logs\check_v123.log
echo.
echo Done. Open reports\check_specific_above500k.md
pause
endlocal
