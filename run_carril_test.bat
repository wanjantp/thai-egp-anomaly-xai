@echo off
REM Carril (2021)-style splitting test (GIQ manuscript v5): bunching below THB 500,000 without repeat unit-vendor transactions.
REM Run from Anaconda Prompt. Resumable (cache: data\processed\_revision\carril_YYYY.parquet). About 1-2 minutes per year.
REM Output: reports\carril_test.md   Log: logs\carril_test.log
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
python -u carril_test.py > ..\logs\carril_test.log 2>&1
type ..\logs\carril_test.log
echo.
echo Done. Open reports\carril_test.md
pause
endlocal
