@echo off
REM Splitting across vendors (v3 of the GIQ manuscript): how much does the same-vendor measure miss?
REM Run from Anaconda Prompt. Resumable (cache: data\processed\_revision\xv_YYYY.parquet). About 1-3 minutes per year.
REM Output: reports\split_cross_vendor.md   Log: logs\split_cross_vendor.log
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
python -u split_cross_vendor.py > ..\logs\split_cross_vendor.log 2>&1
type ..\logs\split_cross_vendor.log
echo.
echo Done. Open reports\split_cross_vendor.md
pause
endlocal
