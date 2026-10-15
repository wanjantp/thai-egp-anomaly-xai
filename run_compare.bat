@echo off
REM Compare anomaly models (IF vs ECOD vs HBOS vs ensemble) for FY2561-2568, ~30-60 s per year
REM Run from Anaconda Prompt:  C:\thai-egp-anomaly-xai\run_compare.bat     Log: logs\compare.log
setlocal enabledelayedexpansion
cd /d C:\thai-egp-anomaly-xai\src
if not exist ..\logs mkdir ..\logs
set LOG=..\logs\compare.log
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ===== START !date! !time! >> %LOG%
call conda activate egp-xai
if errorlevel 1 (
  echo ERROR: activate egp-xai failed >> %LOG%
  exit /b 1
)
for %%Y in (2561 2562 2563 2564 2565 2566 2567 2568) do (
  if not exist ..\reports\model_compare_%%Y.json (
    echo ===== compare %%Y !date! !time! >> %LOG%
    python -u model_compare.py --year %%Y >> %LOG% 2>&1
    echo ----- compare %%Y exit !errorlevel! !time! >> %LOG%
  )
)
echo ===== DONE !date! !time! >> %LOG%
endlocal
