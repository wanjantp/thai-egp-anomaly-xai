@echo off
REM Overnight: model comparison round 3 (consistent data-level injection, AE max_iter 300) for FY2568..2561
REM Run from Anaconda Prompt:  C:\thai-egp-anomaly-xai\run_compare3.bat     Log: logs\compare3.log
setlocal enabledelayedexpansion
cd /d C:\thai-egp-anomaly-xai\src
if not exist ..\logs mkdir ..\logs
set LOG=..\logs\compare3.log
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ===== START !date! !time! >> %LOG%
call conda activate egp-xai
if errorlevel 1 (
  echo ERROR: activate egp-xai failed >> %LOG%
  exit /b 1
)
for %%Y in (2568 2567 2566 2565 2564 2563 2562 2561) do (
  if not exist ..\reports\model_compare3_%%Y.json (
    echo ===== compare3 %%Y !date! !time! >> %LOG%
    python -u model_compare3.py --year %%Y >> %LOG% 2>&1
    echo ----- compare3 %%Y exit !errorlevel! !time! >> %LOG%
  )
)
python -u summarize_compare3.py >> %LOG% 2>&1
echo ===== DONE !date! !time! >> %LOG%
endlocal
