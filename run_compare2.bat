@echo off
REM Overnight: model comparison round 2 (Autoencoder + LOF + max-rank ensembles) for FY2568..2561
REM Run from Anaconda Prompt:  C:\thai-egp-anomaly-xai\run_compare2.bat     Log: logs\compare2.log
setlocal enabledelayedexpansion
cd /d C:\thai-egp-anomaly-xai\src
if not exist ..\logs mkdir ..\logs
set LOG=..\logs\compare2.log
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ===== START !date! !time! >> %LOG%
call conda activate egp-xai
if errorlevel 1 (
  echo ERROR: activate egp-xai failed >> %LOG%
  exit /b 1
)
for %%Y in (2568 2567 2566 2565 2564 2563 2562 2561) do (
  if not exist ..\reports\model_compare2_%%Y.json (
    echo ===== compare2 %%Y !date! !time! >> %LOG%
    python -u model_compare2.py --year %%Y >> %LOG% 2>&1
    echo ----- compare2 %%Y exit !errorlevel! !time! >> %LOG%
  )
)
echo ===== DONE !date! !time! >> %LOG%
endlocal
