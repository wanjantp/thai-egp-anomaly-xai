@echo off
REM Daytime jobs: IF+SHAP model (scope le500k) for FY2561-2565, skips years already done
REM Run from Anaconda Prompt:  C:\thai-egp-anomaly-xai\run_day.bat     Log: logs\day.log
setlocal enabledelayedexpansion
cd /d C:\thai-egp-anomaly-xai
if not exist logs mkdir logs
set LOG=logs\day.log
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ===== START !date! !time! >> %LOG%
call conda activate egp-xai
if errorlevel 1 (
  echo ERROR: activate egp-xai failed >> %LOG%
  exit /b 1
)
for %%Y in (2565 2564 2563 2562 2561) do (
  if not exist data\processed\scores_%%Y_le500k.parquet (
    echo ===== model %%Y !date! !time! >> %LOG%
    python -u src\model_iforest.py --year %%Y >> %LOG% 2>&1
    echo ----- model %%Y exit !errorlevel! !time! >> %LOG%
  )
)
echo ===== DONE !date! !time! >> %LOG%
endlocal
