@echo off
REM Round 4 (overnight or daytime, ~45-60 min):
REM   1) model comparison round 3 RERUN with float32 fix (v2) for FY2568..2561  -> reports\model_compare3_*.md
REM   2) ceiling-aware purchase-level screener for FY2568..2561                 -> reports\screen_group_*.md
REM Run from Anaconda Prompt:  C:\thai-egp-anomaly-xai\run_round4.bat     Log: logs\round4.log
setlocal enabledelayedexpansion
cd /d C:\thai-egp-anomaly-xai\src
if not exist ..\logs mkdir ..\logs
set LOG=..\logs\round4.log
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ===== START !date! !time! >> %LOG%
call conda activate egp-xai
if errorlevel 1 (
  echo ERROR: activate egp-xai failed >> %LOG%
  exit /b 1
)
for %%Y in (2568 2567 2566 2565 2564 2563 2562 2561) do (
  if not exist ..\reports\screen_group_%%Y.json (
    echo ===== screen_group %%Y !date! !time! >> %LOG%
    python -u screen_group.py --year %%Y >> %LOG% 2>&1
    echo ----- screen_group %%Y exit !errorlevel! !time! >> %LOG%
  )
)
python -u summarize_screen_group.py >> %LOG% 2>&1
for %%Y in (2568 2567 2566 2565 2564 2563 2562 2561) do (
  if not exist ..\reports\model_compare3_%%Y.json (
    echo ===== compare3 v2 %%Y !date! !time! >> %LOG%
    python -u model_compare3.py --year %%Y >> %LOG% 2>&1
    echo ----- compare3 v2 %%Y exit !errorlevel! !time! >> %LOG%
  )
)
python -u summarize_compare3.py >> %LOG% 2>&1
echo ===== DONE !date! !time! >> %LOG%
endlocal
