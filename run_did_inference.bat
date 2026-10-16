@echo off
REM DiD inference with band-level shocks (Reviewer-2 round 2, item 4): Newey-West over monthly gaps,
REM placebo break dates, placebo bands.  Run from Anaconda Prompt.  Needs the DiD cache from run_revision_checks.bat.
REM Output: reports\did_inference.md   Log: logs\did_inference.log
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
python -u did_inference.py > ..\logs\did_inference.log 2>&1
type ..\logs\did_inference.log
echo.
echo Done. Open reports\did_inference.md
pause
endlocal
