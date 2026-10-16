@echo off
REM Robustness checks for the GIQ manuscript (Reviewer-2 audit): DiD margins + balanced panel, alternative anchors,
REM agency-clustered bootstrap, tax-ID vendor check.  Run from Anaconda Prompt.  Log: logs\revision_checks.log
REM Output: reports\revision_checks.md  (resumable: cached aggregates in data\processed\_revision)
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
python -u revision_checks.py >> ..\logs\revision_checks.log 2>&1
type ..\logs\revision_checks.log
echo.
echo Done. Open reports\revision_checks.md
pause
endlocal
