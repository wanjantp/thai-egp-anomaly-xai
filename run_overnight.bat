@echo off
REM Overnight jobs: create env if missing -> IF+SHAP model FY2566, 2567 -> features FY2565..2558
REM Run from Anaconda Prompt:  C:\thai-egp-anomaly-xai\run_overnight.bat
REM Log: logs\overnight.log  (never touches data\raw; skips jobs whose output exists)
setlocal enabledelayedexpansion
cd /d C:\thai-egp-anomaly-xai
if not exist logs mkdir logs
set LOG=logs\overnight.log
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ===== START !date! !time! >> %LOG%

call conda env list | findstr /b /c:"egp-xai " >nul
if errorlevel 1 (
  echo ===== create env egp-xai !time! >> %LOG%
  call conda env create -f environment.yml >> %LOG% 2>&1
)
call conda activate egp-xai
if errorlevel 1 (
  echo ERROR: activate egp-xai failed >> %LOG%
  exit /b 1
)
python -c "import duckdb, polars, pandas, sklearn, shap, matplotlib; print('env ok', duckdb.__version__)" >> %LOG% 2>&1
if errorlevel 1 (
  echo ERROR: packages missing >> %LOG%
  exit /b 1
)

for %%Y in (2566 2567) do (
  if not exist data\processed\scores_%%Y_le500k.parquet (
    echo ===== model %%Y !date! !time! >> %LOG%
    python -u src\model_iforest.py --year %%Y >> %LOG% 2>&1
    echo ----- model %%Y exit !errorlevel! !time! >> %LOG%
  )
)

for %%Y in (2565 2564 2563 2562 2561 2560 2559 2558) do (
  if not exist data\processed\features_%%Y.parquet (
    echo ===== features %%Y !date! !time! >> %LOG%
    python -u src\features.py --years %%Y --threads 4 --db "%TEMP%\egp_feat_%%Y.duckdb" >> %LOG% 2>&1
    echo ----- features %%Y exit !errorlevel! !time! >> %LOG%
  )
)

echo ===== DONE !date! !time! >> %LOG%
endlocal
