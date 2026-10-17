#!/usr/bin/env bash
# Paper 2, round 4 on Chopin (Linux). Same steps as run_round4.bat.
#   bash run_round4.sh [JOBS]      JOBS = years run at once (default 2; each needs ~8 GB RAM)
# Run inside tmux. Log: logs/round4.log
set -euo pipefail
cd "$(dirname "$0")/src"
mkdir -p ../logs ../reports
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export TMPDIR="${TMPDIR:-$HOME/egp/scratch}"; mkdir -p "$TMPDIR"
JOBS="${1:-2}"
# limit threads per process so parallel years do not take every core on a shared machine
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-4}"
LOG=../logs/round4.log
YEARS="2568 2567 2566 2565 2564 2563 2562 2561"
echo "===== START $(date -Is) host=$(hostname) commit=$(git rev-parse --short HEAD) jobs=$JOBS" >> "$LOG"

run_year() {  # $1 = script, $2 = year, $3 = report prefix
  [ -f "../reports/$3_$2.json" ] && return 0
  echo "===== $1 $2 $(date -Is)" >> "$LOG"
  python -u "$1" --year "$2" > "../logs/round4_$3_$2.log" 2>&1; rc=$?
  echo "----- $1 $2 exit $rc $(date -Is)" >> "$LOG"
}
export -f run_year; export LOG

printf '%s\n' $YEARS | xargs -P "$JOBS" -I{} bash -c 'run_year screen_group.py {} screen_group'
python -u summarize_screen_group.py >> "$LOG" 2>&1
printf '%s\n' $YEARS | xargs -P "$JOBS" -I{} bash -c 'run_year model_compare3.py {} model_compare3'
python -u summarize_compare3.py >> "$LOG" 2>&1
echo "===== DONE $(date -Is)" >> "$LOG"
