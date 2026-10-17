#!/usr/bin/env bash
# One-time setup and checks on Chopin before the trip (about 20-30 min). Safe to run again.
#   bash queue/setup_trip.sh NTFY_TOPIC
# Ends with a list of OK / FAILED checks. Start the queue only when everything is OK.
set -uo pipefail
EGP=${EGP:-$HOME/egp}; REPO=$EGP/repo
TOPIC=${1:-}
cd "$REPO" || exit 1
RESULTS=()
ok()   { echo "   OK: $1"; RESULTS+=("OK      $1"); }
bad()  { echo "   FAILED: $1"; RESULTS+=("FAILED  $1"); }
step() { echo; echo "=== $*"; }

step "1/8 conda environments"
eval "$(conda shell.bash hook)"
PY_CPU=$(conda run -n egp-xai python -c 'import sys; print(sys.executable)' 2>/dev/null)
[ -x "$PY_CPU" ] && ok "egp-xai python: $PY_CPU" || { bad "egp-xai env missing"; exit 1; }
if ! conda env list | grep -qE '^egp-gpu\s'; then
  echo "   creating egp-gpu (downloads PyTorch, ~5-10 min)"; conda env create -q -f envs/gpu.yml
fi
PY_GPU=$(conda run -n egp-gpu python -c 'import sys; print(sys.executable)' 2>/dev/null)
if "$PY_GPU" -c "import torch, sentence_transformers as s; assert torch.cuda.is_available(); print('   torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'st', s.__version__)"; then
  ok "egp-gpu sees the GPU"; else bad "egp-gpu cannot use the GPU"; fi

step "2/8 settings file $EGP/egpq.env"
if [ -z "$TOPIC" ] && [ -f "$EGP/egpq.env" ]; then TOPIC=$(grep '^NTFY_TOPIC=' "$EGP/egpq.env" | cut -d= -f2); fi
cat > "$EGP/egpq.env" <<ENV
EGP=$EGP
REPO=$REPO
PY_CPU=$PY_CPU
PY_GPU=$PY_GPU
NTFY_TOPIC=$TOPIC
HF_HOME=$EGP/models/hf
EGPQ_HEARTBEAT_H=8
EGPQ_MIN_FREE_GB=40
ENV
chmod 600 "$EGP/egpq.env"; mkdir -p "$EGP/models/hf"
grep -q 'alias egpq=' ~/.bashrc || echo "alias egpq='$PY_CPU $REPO/queue/egpq.py'" >> ~/.bashrc
ok "settings written (topic: ${TOPIC:-none})"

step "3/8 work trees (separate copies so reruns never touch ~/egp/data or the PC results)"
for t in p2 p1repro; do
  [ -d "$EGP/work/$t" ] || git worktree add --detach "$EGP/work/$t" HEAD >/dev/null
  mkdir -p "$EGP/work/$t"/{logs,models,reports/figures,data/processed}
  ln -sfn "$EGP/data/raw" "$EGP/work/$t/data/raw"
done
ln -sfn "$EGP/data/interim" "$EGP/work/p2/data/interim"
mkdir -p "$EGP/work/p1repro/data/interim"
ok "trees at commit $(git -C "$EGP/work/p2" rev-parse --short HEAD)"
if [ -d "$EGP/work/pc_snapshot/reports" ]; then ok "PC reports snapshot present ($(ls "$EGP/work/pc_snapshot/reports" | wc -l) files)";
else bad "PC reports snapshot missing: on the PC run  python src\\sync_to_chopin.py reports --remote-root egp/work/pc_snapshot"; fi

step "4/8 imports of paper 1-2 scripts"
if (cd "$EGP/work/p2/src" && "$PY_CPU" -c "import features, model_iforest, model_compare, model_compare2, model_compare3, screen_group, bunching" ); then
  ok "paper 1-2 modules import"; else bad "paper 1-2 imports"; fi

step "5/8 queue self-test (about 1-2 min; your phone should get several notifications)"
"$PY_CPU" queue/egpq.py notify-test
rm -rf "$EGP/runs/queue/selftest"
EGPQ_MIN_FREE_GB=1 "$PY_CPU" queue/egpq.py run queue/selftest.tsv >/dev/null 2>&1
S=$("$PY_CPU" queue/egpq.py status selftest)
echo "$S" | grep -E '^\s+(ok|XX|TO|--) '
if echo "$S" | grep -q 'done 3, failed 1, skipped 1, timeout 1'; then ok "runner self-test"; else bad "runner self-test"; fi

step "6/8 foundation tables on FY2568 only (smoke test, a few minutes)"
rm -rf "$EGP/scratch/smoke_fdn"
if EGPQ_THREADS=4 TMPDIR="$EGP/scratch" "$PY_CPU" foundation/fdn_tables.py all --years 2568 --out "$EGP/scratch/smoke_fdn" | tail -3; then
  ok "foundation tables"; else bad "foundation tables"; fi

step "7/8 the three embedding models on 20,000 vendor names (downloads ~6 GB of models once)"
for m in bge-m3 e5-large qwen3-emb-0.6b; do
  if CUDA_VISIBLE_DEVICES=0 HF_HOME="$EGP/models/hf" "$PY_GPU" foundation/fdn_embed.py --set vendor_names --model $m \
       --fdn "$EGP/scratch/smoke_fdn" --limit 20000 --out "$EGP/scratch/smoke_emb/$m" 2>&1 | grep -E "rows/s|Error|error" | tail -2; then
    [ -f "$EGP/scratch/smoke_emb/$m/manifest.json" ] && ok "embedding $m" || bad "embedding $m"
  else bad "embedding $m"; fi
done

step "8/8 watchdog in crontab (restarts the runner within 15 min after a reboot or crash)"
LINE="*/15 * * * * $PY_CPU $REPO/queue/egpq.py run $REPO/queue/trip.tsv --bg --only-if-started >/dev/null 2>&1"
(crontab -l 2>/dev/null | grep -v 'queue/egpq.py'; echo "$LINE") | crontab - && ok "crontab watchdog" || bad "crontab"

echo; echo "================ SUMMARY"; printf '%s\n' "${RESULTS[@]}"
if printf '%s\n' "${RESULTS[@]}" | grep -q FAILED; then echo "Fix the FAILED lines (send them to Claude) before starting.";
else echo "All checks passed. Start with:  egpq start queue/trip.tsv   (open a new shell first so the alias exists)"; fi
