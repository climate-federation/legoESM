#!/bin/bash
# LES-truth suite campaign: emit the Q1 flux sweep (GPU) -> Q1a structural diagnostic
# -> derivative-free tuning of 8 closures per flux (CPU) -> Q2/Q3 scorecard.
# IDEMPOTENT/RESUMABLE: skips artifacts + tuned JSONs that already exist, so a
# timed-out job can just be resubmitted. All output under results/les_suite/.
set -u
REPO="${REPO:-/burg-archive/glab/users/ac5006/legoESM}"
cd "$REPO" || exit 2
PY=.venv/bin/python
OUT=results/les_suite
ART=$OUT/artifacts
mkdir -p "$ART" "$OUT/tuned" "$OUT/campaign"
LOG=$OUT/campaign/progress.log
say(){ echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

# Single-instance guard. A campaign orphaned from a dead parent keeps running
# (reparented to init); a resubmit then races it — two copies halve each other's
# CPU throughput and clobber the same tuned JSON. flock (held on fd 9 for the
# shell's lifetime, auto-released on exit) makes a concurrent resubmit a no-op.
exec 9>"$OUT/campaign/.campaign.lock"
if ! flock -n 9; then
  say "another campaign instance holds the lock -> exiting (no-op)"
  exit 0
fi

say "=== LES-suite campaign START (host=$(hostname)) ==="

# 1) Q1 buoyancy-axis sweep: free-convective CBLs at these surface fluxes (GPU).
FLUXES="0.02 0.04 0.08 0.12"     # 0.06 already emitted unlabeled -> 5 flux points
for q0 in $FLUXES; do
  label="q0_${q0}"
  f="$ART/cbl_nieuwstadt__lasd__${label}.npz"
  if [ -f "$f" ]; then say "skip emit $label (exists)"; continue; fi
  say "emit $label"
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_les_suite.py \
    --case cbl_nieuwstadt --q0 "$q0" --label "$label" --hours 2.0 --frames 12 \
    --output "$ART" >>"$LOG" 2>&1 || say "  emit $label FAILED (continuing)"
done

# 2) Q1a structural-ceiling sweep (fast, CPU) over ALL flux artifacts.
say "Q1 counter-gradient sweep"
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY \
  scripts/validate/les_suite/q1_counter_gradient_sweep.py \
  --artifacts-dir "$ART" --output "$OUT/q1_counter_gradient_sweep.json" >>"$LOG" 2>&1 \
  || say "  Q1 sweep FAILED"

# 3) Q2 tuning campaign (CPU, slow): each flux artifact x each closure.
for f in "$ART"/cbl_nieuwstadt__lasd*.npz; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .npz)
  for s in smagorinsky louis holtslag_boville ysu tke mynn25 clubb_lite edmf; do
    out="$OUT/tuned/${base}__${s}__df.json"
    if [ -f "$out" ]; then say "skip tune $base $s (exists)"; continue; fi
    say "tune $base $s"
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY scripts/run/tune_scm_to_les.py \
      --artifact "$f" --scheme "$s" --tiers 1 --n-random 3 --dt 10 --nlev 24 \
      --output "$out" >>"$LOG" 2>&1 || say "  tune $base $s FAILED (continuing)"
  done
done

# 4) Q2/Q3 scorecard.
say "scorecard"
JAX_PLATFORMS=cpu $PY scripts/validate/les_suite/build_les_scorecard.py \
  --tuned-dir "$OUT/tuned" --output "$OUT/scorecard.md" >>"$LOG" 2>&1 \
  || say "  scorecard FAILED"

say "=== LES-suite campaign DONE ==="
