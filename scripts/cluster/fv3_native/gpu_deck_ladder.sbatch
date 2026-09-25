#!/bin/bash
#SBATCH --job-name=fv3_deck_ladder
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=12
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
# A scaling ladder across card models measures the cards, not the scaling, so the
# model is PINNED here. rtx8000 is the only model with enough nodes in `short` for
# 24 ranks (18 nodes; 12 survive the exclusion below, which is exactly what we need).
#SBATCH --constraint=rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclusive
#SBATCH --time=04:00:00
# these compile the sharded step and then hang in its first execution
# (pinned-node A/B, jobs 9777265 good vs 9777266 hung); delete when fixed
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/gpu_deck_ladder_%j.log
# GPU DECK LADDER: for each deck, build its own six-face reference and then
# score the sharded step against it at 6*kt*kt ranks, all inside ONE
# allocation. Closes the two GPU gaps the rank ladder does not: decks larger
# than C24, and more than one step.
# Written by GLM-5.2; reviewed by Claude and codex.
set -uo pipefail

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
WT=${WT:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
GATE="$WT/scripts/validate/fv3_native/tiled_m6_model_gate.py"
BASE=/burg-archive/glab/users/pg2328/fv3_duo_gaps
cd "$WT" || exit 3

export PYTHONPATH="$WT/packages/core:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ocean:$WT/packages/ml:$WT/packages/tools:$WT/src"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# one GPU per task: the cgroup hides the node's other GPU but the driver
# still counts it, and jax then fails on CUDA:1 (job 9650417)
export JAX_LOCAL_DEVICE_IDS=0
export NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# autotuning deadlocks when several ranks compile (job 9746516); level 0 is
# the only setting that completes, and it costs nothing (job 9776324)
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"

CPUS="${SLURM_CPUS_PER_TASK:-8}"
NNODES="${SLURM_JOB_NUM_NODES:-12}"
OUT="$BASE/gpu_deck_ladder_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT" || { echo "FATAL: cannot create $OUT"; exit 3; }
for f in "$PY" "$GATE"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 3; }
done

echo "host $(hostname)  nodes ${SLURM_JOB_NODELIST:-?}  ($NNODES) cpus/task $CPUS"
echo "sha $(git -C "$WT" rev-parse HEAD)  dirty $(git -C "$WT" status --porcelain | wc -l)"
echo "XLA_FLAGS=$XLA_FLAGS"
echo "outdir $OUT"

# A ladder across GPU models measures the cards, not the model. The allocation is pinned to
# rtx8000 above; the models are collected here anyway and a mixed allocation is
# refused, because a pin can be dropped or a feature renamed.
srun -N "$NNODES" -n "$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=0 \
  nvidia-smi --query-gpu=name --format=csv,noheader > "$OUT/gpu_models.txt" 2>"$OUT/gpu_models.err"
GPU_RC=$?
if [ "$GPU_RC" -ne 0 ]; then
  echo "REFUSED: could not collect GPU models (rc=$GPU_RC)"; sed 's/^/  /' "$OUT/gpu_models.err"; exit 4
fi
if [ "$(sort -u "$OUT/gpu_models.txt" | grep -c .)" -ne 1 ]; then
  echo "REFUSED: the allocation straddles GPU models:"; sort -u "$OUT/gpu_models.txt" | sed 's/^/  /'; exit 4
fi
echo "gpu model: $(sort -u "$OUT/gpu_models.txt")"

#        N | KM | KT | PAD | NSPLIT | DT  | STEPS
DECKS=(
  "24|5|2|5|1|900|2"
  "48|10|2|11|3|900|2"
  # C96 at 6 ranks first: the 24-rank row hung on first dispatch on UNIFORM
  # RTX 8000 (ladder 9867745, compiled 558 s, no dispatch in ~30 min); if this
  # row completes the hang is the 24-rank collective at C96 sizes, not the deck
  "96|10|1|5|3|450|2"
  "96|10|2|11|3|450|2"
)
REF_CAP=${REF_CAP:-1800}
SCORE_CAP=${SCORE_CAP:-2400}
FAIL=0
SUMMARY=()

for d in "${DECKS[@]}"; do
  # SKIP_DECK=<glob> drops rows (e.g. "96|10|2|*" while the 24-rank C96 hang is open)
  [ -n "${SKIP_DECK:-}" ] && [[ "$d" == $SKIP_DECK ]] && { echo "===== SKIPPED $d (SKIP_DECK) ====="; continue; }
  IFS='|' read -r N KM KT PAD NSPLIT DT STEPS <<<"$d"
  RANKS=$((6 * KT * KT)); SNODES=$((RANKS / 2))
  REF="$OUT/ref_c${N}_km${KM}_kt${KT}.npz"
  RLOG="$OUT/c${N}_kt${KT}_ref.log"; SLOG="$OUT/c${N}_kt${KT}_score.log"
  echo ""
  echo "===== DECK C$N km=$KM kt=$KT pad=$PAD n_split=$NSPLIT dt=$DT steps=$STEPS ====="

  # STEP A -- the reference, 6 ranks on 3 nodes.
  #  * --distributed is REQUIRED even here: without it each task is its own
  #    process seeing one GPU and --flat-ref refuses ("needs exactly 6
  #    devices"). The six tasks must form ONE 6-device job.
  #  * --kt 1, NOT the deck's kt (codex 2026-09-16): the gate checks
  #    process_count == 6*kt*kt BEFORE it reaches the --flat-ref branch, so
  #    a kt=2 reference build on six ranks returns 2 every time. The
  #    reference is the six-FACE state and carries no window geometry, which
  #    is why a kt=1 build is what the kt=2 scoring run consumes (as
  #    gpu_ref_c24_v2.npz already does, jobs 9756805 -> 9756806).
  timeout --kill-after=60 "$REF_CAP" \
    srun -N 3 -n 6 --ntasks-per-node=2 --gpus-per-task=1 --kill-on-bad-exit=0 -c "$CPUS" \
      "$PY" "$GATE" --n "$N" --km "$KM" --kt 1 --pad "$PAD" \
      --n-split "$NSPLIT" --dt "$DT" --steps "$STEPS" \
      --distributed --flat-ref --save-npz "$REF" \
    > "$RLOG" 2>&1
  REF_RC=$?
  [ "$REF_RC" -eq 124 ] && echo "  step A TIMEOUT after ${REF_CAP}s"
  if [ "$REF_RC" -eq 0 ] && [ ! -s "$REF" ]; then REF_RC=10; fi
  echo "  step A reference: rc=$REF_RC  $( [ -s "$REF" ] && echo "npz $(du -h "$REF" | cut -f1)" || echo 'npz MISSING' )"

  VERDICT="NO VERDICT"; TIMING="NO TIMING"; SCORE_RC=0
  if [ "$REF_RC" -ne 0 ]; then
    SCORE_RC=$REF_RC
    echo "  reference unusable -- SKIPPING this deck's scoring (tail of $RLOG):"
    tail -n 12 "$RLOG" | sed 's/^/    /'
  else
    timeout --kill-after=60 "$SCORE_CAP" \
      srun -N "$SNODES" -n "$RANKS" --ntasks-per-node=2 --gpus-per-task=1 --kill-on-bad-exit=0 -c "$CPUS" \
        "$PY" "$GATE" --n "$N" --km "$KM" --kt "$KT" --pad "$PAD" \
        --n-split "$NSPLIT" --dt "$DT" --steps "$STEPS" \
        --distributed --ref-npz "$REF" --gpu-max-rel-peak "${GPU_MAX_REL_PEAK:-1e-12}" --timing 5 \
      > "$SLOG" 2>&1
    SCORE_RC=$?
    [ "$SCORE_RC" -eq 124 ] && echo "  step B TIMEOUT after ${SCORE_CAP}s"
    VERDICT=$(grep -m1 -F '[m6] VERDICT' "$SLOG" || true)
    TIMING=$(grep -m1 '^\[m6\] TIMING .*p50 ' "$SLOG" || true)
    if [ -z "$VERDICT" ]; then
      VERDICT="NO VERDICT"
      echo "  no VERDICT line -- this deck FAILED (tail of $SLOG):"; tail -n 12 "$SLOG" | sed 's/^/    /'
      [ "$SCORE_RC" -eq 0 ] && SCORE_RC=11
    fi
    if [ -z "$TIMING" ]; then
      TIMING="NO TIMING"
      echo "  no usable TIMING line -- this deck FAILED"
      [ "$SCORE_RC" -eq 0 ] && SCORE_RC=12
    fi
    echo "  step B scoring ($RANKS ranks / $SNODES nodes): rc=$SCORE_RC"
  fi
  [ "$SCORE_RC" -eq 0 ] || FAIL=1
  SUMMARY+=("DECK C${N} km=${KM} kt=${KT} ref_rc=${REF_RC} score_rc=${SCORE_RC} ${VERDICT} ${TIMING}")
done

echo ""
echo "==== DECK LADDER SUMMARY ===="
for s in "${SUMMARY[@]}"; do echo "$s"; done
echo "outdir=$OUT"
# the marker says what happened (codex 2026-09-16: a bare DONE printed even
# when every deck was skipped, and marker-only monitoring would read that as
# success)
if [ "$FAIL" -eq 0 ]; then
  echo "GPU_DECK_LADDER_DONE ok"
  exit 0
fi
echo "GPU_DECK_LADDER_DONE FAILED"
exit 1
