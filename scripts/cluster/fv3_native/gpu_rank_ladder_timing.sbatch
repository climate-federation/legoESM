#!/bin/bash
#SBATCH --job-name=gpu_rank_ladder
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=12
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
# A scaling ladder across card models measures the cards, not the scaling, so the
# model is PINNED here. rtx8000 is the only model with enough nodes in `short` for
# 24 ranks (18 nodes; 12 survive the exclusion below, which is exactly what we need).
#SBATCH --constraint=rtx8000
#SBATCH --exclusive
# WORKAROUND, delete when the nodes are fixed: g[045-054],g[185-193] compile the
# sharded step and then hang in its first execution (pinned-node A/B, jobs
# 9777265 good vs 9777266 hung).
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/gpu_rank_ladder_%j.log
# Rank-count ladder for the FV3 window-SPMD step on GPU, inside ONE allocation:
#   row 1: kt=1 -> 6 ranks
#   row 2: kt=2 -> 24 ranks
# Both rows see the SAME nodes, so the comparison is not confounded by node
# assignment -- the confound that voided every earlier cross-allocation GPU
# comparison here. Written by GLM-5.2; reviewed by Claude and codex.
set -uo pipefail

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
WT=${WT:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
REF=${REF:-/burg-archive/glab/users/pg2328/fv3_duo_gaps/gpu_ref_c24_v2.npz}
GATE="$WT/scripts/validate/fv3_native/tiled_m6_model_gate.py"
cd "$WT" || exit 1
for f in "$PY" "$GATE" "$REF"; do
  [ -e "$f" ] || { echo "PREFLIGHT FAILED: missing $f"; exit 5; }
done

export PYTHONPATH="$WT/packages/core:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ocean:$WT/packages/ml:$WT/packages/tools:$WT/src"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# one GPU per task: the cgroup hides the node's other GPU but the driver still
# counts it, and jax then fails on CUDA:1 (job 9650417)
export JAX_LOCAL_DEVICE_IDS=0
export NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# autotuning deadlocks when several ranks compile; level 0 is the only setting
# that completes (job 9746516) and costs nothing (job 9776324)
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"

OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/gpu_rank_ladder_${SLURM_JOB_ID:-local}
mkdir -p "$OUT" || exit 1
N=${N:-24}; KM=${KM:-5}; PAD=${PAD:-5}; NSPLIT=${NSPLIT:-1}; TIMING=${TIMING:-10}
CAP=${CAP:-900}
echo "host $(hostname)  nodes ${SLURM_JOB_NODELIST:-?}"
echo "sha $(git -C "$WT" rev-parse HEAD)  dirty $(git -C "$WT" status --porcelain | wc -l)"
echo "XLA_FLAGS=$XLA_FLAGS"
echo "deck C$N km=$KM pad=$PAD n_split=$NSPLIT timing=$TIMING cap=${CAP}s  ref=$REF"

run_row () {   # kt ranks nodes
  local KT=$1 RANKS=$2 NODES=$3 LOG="$OUT/kt${1}_n${2}.log"
  echo "===== row kt=$KT ranks=$RANKS nodes=$NODES $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  # PLACEMENT IS PINNED (codex 2026-09-16): srun inherits the allocation's
  # node count, so six ranks could land one-per-node on twelve nodes while
  # the 24-rank row runs two-per-node -- different locality, and the ladder
  # would be measuring placement. Both rows are forced to two ranks per
  # node. -c is passed explicitly: this Slurm does not propagate the batch
  # --cpus-per-task into srun.
  timeout --kill-after=60 "$CAP" \
    srun --gpus-per-task=1 --kill-on-bad-exit=0 \
      -N "$NODES" -n "$RANKS" --ntasks-per-node=2 -c "${SLURM_CPUS_PER_TASK:-8}" \
      "$PY" "$GATE" --n "$N" --km "$KM" --kt "$KT" --pad "$PAD" \
      --n-split "$NSPLIT" --steps 1 --distributed --ref-npz "$REF" \
      --gpu-max-rel-peak "${GPU_MAX_REL_PEAK:-1e-12}" --timing "$TIMING" \
    > "$LOG" 2>&1
  local RC=$?
  # the gate's own line, and it must carry a p50 -- "[m6] TIMING REFUSED"
  # or no line at all is NOT a timing row (codex 2026-09-16)
  local T; T=$(grep -m1 '^\[m6\] TIMING .*p50 ' "$LOG")
  if [ -z "$T" ]; then
    echo "  NO USABLE TIMING LINE in $LOG -- row FAILED"
    [ "$RC" -eq 0 ] && RC=2
  fi
  # the cards the row actually ran on: a ladder across card types is not a
  # ladder (the allocation is pinned to rtx8000 above, so the card is
  # RECORDED here as well and compared between rows)
  local CARDS; CARDS=$(grep -ho 'visible_devices=\[[^]]*\]' "$LOG" | sort -u | tr '\n' ' ')
  local MODEL; MODEL=$(grep -m1 -ho 'RUNTIME .*' "$LOG" | head -1)
  ROW_RC=$RC; ROW_TIMING=${T:-NO TIMING}
  ROW_NODES=$(grep -m1 -ho 'nodes=[0-9?]*' "$LOG" | head -1)
  echo "rc=$RC  ${ROW_NODES:-nodes=?}  $(grep -m1 '^\[m6\] VERDICT' "$LOG" || echo 'no verdict')"
}

nvidia-smi --query-gpu=name --format=csv,noheader | sort -u > "$OUT/cards_batchnode.txt"
echo "batch-node card(s): $(tr '\n' ' ' < "$OUT/cards_batchnode.txt")"
srun -N 12 --ntasks-per-node=1 bash -c 'nvidia-smi --query-gpu=name --format=csv,noheader | head -1' 2>/dev/null | sort -u > "$OUT/cards_alloc.txt"
echo "allocation card model(s): $(tr '\n' ' ' < "$OUT/cards_alloc.txt")"
if [ "$(wc -l < "$OUT/cards_alloc.txt")" -gt 1 ]; then
  echo "REFUSED: the allocation straddles GPU models -- a rank ladder across"
  echo "card types measures the cards, not the scaling:"; cat "$OUT/cards_alloc.txt"
  exit 4
fi

run_row 1 6  3;  RC1=$ROW_RC;  T1=$ROW_TIMING
run_row 2 24 12; RC2=$ROW_RC;  T2=$ROW_TIMING

echo "===== LADDER summary ====="
echo "LADDER ranks=6 kt=1 rc=$RC1 $T1"
echo "LADDER ranks=24 kt=2 rc=$RC2 $T2"
echo GPU_RANK_LADDER_DONE
[ "$RC1" -eq 0 ] && [ "$RC2" -eq 0 ] || exit 1
exit 0
