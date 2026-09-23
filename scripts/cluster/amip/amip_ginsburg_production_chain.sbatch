#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=amip_gb_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:59:00
#SBATCH --requeue
#SBATCH --output=results/amip/prod_1979/slurm_%j.out
# FOLLOWS PRODUCTION, DELIBERATELY (2026-09-23): the production deck became
# the CAM6 suite on that date and this script keeps naming it, because its
# job is to run whatever the authoritative production configuration IS, not
# to reproduce one past run.  Runs started before that date continued from a
# checkpoint must name config/amip/amip_sundqvist_l36.yaml instead.
# DEVICE COUNT raised 2 -> 4 on 2026-09-23, standing user directive: a
# production AMIP arm runs on at least four GPUs.  UNVERIFIED ON THIS MACHINE:
# the four-device evidence run was done on Levante, and this Ginsburg chain
# does not launch run_amip under srun, so it relies on whatever multi-device
# path the single process takes there.  Before the next Ginsburg production
# run, repeat the one-day four-GPU smoke on that machine, and if it needs the
# rank-per-GPU launch the Levante chain uses, port that too.
# =============================================================================
# Self-chaining PRODUCTION AMIP run on Ginsburg (Columbia) — authoritative C48/L40
# config (config/amip/amip_production.yaml) with the REAL staged inputs (ERA5 IC +
# ETOPO land + observed input4MIPs SST/SIC), validated by the short run.
#
#   mkdir -p results/amip/prod_1979           # SLURM opens --output BEFORE the body runs
#   sbatch scripts/cluster/amip/amip_ginsburg_production_chain.sbatch
#   # longer / different target / dir:
#   OUTDIR=$PWD/results/amip/prod_1979 TARGET_DAYS=3650 \
#     sbatch --export=ALL,OUTDIR=...,TARGET_DAYS=... scripts/cluster/amip/amip_ginsburg_production_chain.sbatch
#
# Each link runs until run_amip's --max-wallclock-seconds budget is nearly spent,
# at which point ModelDriver._maybe_wallclock_exit checkpoints + exits cleanly; the
# link then re-submits ITSELF, and the next link --restart-from the latest checkpoint.
# The JAX persistent compile cache (in OUTDIR) amortizes the ~16 min rrtmg JIT across
# links. Stops chaining when TARGET_DAYS is reached, the run fails, or a link makes no
# progress (infinite-loop guard).  Cancel the whole chain with `scancel` of the running
# link (subsequent links are not yet queued — each submits the next only on success).
#
# If you change --time above, set MAX_WALL to ~ (wall_seconds - 30 min) to match.
# =============================================================================
set -uo pipefail

REPO="${LEGOESM_REPO:-/burg-archive/glab/users/ac5006/legoESM}"
cd "$REPO" || { echo "cannot cd REPO=$REPO"; exit 2; }

: "${OUTDIR:=$REPO/results/amip/prod_1979}"
: "${TARGET_DAYS:=365}"
: "${CHECKPOINT_DAYS:=10}"       # checkpoint cadence (a killed link loses <= this many days)
: "${MAX_WALL:=41400}"           # ~11.5 h; driver checkpoints+exits before the 11:59 wall
mkdir -p "$OUTDIR"

# Real-data machine paths -> exports PY + AMIP_PATH_FLAGS (IC/ETOPO/SST/SIC/forcing).
source "$REPO/config/amip/amip_production.ginsburg.sh"

# Persistent rrtmg/JIT cache so only the FIRST link pays the ~16 min compile.
export JAX_COMPILATION_CACHE_DIR="$OUTDIR/jax_cache"; mkdir -p "$JAX_COMPILATION_CACHE_DIR"

_ckpt_day() {  # echo the integer day of the latest checkpoint in OUTDIR, or 0
  local f; f=$(ls -1 "$OUTDIR"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1)
  [ -n "$f" ] && basename "$f" | sed -E 's/.*checkpoint_day_0*([0-9]+)\.npz/\1/' || echo 0
}

cur_day=$(_ckpt_day)
latest=$(ls -1 "$OUTDIR"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1)
echo "=== HOST $(hostname) $(date) ==="
echo "[chain] OUTDIR=$OUTDIR cur_day=$cur_day TARGET_DAYS=$TARGET_DAYS latest=${latest:-none}"

if [ "${cur_day:-0}" -ge "$TARGET_DAYS" ]; then
  echo "[chain] target already reached ($cur_day >= $TARGET_DAYS) — nothing to do."; exit 0
fi

RESTART=()
[ -n "$latest" ] && RESTART=(--restart-from "$latest")

# ONE RANK PER GPU, matching the Levante chain.  Requesting four GPUs and
# then launching a SINGLE process would leave three idle and bill for them,
# which is worse than the one-GPU run this replaces (GLM review).  So the
# launch is changed with the request, not after it.  UNVERIFIED ON THIS
# MACHINE: the four-device evidence run was done on Levante.  If this fails
# here it fails LOUDLY on the first link rather than wasting a chain, and the
# fix is to repeat the one-day smoke on Ginsburg and adjust the srun flags.
: "${AMIP_N_GPUS:=4}"
if (( AMIP_N_GPUS < 4 )); then
  echo "[chain] REFUSING: AMIP_N_GPUS=${AMIP_N_GPUS} is below the floor of 4" >&2
  echo "[chain] (production AMIP arms run on at least four GPUs, 2026-09-23)." >&2
  exit 64
fi
# SLURM hands srun two conflicting cpus-per-task values on this cluster
# (SLURM_CPUS_PER_TASK from the allocation vs SLURM_TRES_PER_TASK from the
# --cpus-per-task request) and srun refuses to start: "cpus-per-task set by two
# different environment variables".  Measured, job 27647178 died in 12 s on it.
# Clear the inherited one and state the value on the srun line instead.
unset SLURM_CPUS_PER_TASK
_GPU_MAP=$(seq -s, 0 $((AMIP_N_GPUS - 1)))
JAX_ENABLE_X64=1 srun --ntasks="${AMIP_N_GPUS}" --ntasks-per-node="${AMIP_N_GPUS}" \
  --gpu-bind="map_gpu:${_GPU_MAP}" --cpus-per-task=8 --kill-on-bad-exit=1 \
  "$PY" -u scripts/run/run_amip.py \
  --config config/amip/amip_production.yaml \
  --distributed \
  "${AMIP_PATH_FLAGS[@]}" \
  --days "$TARGET_DAYS" \
  --checkpoint-days "$CHECKPOINT_DAYS" \
  --max-wallclock-seconds "$MAX_WALL" \
  "${RESTART[@]}" \
  --output "$OUTDIR"
rc=$?
new_day=$(_ckpt_day)
echo "[chain] run_amip rc=$rc  cur_day=$cur_day -> new_day=$new_day  $(date)"

if [ "$rc" -ne 0 ]; then
  echo "[chain] run FAILED (rc=$rc) — NOT chaining; inspect $OUTDIR/slurm_*.out."; exit "$rc"
fi
if [ "${new_day:-0}" -ge "$TARGET_DAYS" ]; then
  echo "[chain] COMPLETE: reached $new_day / $TARGET_DAYS days."; exit 0
fi
if [ "${new_day:-0}" -le "${cur_day:-0}" ]; then
  echo "[chain] NO PROGRESS ($cur_day -> $new_day) — NOT chaining (avoid an infinite loop)."; exit 1
fi

echo "[chain] $new_day/$TARGET_DAYS days done; submitting the next link."
sbatch --export="ALL,OUTDIR=$OUTDIR,TARGET_DAYS=$TARGET_DAYS,CHECKPOINT_DAYS=$CHECKPOINT_DAYS,MAX_WALL=$MAX_WALL" "$0"
