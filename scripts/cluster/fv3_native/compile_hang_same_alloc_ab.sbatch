#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=hangab
#SBATCH --time=01:00:00
#SBATCH --nodes=3 --ntasks=6 --ntasks-per-node=2 --cpus-per-task=8 --mem=64G --gres=gpu:2
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/hang_ab_%j.log
# Every earlier arm landed on a DIFFERENT node set, and the same three nodes
# (g094-096) produced both a pass and a hang, so no earlier comparison is
# attributable.  This runs both arms back to back inside ONE allocation:
#   A = stock environment (the configuration that hung in production)
#   B = per-rank node-local Triton + XLA autotune caches
# Each arm is capped, so a hang costs minutes and still reports.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_codex_m8
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/hangab_${SLURM_JOB_ID}
mkdir -p "$OUT"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false LEGOESM_JIT_CACHE_DIR=""
export JAX_LOCAL_DEVICE_IDS=0 NCCL_SOCKET_IFNAME=ib0
cd "$REPO"
CAP=${CAP:-600}
# Job 9746442, same allocation, 4 runs: node-local caches hung exactly like
# stock, so that candidate is REFUTED.  Arm B is now env-selected.
export ARM_B_FLAGS="${ARM_B_FLAGS:---xla_gpu_autotune_level=0}"
echo "host $(hostname)  nodes $SLURM_JOB_NODELIST  cap ${CAP}s per arm  armB=$ARM_B_FLAGS"
LADDER="scripts/validate/fv3_native/compile_rank_ladder.py --backend gpu --devices 6 --distributed --n 24 --km 5 --pad 5 --n-split 1"
for REP in 1 2; do
  for ARM in A B; do
    T0=$SECONDS
    if [ "$ARM" = A ]; then
      timeout $CAP srun --gpus-per-task=1 --kill-on-bad-exit=0 \
        --output="$OUT/rep${REP}_${ARM}_rank_%t.log" --error="$OUT/rep${REP}_${ARM}_rank_%t.log" \
        $PY $LADDER
    else
      timeout $CAP srun --gpus-per-task=1 --kill-on-bad-exit=0 \
        --output="$OUT/rep${REP}_${ARM}_rank_%t.log" --error="$OUT/rep${REP}_${ARM}_rank_%t.log" \
        bash -c 'export XLA_FLAGS="${ARM_B_FLAGS}"; exec "$0" "$@"' $PY $LADDER
    fi
    RC=$?
    N=$(grep -l 'compile complete' "$OUT"/rep${REP}_${ARM}_rank_*.log 2>/dev/null | wc -l)
    echo "RESULT rep=$REP arm=$ARM rc=$RC ranks_compiled=$N/6 elapsed=$((SECONDS-T0))s"
  done
done
echo HANGAB_DONE
