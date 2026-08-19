#!/bin/bash -l
#SBATCH --job-name=mpas_census
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=mpas_hlo_census.%j.log
# ===========================================================================
# How many collectives does the MPAS step ACTUALLY contain at 64 GPUs?
#
# Nobody knows. Every MPAS GPU row ever written recorded
# `hlo_collectives: null` — not because the census was unsupported, but
# because it re-lowers the step, the step closes over sharded schedule
# arrays, and multicontroller forbids that. The exception was swallowed.
# The nsys route that was used instead turned out to omit
# ncclDevKernel_SendRecv entirely in a subset of runs, with no error.
#
# So this lane's headline ratio (measured/bound 3.16-4.47x) rests on a
# collective count that was inferred, never read off the executable.
#
# THE METHOD. XLA_FLAGS=--xla_dump_to makes the compiler write the module it
# emitted, during the real run, with the real flags and device count. No
# re-lowering, no sampling. census_xla_dump.py then counts the collectives in
# the post-optimization HLO. Validated on two known answers already: it
# reproduces a hand grep exactly on an allgather run, and reports 3
# collective-permutes where the schedule independently predicts 1 round/fill
# x 3 RK fills.
#
# (a) NUMBER: collective-permute / all-reduce / all-gather counts in the
#     compiled s9@64 step.
# (b) WHAT IT DISCRIMINATES. The offline schedule says 11 rounds per halo
#     fill at reorder_target=128, and rank 0 participates in 8 of them; the
#     nsys trace showed 24 collectives per step for rank 0 (8 x 3 fills). If
#     the census lands near 33 (11 x 3), XLA is NOT combining on this lane and
#     the round count is the honest cost. If it lands far below, GPU
#     collective combining is already doing what a fused-exchange rewrite
#     would do, and that lever is dead before it is built — which is exactly
#     how the cube lever was just priced out.
# (c) WHY NOT CHEAPER: a CPU proxy gives the WRONG count. Measured on the cube
#     lane the same day: 396 on a 24-device CPU proxy against 123 in the GPU
#     executable, a 3.2x combining factor. Structure transfers; counts do not.
#
# Only rank 0 dumps. Every process would otherwise write hundreds of modules
# and the useful one is identical across ranks of the same shape.
#
# SUBMIT:  sbatch scripts/cluster/scaling_levante/mpas_hlo_census.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_census_j${SLURM_JOB_ID}}"
DUMP="$OUTDIR/dump_rank0"
mkdir -p "$DUMP"; echo "outdir=$OUTDIR"

# Pin the collective-affecting flags. An inherited LEGOESM_XLA_OVERLAP would
# enable pipelined p2p and change the very count being read.
export LEGOESM_XLA_OVERLAP=0

cat > "$OUTDIR/rank_wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
# Dump only from rank 0: every process compiles the same-shaped module, and 64
# processes x hundreds of modules is a lot of disk for one answer.
set -uo pipefail
if [ "${SLURM_PROCID:-1}" = "0" ]; then
  export XLA_FLAGS="--xla_dump_to=${DUMP_DIR}${XLA_FLAGS:+ $XLA_FLAGS}"
fi
exec "$PY_BIN" "$@"
WRAP
chmod +x "$OUTDIR/rank_wrapper.sh"
export DUMP_DIR="$DUMP" PY_BIN="$PY"

JAX_ENABLE_X64=0 srun --ntasks=64 --distribution=block:block \
  --cpu-bind=cores --kill-on-bad-exit=1 \
  "$OUTDIR/rank_wrapper.sh" scripts/bench/bench_mpas_spmd_scaling.py \
    --n-devices 64 --subdivision 9 --nlev 26 --steps 12 --warmup 3 \
    --lloyd 0 --partition-method sfc --reorder-for 128 --multicontroller \
    --out "$OUTDIR/s9np64.jsonl"
RC=$?
echo "[census] srun exit=$RC"

n_mod=$(ls "$DUMP" 2>/dev/null | grep -c "after_optimizations.txt" || true)
echo "[census] dumped optimized modules: $n_mod"
if [ "${n_mod:-0}" -eq 0 ]; then
  echo "[census] NO modules dumped — XLA_FLAGS did not reach rank 0"
  echo "CENSUS_NO_DUMP"; exit 4
fi

"$PY" scripts/bench/census_xla_dump.py --dump-dir "$DUMP" \
    --out "$OUTDIR/census.json"
RC2=$?
[ "$RC2" -eq 0 ] || { echo "CENSUS_FAILED"; exit "$RC2"; }
[ "$RC" -eq 0 ] || { echo "CENSUS_RUN_FAILED"; exit "$RC"; }
echo "CENSUS_DONE"
