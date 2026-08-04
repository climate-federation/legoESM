#!/bin/bash -l
#SBATCH --job-name=oc128
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=oc128.%j.log
# Ocean lat-lon at 128 GPUs (tracked production launcher). — possible only because of the #1370 fix (residency
# now 0.10 GB/device, ratio 1.00). 4 procs/node is fine here: the
# host-side build peak at LL2304 is ~8.5 GB state x intermediates per
# process; 512 GB node RAM covers 4x that comfortably now that only
# per-band shards reach the GPU.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
# Ocean lane recommendation (job 26692291): fused halo -4.9%; the
# overlap flags showed no benefit here, so they stay OFF.
export LEGOESM_LATLON_SPMD_FUSED_HALO=1
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/oc128_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
srun --ntasks=128 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
     --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
    --multicontroller --n-devices 128 --mode strong \
    --n-lat 2304 --n-lon 4608 --nlev 20 --dt 150 \
    --baro-solver explicit_substep --wide-halo \
    --steps 18 --warmup 3 \
    --out "$OUTDIR/LL2304_np128.jsonl"
rc=$?
[ -f "$OUTDIR/LL2304_np128.jsonl" ] && "$PY" -c "
import json; d=json.loads(open('$OUTDIR/LL2304_np128.jsonl').readline())
print(f\"LL2304 @128: {d['steady_median_ms']:.2f} ms {d.get('mcells_per_s',0)/1000:.1f} GC/s\")"
echo "DONE rc=$rc"; exit $rc
