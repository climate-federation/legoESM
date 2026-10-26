#!/bin/bash
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --job-name=hs1029anat
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_anatomy/anat_%j.out
#
# #1029 -- anatomy of the burst (diagnosis only, no model change).
# (a) number: per-step wind max, theta-inversion count, thinnest layer, p_s
#     range up to the trip, seed 65537 (shortest-lived member, 10.95 d).
# (b) reading: CONTROL first -- trip step must equal the matrix interval-1
#     trip (4730); otherwise the probe is not the matrix model and nothing
#     below is read.  Then: if theta inversions over the mountain appear and
#     grow before the wind does -> static (convective) instability; if the
#     wind jumps >10x in one step with stability intact -> a numerical
#     singularity (read dp_min / p_s / T at that step).  Flat (h0=0) arm,
#     same seed, gives the inversion-count baseline of a healthy run.
# (c) cost: two runs <= 4800 steps, one GPU, << 1 GPU-hour.
set -uo pipefail
ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/wt_1029
OUT=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_anatomy/job_$SLURM_JOB_ID
mkdir -p $OUT; cd $ROOT
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | tr '\n' ':')"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
echo "sha=$(git -c safe.directory=$ROOT rev-parse --short HEAD) dirty=$(git -c safe.directory=$ROOT status --porcelain | wc -l) $(date -u)"
$PY -u scripts/cluster/dycore_1029/burst_anatomy_probe.py --seed 65537 --h0 2000 --out $OUT/topo --save-at 2000 3000 4000 4400
$PY -u scripts/cluster/dycore_1029/burst_anatomy_probe.py --seed 65537 --h0 0 --fine-from 999999 --max-steps 4800 --out $OUT/flat
echo DONE $(date -u)
