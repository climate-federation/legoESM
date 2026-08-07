#!/bin/bash
#SBATCH --job-name=era5_ref_stage
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/b/b309178/era5_stage/logs/stage_%j.out
#SBATCH --error=/scratch/b/b309178/era5_stage/logs/stage_%j.out
#
# Stage ERA5 monthly `wap` (omega) and `psl` reference variables for the AMIP
# pattern scorer.  Runs on a COMPUTE node: the merged wap file is ~34 GB and
# the shared login node has ~251 GB across all users.
#
# Memory sizing (not a guess):
#   * CDO streams one 2-D field at a time; the bilinear remap weights for
#     1038240 target points are ~50 MB.
#   * verification opens with chunks={"time": 1} => 19*721*1440*4 = 79 MB per
#     chunk, and the loaded results are single 721x1440 maps (~8 MB).
#   48 GB is ~2 orders of magnitude of headroom over the largest live array.
#
# Disk: ~34 GB (wap) + ~1.8 GB (psl) final, plus an equal-sized temporary
# per-year staging area that is removed on exit.  /scratch had ~15 TB free.

set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_ap
REF_ROOT=/scratch/b/b309178/climateeval_data_ext
PY=/work/bd1083/b309178/mambaforge/envs/diffesm/bin/python

mkdir -p /scratch/b/b309178/era5_stage/logs

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
echo "git=$(cd "$REPO" && git rev-parse --short HEAD)"
echo "ref_root=$REF_ROOT"

srun "$PY" "$REPO/scripts/data/prep_era5_reference_mon.py" \
    --var wap psl \
    --start-year 1979 --end-year 2014 \
    --ref-root "$REF_ROOT" \
    --verify

rc=$?
echo "exit=$rc end=$(date -Is)"
exit $rc
