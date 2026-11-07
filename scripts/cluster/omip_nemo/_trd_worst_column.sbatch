#!/bin/bash
#SBATCH --job-name=trdworst
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trdworst_%j.out
#
# Worst-column dump for the Stage-B EVD-salt contradiction (ours 46x NEMO's
# strd_zdf on the same state while T is 5x SMALLER).  Offline, CPU.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TRD=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD

$PY -u scripts/validate/ocean_fidelity/tendency_stageb_worst_column.py \
  --tfile $TRD/keep_run3_1h_T.nc \
  --ufile $TRD/keep_run3_1h_U.nc \
  --vfile $TRD/keep_run3_1h_V.nc \
  --rec 1
rc=$?
echo "=== rc=$rc ==="
exit $rc
