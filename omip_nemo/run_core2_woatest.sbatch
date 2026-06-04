#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_woatst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/woatst_%j.out

# Parametric WOA-init stability smoke. $1 = extra runner args (PGF / viscosity
# overrides), word-split intentionally. Goal: find a config that survives the
# realistic-IC cold-start (rest-state stable, WOA-init blows up by day 1).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
EXTRA="$1"
OUTD="results/omip_nemo/woatest_${SLURM_JOB_ID}"
echo "=== HOST ==="; hostname; date; echo "EXTRA_ARGS: $EXTRA"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_omip_core2.py \
  --smoke --dt 600 --woa-init --diag-every-days 1 $EXTRA \
  --output "$OUTD" 2>&1 | tail -40
echo "woatst_rc=${PIPESTATUS[0]}"
echo "=== DONE ==="
