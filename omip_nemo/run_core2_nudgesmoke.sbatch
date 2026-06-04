#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_ndg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ndg_%j.out

# Smoke-test the NUDGE-from-rest spinup: start at the (stable) rest state and
# nudge T,S toward WOA over tau days. Builds realistic stratification gradually
# instead of the violent WOA cold-start. $1 = tau days (default 60).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TAU="${1:-60}"
echo "=== HOST ==="; hostname; date; echo "nudge tau=${TAU}d"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_omip_core2.py \
  --smoke --dt 600 --nudge-woa-tau-days "$TAU" --diag-every-days 1 \
  --output results/omip_nemo/legoesm_tripole_nudge_smoke 2>&1 | tail -40
echo "ndg_rc=${PIPESTATUS[0]}"
echo "=== DONE ==="
