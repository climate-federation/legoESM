#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_fc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fc_%j.out

# Test the in-step forcing integration (model.step surface_forcing) vs the old
# operator-split applicator. $1=grid $2=extra. Rest-state, ~36 days, watch
# max|u|/max|v|: old split run hit 5-10 m/s by day 30 then blew up. If the new
# path keeps velocities physical (~1-2 m/s) -> the split was the runaway cause.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
GRID="${1:-tripole}"; EXTRA="$2"; YEARS="${3:-0.1}"
echo "=== HOST ==="; hostname; date; echo "grid=$GRID years=$YEARS extra=$EXTRA"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_omip_core2.py \
  --grid "$GRID" --years "$YEARS" --dt 600 --diag-every-days 10 $EXTRA \
  --output results/omip_nemo/test_newforcing_${GRID} 2>&1 | tail -55
echo "fc_rc=${PIPESTATUS[0]}"
echo "=== DONE ==="
