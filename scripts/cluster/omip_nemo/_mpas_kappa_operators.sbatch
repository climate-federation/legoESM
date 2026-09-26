#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=kappaops
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/kappaops_%j.out
#
# The "then measure" half of the MPAS Treguier port: how much does the slope
# OPERATOR change the coefficient, on a real OMIP ocean state rather than an
# idealised one?  Offline from one saved restart — a free-running two-lane
# comparison is confounded by trajectory divergence within days.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
RESTART=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/mpas_orca1faithtint_d30/restart_leg2.npz

[ -f "$RESTART" ] || { echo "FATAL: no restart at $RESTART"; exit 1; }
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -u \
  scripts/validate/ocean_fidelity/mpas_treguier_kappa_operators.py \
  --restart "$RESTART" --level 7 --lloyd 20 \
  --map $ROOT/omip_nemo/kappa_jensen_gap.png
rc=$?
echo "DONE rc=$rc"
exit $rc
