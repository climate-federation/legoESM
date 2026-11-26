#!/bin/bash
#SBATCH --job-name=varbudget
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/varbudget_%j.out
#
# WHICH OPERATOR DESTROYS NEMO'S TRACER VARIANCE?
#
# The census said the tracer DISTRIBUTION differs but is blind to which
# operator did it. chi_op = 2*integral(theta' * dtheta/dt|_op) dV is the
# discriminator: the most negative term NAMES the operator.
#
# This is NEMO's OWN side only -- the oracle's reference budget. Our
# operators come later, on a matched state via the Mode-A machinery.
#
# The daily file has the operators but NOT the state, so theta comes from the
# hourly file. The product of daily means is not the daily mean of products,
# and that error is MEASURED rather than assumed: the hourly file carries both
# votemper and ttrd_zdf, so chi_zdf is computed exactly hour-by-hour AND from
# daily means, and their ratio is an error bar on every daily-mean term. A
# ratio far from 1.0 means the daily-mean terms are not quotable.
set -euo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

TRD=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD
MESH=/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
for f in "$TRD/keep_run1_1d_trd.nc" "$TRD/keep_run1_1h_trd.nc" "$MESH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

$PY -u scripts/validate/ocean_fidelity/tracer_variance_budget_nemo.py \
  --daily-trd "$TRD/keep_run1_1d_trd.nc" \
  --hourly-trd "$TRD/keep_run1_1h_trd.nc" \
  --mesh-mask "$MESH" \
  --out-dir "$R/variance_budget_nemo" 2>&1 | grep -vE "Warning|warn" | tail -50
