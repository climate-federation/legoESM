#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=kn2day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/kn2day_%j.out
#
# IS THE SURFACE-K STARVATION A STRATIFICATION EFFECT OR A TKE EFFECT?
# The 5-day-mean K found surface avt 4x too small.  The N2 comparison is
# CONTAMINATED by a day-30-only over-stratification transient (ratio 0.58 ->
# 2.48 across days 26-30), so it cannot be the explanation for the mean K,
# which averages days 26-28 where our N2 was NORMAL-to-LOW (ratio 0.58-0.90).
# Low N2 should give MORE mixing, not less -- so if K is STILL starved on those
# days, the starvation is NOT stratification; it is the TKE magnitude (surface
# wind->TKE Dirichlet BC, production, or dissipation).
#
# Measure our avm/avt vs NEMO on day 28 (N2 ratio ~0.90, matched) and day 26
# (N2 ratio ~0.58, we are LESS stratified than NEMO).  Reads existing daily
# snapshots, runs no model.
#   K still 4x small where N2 <= NEMO -> TKE-magnitude defect (the lever)
#   K near NEMO where N2 matched       -> the day-30 mean was dragged by the
#                                         transient; re-open stratification
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
GW_W=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
D=$ROOT/results/omip_nemo/nemolev_trp_kmean_d30
for day in 0026 0028; do
  SNAP=$D/snapshot_day${day}.npz
  [ -f "$SNAP" ] || { echo "day $day MISSING"; continue; }
  echo "########## our day $day vs NEMO rec 5 ##########"
  $PY -u scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
    --nemo-gridw "$GW_W" --rec 5 --legoesm-snapshot "$SNAP" 2>&1 \
    | grep -E "surface 5-65|entrainment 65-105" | head -4
done
echo DONE
