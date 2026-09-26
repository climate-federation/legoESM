#!/bin/bash
#SBATCH --job-name=upwell
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/upwell_%j.out
# Is the cold deficit at 20-60 m ADVECTIVE? Equatorial upwelling ours (mass_flux_w,
# 00 UTC snapshot) vs NEMO wo (5-day mean), records matched (day 30 = rec 5,
# day 5 = rec 0). The 2026-09-13 note "ours 1.25-1.29x the oracle's" was read on
# a 00Z snapshot pair; re-read here with the window-matched oracle record.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
ND=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
for D in "day0030 5:6" "day0005 0:1"; do set -- $D
  echo "##### $1 (NEMO records $2)"
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py --legoesm-snapshot $R/trp_statemean_d30/snapshot_$1.npz \
     --nemo-gridt $ND/ORCA1_5d_20000101_20000331_grid_T.nc --nemo-t-recs $2 \
     --nemo-wfile $ND/ORCA1_5d_20000101_20000331_grid_W.nc --nemo-w-var wo --nemo-w-recs $2 --label tripole 2>&1 | grep -vi "warning\|^\s*$"
done
echo "=== card exit: $? ==="
