#!/bin/bash
#SBATCH --job-name=euc_mpas
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/euc_mpas_%j.out
# MPAS unified day-30 snapshot predates the runner's u_east: add it offline
# (same mesh builder + Perot reconstruction), then the same undercurrent
# instrument (NEMO records 5:6) as the tripole (9648737) and FESOM (9648974).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=$ROOT/results/omip_nemo
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
SNAP=$R/mpas_unified_180d/snapshot_day0030.npz
UG=$R/mpas_unified_180d/snapshot_day0030_ugeo.npz
[ -f "$UG" ] || $PY -u scripts/validate/ocean_fidelity/mpas_snapshot_add_u_east.py --snapshot $SNAP --level 7 2>&1 | grep -vE "Warning|warn" | tail -5
[ -f "$UG" ] || { echo "FATAL: no $UG"; exit 1; }
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$UG" --nemo-gridt $GW/ORCA1_5d_20000101_20000331_grid_T.nc --nemo-t-recs 5:6 \
  --label mpas_unified_d30 --nemo-ufile $GW/ORCA1_5d_20000101_20000331_grid_U.nc --nemo-w-recs 5:6 \
  2>&1 | grep -vE "Warning|warn|^\s+[a-zA-Z_]+ = np\." | tail -120
echo "=== rc=$?"
echo DONE
