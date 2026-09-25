#!/bin/bash
#SBATCH --job-name=eucmerid
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eucmerid_%j.out
#
# DISCRIMINATOR round 2 for the 120W missing ascent (ours -5.0e-6 m/s vs NEMO
# +21.9e-6, job 9430332): is the ascent DISPLACED off-equator or ABSENT, and
# is the undercurrent under it intact?  Both blocks are native-to-native on
# the shared eORCA1 mesh, matched simulated days (recs 17:18 = days 86-90 for
# the day-90 snapshot; 5:6 = days 26-30 for day-30).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
EXP=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
GT=$EXP/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
GW=$EXP/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
GU=$EXP/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_U.nc
R=$ROOT/results/omip_nemo
echo "HEAD: $(git -C $ROOT log --oneline -1)"
for f in "$GT" "$GW" "$GU"; do [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }; done

echo "=== [1/2] day 90, recs 17:18 ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot $R/nemolev_trp_signedn2_d90/snapshot_day0090.npz \
  --nemo-gridt "$GT" --nemo-month 3 --label signedn2_d90 \
  --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 17:18 \
  --nemo-ufile "$GU" --meridional-lon 200,240 2>&1 | tail -70
echo "=== [1/2] rc=$?"

echo "=== [2/2] day 30, recs 5:6 ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot $R/nemolev_trp_signedn2_d30/snapshot_day0030.npz \
  --nemo-gridt "$GT" --nemo-month 1 --label signedn2_d30 \
  --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile "$GU" --meridional-lon 200,240 2>&1 | tail -70
echo "=== [2/2] rc=$?"
