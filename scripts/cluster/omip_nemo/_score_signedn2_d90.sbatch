#!/bin/bash
#SBATCH --job-name=sc90enso
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/sc90enso_%j.out
#
# Day-90 skill of the signed-TEOS-10 arm 9408833 vs NEMO month 3 (same
# protocol as every arm in the campaign: freeze clamp -1.8 C, --sss-faithful),
# PLUS the first run of the matched-window equatorial UPWELLING discriminator:
# NEMO's 5-day `wo` from RUN_GATEWAY (the matched cold-start ref) at the SAME
# simulated days as our snapshots, per the codex 9417318 recommendation --
# "compare the vertical velocity before touching the closure again".
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
EXP=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
GT=$EXP/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
GW=$EXP/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
R=$ROOT/results/omip_nemo
echo "HEAD: $(git -C $ROOT log --oneline -1)"
for f in "$GT" "$GW" "$R/nemolev_trp_signedn2_d90/snapshot_day0090.npz"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

echo "=== [1/3] day-90 skill vs NEMO m3 (ENSO boxes, maps, zonal) ==="
$PY scripts/validate/compare_omip_nemo.py \
  --legoesm-snapshot $R/nemolev_trp_signedn2_d90/snapshot_day0090.npz \
  --nemo-gridt "$GT" --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label signedn2_d90 \
  --output-dir $R/compare_signedn2_d90_m03_enso 2>&1 | tail -60
echo "=== [1/3] rc=$?"

echo "=== [2/3] Z20 + upwelling, day 90 vs RUN_GATEWAY wo recs 17:18 (days 86-90) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot $R/nemolev_trp_signedn2_d90/snapshot_day0090.npz \
  --nemo-gridt "$GT" --nemo-month 3 --label signedn2_d90 \
  --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 17:18 2>&1 | tail -40
echo "=== [2/3] rc=$?"

echo "=== [3/3] Z20 + upwelling, day 30 vs RUN_GATEWAY wo recs 5:6 (days 26-30) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot $R/nemolev_trp_signedn2_d30/snapshot_day0030.npz \
  --nemo-gridt "$GT" --nemo-month 1 --label signedn2_d30 \
  --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 5:6 2>&1 | tail -40
echo "=== [3/3] rc=$?"
