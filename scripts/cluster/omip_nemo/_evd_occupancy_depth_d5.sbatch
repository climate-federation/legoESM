#!/bin/bash
#SBATCH --job-name=evd_depth
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/evd_depth_%j.out
# How DEEP does the nocturnal instability reach, ours vs NEMO, days 0-5?
# Ours: area fraction of 220-240E |lat|<=2 interfaces with N2 < threshold in
# each 6-hourly snapshot of the EVD occupancy arm (four clock phases: 03, 09,
# 15, 21 local), then the 4-phase mean per interface (= the daily mean).
# NEMO: (mean avt - median avt)/rn_evd per interface from record 0, i.e. the
# window fraction EVD fired if the median is the TKE-only value.
# PRE-REGISTERED: NEMO's implied fraction falls to <0.01 between 9 and 13 m at
# day 5 (avt 0.385 -> 1.9e-3); if ours falls to <0.01 SHALLOWER than 7 m the
# defect is the depth of the night convection (heat budget / stratification
# just below the diurnal layer), not the EVD operator; if it reaches as deep
# as NEMO's while the lens stays 1.7x, the EVD operator delivers no mixing at
# the interfaces it fires on (solve / placement) -> read the solve.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_evdocc_d5
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
for S in $(ls $R/snapshot_day000[1-4].500.npz | sort); do   # 12 UTC = 03 local at 230E
  echo "##### $(basename $S)"
  $PY -u scripts/validate/ocean_fidelity/equatorial_shear_ri_vs_nemo.py --snapshot $S --nemo-dir $N --rec 0 --n-levels 14 2>&1 | grep -vi warning
done
echo "=== card exit: $? ==="
# 2026-09-26 second pass (03 local only): at OUR post-step unstable interfaces,
# does the model's own diagnosed K_H exceed 10 (EVD engaged), and is the
# instability near-threshold (generated after the solve) or ~1e-5 (unmixed)?
# Note: the first pass compared post-step residual instability with NEMO's
# trigger-time fraction -- not the same quantity at 1-5 m, where the solve
# removes what the trigger saw. NEMO's (mean-median)/rn_evd column saturates
# to 0 wherever EVD fires >50% of the window (median = rn_evd); use avt/100.
