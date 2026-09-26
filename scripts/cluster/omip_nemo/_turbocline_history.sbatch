#!/bin/bash
#SBATCH --job-name=tc_hist
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tc_hist_%j.out
# Time history of the turbulent-layer depth (NEMO mldkz5 rule), 220-240E
# |lat|<=2: ours from the first 6-hourly snapshots (days 0.25-1, EVD arm ==
# control physics + EVD) and from the production d30 run at 00 UTC (15 local)
# days 5-30, vs NEMO mldkz5 records 0-5 (5-day window means). Also the K_M
# profile at each time (Prandtl block).
# PRE-REGISTERED: ours ~19 m already at day 0.25-0.5 while NEMO's record 0 is
# 36 m -> the shallow layer is the closure's IMMEDIATE response to the shared
# IC + first-day wind (look at TKE init / first-step sources); ours deepens
# from >30 m toward 19 m over days 1-5 -> a feedback (N2 -> Pr/length) that
# NEMO does not have.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
ND=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
for S in $R/trp_evdocc_d5/snapshot_day0000.250.npz $R/trp_evdocc_d5/snapshot_day0000.500.npz $R/trp_evdocc_d5/snapshot_day0000.750.npz $R/trp_evdocc_d5/snapshot_day0001.npz $R/trp_evdocc_d5/snapshot_day0002.npz; do
  echo "##### $(basename $S) rec 0"
  $PY -u scripts/validate/ocean_fidelity/equatorial_shear_ri_vs_nemo.py --snapshot $S --nemo-dir $ND --rec 0 --n-levels 22 2>&1 | grep -E "turbocline avt|quantiles|KM ours|^ *(2|8|10|12|14|16|18|20) " 
done
for D in 5 10 15 20 25 30; do
  REC=$(( D/5 - 1 )); S=$R/trp_statemean_d30/snapshot_day00$(printf %02d $D).npz
  echo "##### $(basename $S) rec $REC"
  $PY -u scripts/validate/ocean_fidelity/equatorial_shear_ri_vs_nemo.py --snapshot $S --nemo-dir $ND --rec $REC --n-levels 22 2>&1 | grep -E "turbocline avt|quantiles|KM ours|^ *(2|8|10|12|14|16|18|20) "
done
echo "=== card exit: $? ==="
