#!/bin/bash
#SBATCH --job-name=tkenolc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tkenolc_%j.out
# NEMO with and without Langmuir: does the subsurface TKE maximum at 9-11 m
# survive ln_lc=F? Same ours snapshot (control EVD arm) in both calls; read the
# NEMO columns. PRE-REGISTERED: noLC still has e rising with depth to 9-11 m
# -> the feeder is not Langmuir (transport/production puzzle stands);
# noLC decays monotonically like ours -> LC shapes the profile but the
# nocturnal deepening is EVD/convective and lives in the T/S column.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
git -C $ROOT rev-parse --short HEAD | grep -q bb0771ff5 || { echo "worktree HEAD moved"; exit 2; }
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
B=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
for D in RUN_HOURLY RUN_HOURLY_NOLC; do for H in 12 16 36; do
  S=$R/trp_evdocc_d5/snapshot_day000$( [ $H = 12 ] && echo 0.500 || ( [ $H = 16 ] && echo 0.750 || echo 1.500 ) ).npz
  echo "##### $D step $H (ours column = $(basename $S), matched only for 12/36)"
  $PY -u scripts/validate/ocean_fidelity/equatorial_tke_length_vs_nemo_restart.py --restart-glob "$B/$D/ORCA1_000000${H}_restart_oce_*.nc" --snapshot $S --n-levels 16 2>&1 | grep -E "^\s+[0-9]+\s+[0-9.]+ \||surface|SystemExit|Error" | cut -c1-118
done; done
echo "=== card exit: $? ==="
