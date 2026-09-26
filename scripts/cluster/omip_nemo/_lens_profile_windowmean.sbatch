#!/bin/bash
#SBATCH --job-name=lens_prof
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/lens_prof_%j.out
# Near-surface T profile, nino3-ish band 220-240E |lat|<=2, per level and
# cumulative excess heat vs NEMO, on the WINDOW MEAN (day 30 = NEMO record 5,
# early = day 5 = record 0) and, for reference, the instantaneous snapshot.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
A=$R/trp_statemean_d30
echo "##### WINDOW MEAN"
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py --surface-profile --use-mean-fields \
   --snapshot $A/snapshot_day0030.npz --label tripole --nemo-gridt $N --nemo-rec 5 \
   --snapshot-early $A/snapshot_day0005.npz --nemo-rec-early 0 2>&1 | grep -vi warning
echo "##### INSTANTANEOUS (same snapshots)"
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py --surface-profile \
   --snapshot $A/snapshot_day0030.npz --label tripole --nemo-gridt $N --nemo-rec 5 \
   --snapshot-early $A/snapshot_day0005.npz --nemo-rec-early 0 2>&1 | grep -vi warning
echo "=== card exit: $? ==="
