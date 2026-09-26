#!/bin/bash
#SBATCH --job-name=prof0_6
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/prof0_6_%j.out
# Where do the two profiles part company in the first day? Box-mean T/S at
# matched UTC hours, each model's change since NEMO's first hour.
# PRE-REGISTERED: (a) at hour 6 our 9-13 m band is already warmer above /
# cooler below than NEMO's by > 0.05 K -> a first-hours process (solar
# deposition or the afternoon mixing) sets the extra stratification before
# the first night; (b) hour-6 profiles agree within 0.05 K and diverge only
# after the first night -> the nocturnal entrainment itself is short.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
git -C $ROOT rev-parse --short HEAD | grep -q c51f44b2e || { echo "worktree HEAD moved"; exit 2; }
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_evdocc_d5
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_HOURLY
$PY -u scripts/validate/ocean_fidelity/equatorial_box_profile_tendency_vs_nemo.py --nemo-dir $N \
  --pairs 0:$R/snapshot_day0000.250.npz 1:$R/snapshot_day0000.250.npz 2:$R/snapshot_day0000.250.npz 3:$R/snapshot_day0000.250.npz 4:$R/snapshot_day0000.250.npz 5:$R/snapshot_day0000.250.npz \
echo "=== card exit: $? ==="
