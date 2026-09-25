#!/bin/bash
#SBATCH --job-name=sstphase
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/sstphase_%j.out
# Phase dependence of the equatorial SST bias: the four 6-hourly snapshots of
# day 4-5 of trp_phase4_d5 (00/06/12/18 UTC) plus the pinned control's day-5
# 00 UTC snapshot, all against NEMO record 0 (days 1-5 mean). Pre-registration
# in _trp_phase4_d5.sbatch.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
$PY -u scripts/validate/ocean_fidelity/sst_phase_vs_nemo.py --nemo-gridt $N --rec 0 --snapshots \
  $R/trp_phase4_d5/snapshot_day0004.npz $R/trp_phase4_d5/snapshot_day0004.250.npz \
  $R/trp_phase4_d5/snapshot_day0004.500.npz $R/trp_phase4_d5/snapshot_day0004.750.npz \
  $R/trp_phase4_d5/snapshot_final.npz $R/trp_pinned_d30/snapshot_day0005.npz \
  $R/trp_phase4_d5/snapshot_day0001.npz $R/trp_phase4_d5/snapshot_day0001.500.npz
echo "=== card exit: $? ==="
