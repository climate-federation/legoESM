#!/bin/bash
#SBATCH --job-name=score_evdph
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score_evdph_%j.out
# EVD phase arm (trp_phase4evd_d5) vs its control (trp_phase4_d5): every
# 6-hourly snapshot and the days-0.25-5 SAMPLE MEAN against NEMO record 0.
# PRE-REGISTERED (card _trp_phase4evd_d5): 4-phase-mean nino3 +0.27 -> <=+0.13
# and pre-dawn SST-T10 contrast +0.15 -> <=+0.05 confirms EVD as the lever;
# a move < 0.05 refutes it (on the daily mean, which the 00 UTC read could not).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
for D in trp_phase4evd_d5 trp_phase4_d5; do
  [ -e $R/$D/snapshot_day0005.npz ] || ln -s snapshot_final.npz $R/$D/snapshot_day0005.npz
  echo "##### $D"
  SNAPS=$(ls $R/$D/snapshot_day00*.npz | grep -vE "day0000\.npz" | sort)
  $PY -u scripts/validate/ocean_fidelity/sst_phase_vs_nemo.py --nemo-gridt $N --rec 0 --mean --snapshots $SNAPS
done
echo "=== card exit: $? ==="
