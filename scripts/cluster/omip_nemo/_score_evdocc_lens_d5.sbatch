#!/bin/bash
#SBATCH --job-name=sc_evdocc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/sc_evdocc_%j.out
# EVD occupancy arm (trp_evdocc_d5, 6-hourly windows) vs the production
# control (trp_statemean_d30 day-5 five-day window): days-0-5 WINDOW MEAN
# nino3 SST and SST-T(10) lens against NEMO record 0, both sides tiled by the
# probe onto [0,5]. The arm fired 0.44 of the day at interface 1-2 (NEMO
# implied 0.54), so the trigger is not the defect.
# PRE-REGISTERED (card _trp_evdocc_d5): lens moves < 0.05 K vs control ->
# delivered-K defect (fires, does not mix); lens closes to NEMO -> the earlier
# EVD null was the 00 UTC read and the sign test.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
D=trp_evdocc_d5
[ -e $R/$D/snapshot_day0005.npz ] || ln -s snapshot_final.npz $R/$D/snapshot_day0005.npz
echo "##### $D (window means, 6-h tiles)"
SNAPS=$(ls $R/$D/snapshot_day00*.npz | grep -vE "day0000\.npz" | sort)
$PY -u scripts/validate/ocean_fidelity/sst_phase_vs_nemo.py --nemo-gridt $N --rec 0 --mean --use-mean-fields --snapshots $SNAPS
echo "##### trp_statemean_d30 (control, one 5-day window)"
$PY -u scripts/validate/ocean_fidelity/sst_phase_vs_nemo.py --nemo-gridt $N --rec 0 --mean --use-mean-fields --snapshots $R/trp_statemean_d30/snapshot_day0005.npz
echo "##### $D scorecard (window means)"
$PY -u scripts/validate/compare_omip_nemo.py --legoesm-snapshot $R/$D/snapshot_final.npz --nemo-gridt $N --nemo-time-idx 0 \
    --output-dir $R/compare_evdocc_d5_mean --grid-label tripole --use-mean-fields 2>&1 | grep -vi warning
echo "=== card exit: $? ==="
