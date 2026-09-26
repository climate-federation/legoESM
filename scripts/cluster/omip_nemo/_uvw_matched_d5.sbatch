#!/bin/bash
#SBATCH --job-name=uvw_match
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/uvw_match_%j.out
# Matched-statistic day-5 read of the production control with window-mean
# state AND velocities (trp_stmean_uvw_d5, one 5-day window) against NEMO
# record 0: (1) upwelling at 50 m from mass_flux_w_mean vs NEMO wo;
# (2) shear/Ri/N2 from u_mean/v_mean/T_mean_hw; (3) per-level heat excess
# above/below 16 m; (4) nino3 window-mean SST and lens (same-run control).
# PRE-REGISTERED (memory 2026-09-25): w(50 m) ratio ours/NEMO >= 1.5 confirms
# excess upwelling as a live term at day 5; <= 1.2 refutes it (the 6x was a
# snapshot artifact). Heat split: excess above 16 m with deficit below at
# day 5 -> vertical redistribution/advection; excess above with NO deficit
# below -> net surface heat input (solar deposition or flux).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
ND=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
N=$ND/ORCA1_5d_20000101_20000331_grid_T.nc
S=$R/trp_stmean_uvw_d5/snapshot_day0005.npz
[ -e $S ] || { echo "no $S"; exit 2; }
echo "##### (1) upwelling, window means"
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py --legoesm-snapshot $S --use-mean-fields \
   --nemo-gridt $N --nemo-t-recs 0:1 --nemo-wfile $ND/ORCA1_5d_20000101_20000331_grid_W.nc --nemo-w-var wo --nemo-w-recs 0:1 --label tripole 2>&1 | grep -vi "warning\|^\s*$"
echo "##### (2) shear / Ri / N2, window means"
$PY -u scripts/validate/ocean_fidelity/equatorial_shear_ri_vs_nemo.py --snapshot $S --use-mean-fields --nemo-dir $ND --rec 0 2>&1 | grep -vi "warning\|^\s*$"
echo "##### (3) per-level heat excess, window means"
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py --surface-profile --use-mean-fields \
   --snapshot $S --label tripole --nemo-gridt $N --nemo-rec 0 2>&1 | grep -vi warning
echo "##### (4) nino3 window mean + lens"
$PY -u scripts/validate/ocean_fidelity/sst_phase_vs_nemo.py --nemo-gridt $N --rec 0 --mean --use-mean-fields --snapshots $S 2>&1 | grep -vi warning
echo "=== card exit: $? ==="
