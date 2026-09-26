#!/bin/bash
#SBATCH --job-name=sc_mxl4
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/sc_mxl4_%j.out
# Mixing-length arm (trp_mxl4_d5: tke_mxl_choice 4 == ORCA1 nn_mxl=2) vs the
# production control, same probes, same NEMO record 0:
#  (1) turbocline by clock phase, days 1-4.75 (NEMO mldkz5 35.7 m; control 19.3 m);
#  (2) days-0-5 window-mean nino3 SST and SST-T(10) lens (control +0.103 / 0.178);
#  (3) Prandtl split + patchiness at day 3, four phases.
# PRE-REGISTERED (card _trp_mxl4_d5): turbocline toward 36 m AND lens toward
# 0.108 -> mixing-length row is the lever; turbocline within +3 m of 19.3 ->
# refuted. Reading of zdftke.F90:658-700 predicts nn_mxl=2 has the SHORTER
# lengths, i.e. a SHALLOWER layer (faithful-but-worse): that outcome settles
# the harmonization row without closing the gap.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
A=$R/trp_mxl4_d5
ND=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
N=$ND/ORCA1_5d_20000101_20000331_grid_T.nc
[ -e $A/snapshot_day0005.npz ] || ln -s snapshot_final.npz $A/snapshot_day0005.npz
echo "##### (1) turbocline by phase"
for S in $(ls $A/snapshot_day000[1-4]*.npz | sort); do
  echo "##### $(basename $S)"
  $PY -u scripts/validate/ocean_fidelity/equatorial_shear_ri_vs_nemo.py --snapshot $S --nemo-dir $ND --rec 0 --n-levels 3 2>&1 | grep -E "turbocline avt"
done
echo "##### (2) window-mean nino3 + lens, arm then control"
SNAPS=$(ls $A/snapshot_day00*.npz | grep -vE "day0000\.npz" | sort)
$PY -u scripts/validate/ocean_fidelity/sst_phase_vs_nemo.py --nemo-gridt $N --rec 0 --mean --use-mean-fields --snapshots $SNAPS 2>&1 | grep -E "WINDOW MEAN|CONTRAST"
$PY -u scripts/validate/ocean_fidelity/sst_phase_vs_nemo.py --nemo-gridt $N --rec 0 --mean --use-mean-fields --snapshots $R/trp_stmean_uvw_d5/snapshot_day0005.npz 2>&1 | grep -E "WINDOW MEAN"
echo "##### (3) Prandtl split + patchiness, day 3 phases"
for S in $(ls $A/snapshot_day0003*.npz | sort); do
  echo "##### $(basename $S)"
  $PY -u scripts/validate/ocean_fidelity/equatorial_shear_ri_vs_nemo.py --snapshot $S --nemo-dir $ND --rec 0 --n-levels 22 2>&1 | grep -vi warning
done
echo "=== card exit: $? ==="
