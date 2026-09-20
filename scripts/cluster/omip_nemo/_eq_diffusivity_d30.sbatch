#!/bin/bash
#SBATCH --job-name=eqdiff
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqdiff_%j.out
#
# GLM's top-ranked discriminator for the nino3 vertical dipole (warm above
# 10 m, cold at 10-50 m, column heat correct): the vertical HEAT diffusivity
# profile in the top 50 m, ours vs NEMO's published avt at the SAME record.
#
# Window is the PROBE'S OWN default (200-260E, +/-2 deg of the equator), not a
# window chosen for this question -- slightly west of the nino3 box the dipole
# was measured in. Record 5 = the 5-day mean ending day 30, matching the
# snapshot.
#
# Reads our side from K_H_diag/K_M_diag, written by --kprofile-snapshots, which
# the control run passed. It is never re-derived from TKE and a mixing length.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_fluxacc_wt
MAIN=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
RUN=$MAIN/results/omip_nemo/trp_orca1faithtint_d30
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
$PY scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
    --nemo-gridw $GW --rec 5 --depth-max 300 \
    --legoesm-snapshot $RUN/snapshot_day0030.npz
echo "=== rc=$? ==="
