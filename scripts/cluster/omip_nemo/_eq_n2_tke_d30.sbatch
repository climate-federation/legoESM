#!/bin/bash
#SBATCH --job-name=eqn2tke
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqn2tke_%j.out
#
# Our equatorial turbulence dies below ~12 m. K ~ C_k * l * sqrt(e), so three
# things can do it: too little energy, too much stratification, or too short a
# length. This measures stratification on BOTH sides through the model's own
# NEMO bn2 routine -- one instrument, not two -- and prints our TKE.
#
# PRE-REGISTERED, 15-65 m band (the one with little diurnal signal):
#   N2 ratio ~1  -> stratification exonerated; the deficit is energy or length.
#   N2 ratio >>1 -> our own warm surface is restratifying and shutting the
#                   closure off: a FEEDBACK, and the bias is self-sustaining.
#   N2 ratio <<1 -> we are too WELL mixed, which would contradict the
#                   diffusivity measurement and means one of the two probes is
#                   wrong -- report as a conflict, do not pick a winner.
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
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
$PY scripts/validate/ocean_fidelity/equatorial_n2_vs_nemo.py \
    --snapshot $RUN/snapshot_day0030.npz \
    --nemo-gridt $GW/ORCA1_5d_20000101_20000331_grid_T.nc \
    --nemo-gridw $GW/ORCA1_5d_20000101_20000331_grid_W.nc \
    --nemo-day 30 \
    --mesh-mask $MAIN/data/grids/eORCA1.2_mesh_mask.nc \
    --out-json $MAIN/results/omip_nemo/eq_n2_tke_d30.json
echo "=== rc=$? ==="
