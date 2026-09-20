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
#
# SHEAR, added after the N2 answer came back 1.50 (too small to carry a 200x
# coefficient deficit, so the energy is what is missing). PRE-REGISTERED on the
# same 15-65 m band:
#   shear^2 ratio ~0.25 -> the collapse is SELF-CONSISTENT: half the current,
#                          a quarter the production, and the coefficient and
#                          the current hold each other down.
#   shear^2 ratio ~1    -> production is AVAILABLE and the energy still is not
#                          there, so the defect is INSIDE the TKE equation
#                          (dissipation, surface source, or its transport).
#   shear^2 ratio >>1   -> conflict with the half-strength undercurrent on
#                          record; report it, do not pick a winner.
# Ri = N2/shear^2 from two measured fields is also an INDEPENDENT check on the
# Ri ~1.16 vs ~0.27 that was inverted from the Prandtl numbers.
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
    --nemo-gridu $GW/ORCA1_5d_20000101_20000331_grid_U.nc \
    --nemo-gridv $GW/ORCA1_5d_20000101_20000331_grid_V.nc \
    --nemo-day 30 \
    --mesh-mask $MAIN/data/grids/eORCA1.2_mesh_mask.nc \
    --out-json $MAIN/results/omip_nemo/eq_n2_tke_d30.json \
    --plot-out $MAIN/results/omip_nemo/eq_tke_profile_d30.png \
    --tke-budget
echo "=== rc=$? ==="
