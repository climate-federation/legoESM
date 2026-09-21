#!/bin/bash
#SBATCH --job-name=smagmask
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/smagmask_%j.out
#
# HOW BIG IS THE SMAGORINSKY VISCOSITY AT C_smag_lap = 3.0?
#
# GLM's adversarial review broke a claim I had already reported: that the MPAS
# harmonisation eliminated lateral viscosity as a driver of the equatorial warm
# bias. The harmonisation lowered the background A_h 100000 -> 20000 AND raised
# C_smag_lap 0.33 -> 3.0. Both the lat-lon and the Voronoi implementations form
# A_smag = (C_smag_lap * Delta)^2 * |D|, so the coefficient enters SQUARED and
# that second change multiplies the flow-dependent term by (3.0/0.33)^2 = 82.6.
# The two contributions are ADDED as separate tendencies
# (ocean_pe_mpas.py: visc = visc + A_h*del2 ; visc = visc + smag_lap), so the
# NET viscosity may have risen. A null measured across that package cannot be
# read as "viscosity does not matter".
#
# PRE-REGISTERED, against the OLD MPAS background of 100000 m2/s:
#   A_smag near the equator >= ~1e5    the Smagorinsky term alone replaces the
#                                      background that was removed. The null is
#                                      then uninformative about viscosity and
#                                      my elimination claim stays RETRACTED.
#   A_smag near the equator <= ~1e4    the background cut dominates, net
#                                      viscosity really did fall several-fold,
#                                      and the null does constrain viscosity.
#   in between                         no verdict; the MPAS-side probe is then
#                                      required rather than optional.
#
# SCOPE, stated because it limits the conclusion: this runs on the TRIPOLE
# snapshot, because the existing tool calls smagorinsky_viscosity_cgrid and
# there is no Voronoi equivalent that returns the coefficient rather than the
# tendency. Cell scales are comparable (1 degree vs MPAS level 7, ~120 km) but
# the strain field is the tripole's, so transferring the number to MPAS is
# PLAUSIBLE, not CONFIRMED, and is labelled that way wherever it is quoted.
#
# Note also measured, not assumed: the lat-lon path caps A_smag at a CFL limit
# (--smag-cfl-safety) and the MPAS path has NO such cap -- grep for cfl in
# operators_voronoi.py's Smagorinsky and in ocean_pe_mpas.py returns nothing.
# So this tripole number is if anything an UNDER-estimate of what MPAS applied.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_fluxacc_wt
MAIN=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
SNAP=$MAIN/results/omip_nemo/trp_orca1faithtint_d30/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "REFUSING: $SNAP absent"; exit 3; }
CMD="$PY scripts/validate/ocean_fidelity/smag_vs_prescribed_viscosity.py --snapshot $SNAP --mesh $MAIN/data/grids/eORCA1.2_mesh_mask.nc --C-smag 3.0 --A-h-uniform 100000 --level 0"
echo "CMD: $CMD"
$CMD
rc=$?; echo "=== rc=$rc ==="
exit $rc
