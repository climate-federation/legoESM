#!/bin/bash
#SBATCH --job-name=eqprandtl
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqprandtl_%j.out
#
# Codex (job 9887255, finding #4) pointed out that NEMO's Prandtl mapping does
# not take the plain gradient Richardson number: zdftke.F90:381-397 forms
# zri = rn2b * p_avm / (p_sh2 + rn_bshear), a viscosity-weighted ratio.  A card
# read then showed our ORCA1 card selects prandtl_mode="richardson" (plain
# gradient Ri) while this codebase also implements "nemo_ri", the exact NEMO
# form -- a real card/oracle mismatch whose MAGNITUDE is unmeasured.  This run
# measures it, and two other things that need no derivation at all.
#
# THREE INDEPENDENT QUESTIONS.  Each side is scored against ITS OWN archived
# coefficients, so none of this rests on the equilibrium algebra codex broke,
# and none of it compares our instantaneous field to NEMO's five-day mean.
#
# 1. OUR IMPLEMENTATION SELF-CHECK.  Does our archived K_M/K_H equal
#    clamp(Ri/ri_cri, 1, 10) computed from our own N2 and shear?
#      agree high  -> our Prandtl code does what the card says; the question
#                     moves upstream to the Richardson number itself.
#      agree low   -> a defect in our own Prandtl path, which would be the
#                     first thing to fix and would invalidate the 5.24 on
#                     record.  This test can FAIL, which is why it is here.
#
# 2. DOES THE MAPPING REPRODUCE THE ORACLE ON THE ORACLE'S OWN STATE?
#    Same formula, NEMO's archived N2, shear, avm and avt.
#      agree high  -> the Prandtl mapping is the whole story on NEMO's side
#                     too, and our heat-mixing deficit is inherited from Ri.
#      agree low   -> something OTHER than the Prandtl mapping sets NEMO's
#                     heat mixing.  The two candidates are the enhanced-
#                     diffusion branch (rn_evd=100 on tracers) and convective
#                     episodes, which force Pr=1 wherever the before-level N2
#                     is non-positive.  The N2<=0 column measures how much
#                     room that second route has, in the same table.
#
# 3. HOW BIG IS THE CARD GAP?  Both Richardson arguments on our snapshot.
#    The viscosity cancels between numerator and denominator to leading order,
#    so the honest prior is that this is SMALL at the open equator and the
#    mismatch is a fidelity debt rather than the cause of anything.  A large
#    number here would be a surprise and gets reported as one.
#
# PRE-REGISTERED so the read cannot drift: "agree" is the fraction of wet
# interfaces where predicted and archived Prandtl numbers sit within a factor
# of 1.5.  High means above 0.7, low means below 0.3, and anything between is
# reported as INCONCLUSIVE with no story attached.
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
    --out-json $MAIN/results/omip_nemo/eq_prandtl_d30.json \
    --plot-out $MAIN/results/omip_nemo/eq_prandtl_profile_d30.png \
    --prandtl-check
echo "=== rc=$? ==="
