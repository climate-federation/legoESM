#!/bin/bash
#SBATCH --job-name=nino3kv
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nino3kv_%j.out
#
# IS THE 120W THERMOCLINE TOO DEEP BECAUSE WE OVER-MIX THERE?
#
# Z20 at 120W is 81.4 m against NEMO's 55.6 m -- 26 m too deep, in the heart
# of Nino3, and arithmetically enough to explain the +3.19 C SST bias. That
# established WHERE, not WHY. The obvious candidate is vertical diffusivity:
# too much K_H at the thermocline smears it downward.
#
# This asks the question directly, against NEMO's OWN avt on NEMO's OWN state,
# with no model run: the tendency validator already computes exactly that
# comparison, it just reported it over "tropics" = -15..15 across ALL
# longitudes, which averages the cold tongue against the Indian and Atlantic
# basins. Adding lon-boxed Nino3/Nino4/eqPac regions makes it answerable.
#
# READING IT:
#   ours/nemo >> 1 in nino3  -> we over-mix there; the deep thermocline and
#                               the warm cold tongue follow, and the lever is
#                               the closure in high-shear equatorial columns.
#   ours/nemo ~ 1            -> diffusivity is NOT the cause; look at
#                               equatorial upwelling (wind stress, vertical
#                               advection) instead.
# Note nino3 sits in the CALM split -- NEMO's EVD does not fire on the
# equator -- so read the /calm rows.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM; cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
D=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
arm () {   # $1 = tag, rest = overrides
  local tag="$1"; shift
  echo; echo "########## ARM $tag"
  $PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
    --tfile $D/ORCA1_1h_20010101_20010101_trd1h_T.nc \
    --ufile $D/ORCA1_1h_20010101_20010101_trd1h_U.nc \
    --vfile $D/ORCA1_1h_20010101_20010101_trd1h_V.nc \
    --rec 1 --skip-maps \
    --output-dir $REPO/results/omip_nemo/nino3_kv_$tag "$@" 2>&1 \
    | grep -aE "lon-boxed|nino|eqpac|cfg/stage_a\]"
  echo "=== $tag rc=${PIPESTATUS[0]}"
}
# PRODUCTION-MATCHED baseline. The signed d30 arm passes --tke-mxl-choice 3
# while the CARD default is 4, so the first run of this probe measured the
# 2.297 ratio at mxl=4 -- a setting the run being explained does not use. The
# instrument must be configured like the run it explains.
arm prod_mxl3 --mxl-choice 3
# The card default, kept for the mxl comparison it now gives for free.
arm baseline
# THE SUSPECT: NEMO's exact nn_pdl=1 Prandtl. Codex review 9409347 called the
# card's "richardson" a STABLE-COLUMN fidelity gap, and the equatorial
# thermocline is exactly a stable, high-shear column -- the 2.30x over-mixing
# lives there. One variable.
arm nemo_ri --prandtl-mode nemo_ri

# REMAINING SUSPECTS for the 2.3x, each one variable off the PRODUCTION card
# (mxl=3). --cfg-override raises if a value equals the card's, so none of
# these can come back a silent no-op.
#
#  etau: the card injects surface TKE below the mixed layer (etau_mode=
#  below_ml, frac 0.08). On the equator the mixed layer is thin, so that flux
#  lands close to the thermocline -- a plausible way to smear it downward.
arm no_etau   --mxl-choice 3 --cfg-override etau_mode=none
#  lc: Langmuir. Same argument, different term, and it is ON in the card.
arm no_lc     --mxl-choice 3 --cfg-override lc=False
#  background diffusivity: bg_diff_scale is 0.0 on the card so this SHOULD be
#  inert. Running it as a NEGATIVE CONTROL -- if it moves nino3, my reading of
#  the card is wrong and every other arm here is suspect.
arm bg_on     --mxl-choice 3 --cfg-override bg_diff_scale=1.0

echo; echo "=== DONE"

