#!/bin/bash
#SBATCH --job-name=iwmbg
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/iwmbg_%j.out
#
# ONE VARIABLE: the interior BACKGROUND, against an oracle whose background is
# not the one its own namelist declares.
#
# ORCA1 sets rn_avm0 = 1.2e-4 and rn_avt0 = 1.2e-5, and the card carries those
# as kappaM_min/kappaH_min. But ORCA1 also sets ln_zdfiwm = .true., and NEMO's
# zdfiwm_init then OVERWRITES the pair it just read (zdfiwm.F90:375-379):
#
#     avmb(:) = rnu        ! 1.4e-6, the MOLECULAR value
#     avtb(:) = 1.e-10     ! "background avt is specified in zdf_iwm"
#
# with the de Lavergne wave field supplying the real interior mixing, clamped
# to [1.4e-7, 1e-2]. NEMO's own output agrees to the digit: the restart's
# avm_k floor is 1.400e-06 = rnu exactly, and the deep equatorial avt runs down
# to 3.5e-07 with 89% of interfaces BELOW rn_avt0.
#
# So every Stage-A number in this campaign has scored a card floored at
# 1.2e-4/1.2e-5 against an oracle floored at 1.4e-6/1e-10 -- 86x on momentum
# and 1.2e5x on tracer. That is the obvious candidate for the ratios that are
# huge in the WHOLE COLUMN and ~1 above 300 m (arctic/calm 397 vs 7.1,
# antarctic 15.5 vs 2.5, nino3 2.311 vs 1.054).
#
# CONFIRMS: the whole-column ratios collapse toward the shallow ones. The "2.3x
#     cold-tongue over-mixing" was a background mismatch and the closure was
#     never implicated below the thermocline.
# REFUTES: they stay large. Then our deep K really is too big for a reason
#     other than the floor, and the floor was a red herring.
# NEITHER: the shallow rows move too. They should NOT -- the closure is orders
#     above both floors there -- so any shallow movement means the floor is
#     binding where I claimed it cannot, and the reading above is wrong.
#
# Matched pair, both arms this same commit, differing ONLY in
# --iwm-backgrounds. Nothing here runs the model.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM; cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
echo "=== git SHA: $(git -C $REPO rev-parse HEAD) ==="
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
D=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
arm () {
  local tag="$1"; shift
  echo; echo "########## ARM $tag"
  $PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
    --tfile $D/ORCA1_1h_20010101_20010101_trd1h_T.nc \
    --ufile $D/ORCA1_1h_20010101_20010101_trd1h_U.nc \
    --vfile $D/ORCA1_1h_20010101_20010101_trd1h_V.nc \
    --rec 1 --skip-maps --mxl-choice 3 \
    --output-dir $REPO/results/omip_nemo/iwmbg_$tag "$@" 2>&1 \
    | grep -aE "floors detected|control\]|^  region|^  arctic/calm|^  antarctic/calm|^  nino3/calm|^  eqpac/calm|FATAL"
  echo "=== $tag rc=${PIPESTATUS[0]}"
}
arm namzdf                       # the mismatched default: 1.2e-4 / 1.2e-5
arm molecular --iwm-backgrounds  # what ln_zdfiwm=.true. actually leaves behind
echo; echo "=== DONE"
