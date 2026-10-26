#!/bin/bash
#SBATCH --job-name=trdn2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trdn2_%j.out
#
# TENDENCY FIDELITY vs the signed TEOS-10 stratification, one variable.
#
# The task asks for faithfulness "including in the time tendencies to the
# original oracle in both 2D and 3D". The last tendency comparison predates
# the n2_mode="nemo_bn2" + n2_eos_form="teos10" card change (9efc193b4), so
# it says nothing about the current card.
#
# Two arms, differing ONLY in n2_mode. Everything else -- the NEMO trd files,
# the record, the mask, the metric, the card -- is byte-identical, so any
# difference is attributable.
#
# WHAT EACH STAGE MEASURES (from the validator's own docstring):
#   Stage A  : our closure K_H against NEMO's avt. This is where the
#              stratification enters, so it is the stage that should move.
#   Stage B  : the vertical-diffusion OPERATOR driven by NEMO's OWN avt
#              against NEMO's ttrd_zdf. It is fed NEMO's avt, NOT ours, so it
#              should be almost UNCHANGED by an N2 override. If Stage B moves
#              a lot, the override is leaking somewhere it should not, and
#              that is a defect rather than a result.
#
# PRE-REGISTERED, Stage A Antarctic + Arctic K_H vs avt:
#   HELPS    : signed closer to NEMO than insitu in BOTH polar bands.
#   MIXED    : one band improves, one degrades.
#   REFUTED  : signed worse in both -- the one-step KM_ratio gain (job
#              9409902: Arctic 0.664 -> 0.924) did not survive into the
#              tendency comparison, and the two instruments disagree.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
D=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
echo "HEAD: $(git -C $REPO log --oneline -1)"
NFAIL=0
arm () {
  local tag="$1"; shift
  echo; echo "########## ARM $tag"
  $PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
    --tfile $D/ORCA1_1h_20010101_20010101_trd1h_T.nc \
    --ufile $D/ORCA1_1h_20010101_20010101_trd1h_U.nc \
    --vfile $D/ORCA1_1h_20010101_20010101_trd1h_V.nc \
    --rec 1 --skip-maps \
    --output-dir $REPO/results/omip_nemo/trd_n2_$tag "$@"
  local rc=$?; echo "=== $tag rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1))
}

# Card as it stands (nemo_bn2 + teos10) -- no override, so a card drift shows
# up here rather than being hidden behind an explicit flag.
arm signed_card
# One-variable control.
arm insitu_control --n2-mode insitu

# STAGE A2, the discriminator for the calm columns.
#
# Stage A above runs the MODE-B quasi-steady card (prognostic=False, dt=86400,
# 3 iterations). In CALM columns NEMO's en is a decaying remnant sitting near
# rn_emin, which a quasi-steady closure cannot represent -- so calm columns
# are exactly where Mode-B is least trustworthy, and they are where the signed
# N2 looked worst (arctic/calm nrmse 0.89 -> 50.95, i.e. our K_H at 51x NEMO's
# avt) even as the CORRELATION improved 0.040 -> 0.703.
#
# Two candidate causes, and they need different fixes:
#   (a) our closure genuinely over-mixes calm water once the compressibility
#       bias stops holding N2 up -- a real defect in the card;
#   (b) Mode-B cannot reproduce a decaying en, so the calm-column magnitude is
#       an artifact of the quasi-steady approximation, not of the N2 change.
#
# Stage A2 discriminates: it seeds the kernel with NEMO's OWN prognostic en
# from the restart and takes ONE 3600 s step, so `en` is no longer ours to get
# wrong. If (b), the calm-column gap collapses. If (a), it survives.
arm signed_a2   --restart-npz $D/restart_8760_full.npz
arm insitu_a2   --restart-npz $D/restart_8760_full.npz --n2-mode insitu

echo; echo "=== ALL DONE (failed arms: $NFAIL) ==="
exit $((NFAIL > 0))
