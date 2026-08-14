#!/bin/bash
#SBATCH --job-name=buoysink
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/buoysink_%j.out
#
# THE STEP ERROR THE REVERT EXPOSED.  On the reviewed card (2f04628d6) our K
# SHRINKS ~14% over the hour where NEMO's GROWS 5% (Antarctic 0.709 against a
# 0.861 zero-step reference, job 9405430).  Two of the three options codex
# refuted had been masking that; removing them put it back on the table.
#
# CANDIDATE, named by codex 9405307 and missed by my own card auditor until
# now: the card leaves tke_buoyancy_sink="implicit_linearized", while NEMO
# puts the buoyancy term EXPLICITLY on the RHS with the carried avt --
#     en += rn_Dt * ( p_sh2 - p_avt*rn2 + zfact3*dissl*en )   (zdftke.F90:417)
# "nemo_explicit" implements exactly that (tke.py:1072+), using K_H_old, which
# IS NEMO's p_avt.
#
# DIRECTION IS NOT OBVIOUS AND I AM NOT ASSUMING IT.  For a decay rate r over
# dt, implicit gives e/(1+dt*r) and explicit gives e*(1-dt*r); for moderate
# dt*r the EXPLICIT form removes MORE, which would push our already-low result
# lower.  So this may well go the wrong way.  Running it because it is what
# NEMO does and because the magnitude is unknown, not because I expect a fix.
#
# Also note the card runs n2_mode="insitu", which CLIPS N2 >= 0, so the
# convective SOURCE half of this term is inert on both branches -- this arm
# tests the SINK discretisation only.
#
# PRE-REGISTERED, Antarctic K_M/avm (reviewed card 0.709; zero-step 0.861;
# NEMO's own avm grows 5% over the hour, so the target is ~0.90):
#   HELPS   : 0.80-1.00.
#   NEUTRAL : 0.68-0.80.
#   WORSE   : below 0.68 -- the explicit sink removes more, as the algebra
#             above suggests it might, and the step error is elsewhere.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TRD2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
OUT=$WT/results/omip_nemo/tke_source_ablation
echo "HEAD: $(git -C $WT log --oneline -1)"
NFAIL=0
arm () {
  local tag="$1"; shift
  echo; echo "########## ARM $tag"
  $PY -u scripts/validate/ocean_fidelity/nemo_zero_step_closure.py \
    --restart-npz $TRD2/restart_8760_full.npz \
    --mesh-mask $WT/data/grids/eORCA1.2_mesh_mask.nc \
    --trd-tfile $TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc --trd-rec 0 \
    --out-json $OUT/$tag.json "$@" 2>&1 \
    | grep -aE "^\[cfg\]|^antarctic|^arctic|^ONE STEP|FATAL|Error|ValueError|SystemExit"
  local rc=${PIPESTATUS[0]}; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1))
}

arm reviewed_card_baseline
arm nemo_explicit_buoyancy --cfg-override tke_buoyancy_sink=nemo_explicit

echo; echo "########## card audit (tke_buoyancy_sink now in the flag table)"
$PY -u scripts/validate/ocean_fidelity/audit_orca1_tke_card.py --iwm | tail -20
echo; echo "=== ALL DONE (failed arms: $NFAIL) ==="
exit $((NFAIL > 0))
