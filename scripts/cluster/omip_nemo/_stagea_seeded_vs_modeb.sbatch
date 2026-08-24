#!/bin/bash
#SBATCH --job-name=a2seed
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/a2seed_%j.out
#
# ============================================================================
# HOW MUCH OF THE RESIDUAL DIFFUSIVITY EXCESS IS JUST THE SEEDING?
# ============================================================================
#
# Codex 9400815 #6 refused to let the 3.25x Stage-A residual be attributed to
# anything, on the grounds that Stage A substitutes the diagnostic Mode-B
# equilibrium (three 86400 s solves seeded at tke_background) for NEMO's
# CARRIED prognostic `en`.  It called eliminating that the first uncontrolled
# cause.  It also did not notice that the comparator ALREADY implements the
# controlled version: `run_stage_a2_mode_a` seeds the kernel with NEMO's own
# `en` and takes ONE 3600 s prognostic step.  And the seed is already on disk
# -- RUN_TRD2/restart_8760_global.npz carries en/tn/sn/un/vn/sshn at (75,331,
# 360), rebuilt from the per-rank tiles.  So NO NEMO re-run is needed.
#
# THE COMPARISON, one variable, same file, same record, same columns:
#     Stage A   Mode-B, tke_old=None,  dt=86400, n_iterations=3
#     Stage A2  Mode-A, tke_old=NEMO en, dt=3600, n_iterations=1
# Both now run the FIXED card (kappa_convention="veros_sqrte", 5c486825e).
#
# PRE-REGISTERED, Arctic CALM columns (EVD columns are split out by the
# comparator's avt>1 gate; measured this session, that gate covers 7.8% of wet
# cells and essentially all of them sit at exactly avt=100):
#     ratio = our K_H / NEMO avt.
#     Mode-B at veros_sqrte, ALREADY MEASURED (39ce0701c): 3.25x.
#     SEEDING IS THE CAUSE : A2 <= 1.5x -- Stage A's absolute numbers were an
#         artifact of Mode-B and every attribution built on them is void.
#     SEEDING IS NOT       : A2 >= 2.5x -- the excess survives the exact
#         carry, and the next suspects are the closure INPUTS (N2 time level
#         rn2 vs rn2b, T-point-averaged vs face-native shear, local e3t and
#         wet masking) rather than the TKE integration.
#     INDETERMINATE: 1.5 to 2.5x -- report, claim no direction.
#
# INSTRUMENT SELF-CHECK, run first and reported next to the result: Stage A
# with the card default must REPRODUCE 3.25x in Arctic calm columns.  It was
# measured with an explicit --kappa-convention veros_sqrte override; the card
# now carries that value, so a mismatch means the card change did something
# other than what it claims and NOTHING in this job is trustworthy.  The
# second invocation pins the old amplitude explicitly and must give ~4.81x.
#
# CARRIED CAVEATS, stated with every number:
#   * A2's target is NEMO avt at RECORD 1, not 0 -- the restart stores no avt
#     so rec 0 is an initialisation transient.  That leaves a one-step `en`
#     lag in A2 (small; surface TKE e-folds in minutes to hours).
#   * NEMO's avt is composite: TKE + EVD + any implicit K33 + zdfiwm.  The
#     calm/EVD split removes the convective part; zdfiwm is additive and
#     NON-NEGATIVE, so comparing our TKE-only K_H to it BIASES THE APPARENT
#     EXCESS DOWNWARD.  Any ratio above 1 is therefore a lower bound.
#   * The restart carries `en` but the rebuilt npz dropped `avt_k`/`avm_k`,
#     which NEMO feeds to tke_tke as p_avm.  A2 is exact in the TKE carry,
#     not in the previous-step viscosity.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TRD2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
RST=$TRD2/restart_8760_global.npz
R=$REPO/results/omip_nemo
echo "HEAD: $(git -C $REPO log --oneline -1)"
for f in "$TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc" "$RST"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
NFAIL=0
run () { echo; echo "########## $1"; shift; "$@"; local rc=$?; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1)); }

run "SEEDED (A2) vs MODE-B (A) at the FIXED card — the measurement" \
  $PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
    --tfile $TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc \
    --ufile $TRD2/ORCA1_1h_20010101_20010101_trd1h_U.nc \
    --vfile $TRD2/ORCA1_1h_20010101_20010101_trd1h_V.nc \
    --rec 1 --restart-npz $RST \
    --output-dir $R/stagea_seeded_fixedcard

run "CONTROL: the OLD amplitude on Stage A — must give ~4.81x Arctic calm" \
  $PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
    --tfile $TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc \
    --ufile $TRD2/ORCA1_1h_20010101_20010101_trd1h_U.nc \
    --vfile $TRD2/ORCA1_1h_20010101_20010101_trd1h_V.nc \
    --rec 1 --kappa-convention gaspar_sqrt2e \
    --output-dir $R/stagea_oldamp_control

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
