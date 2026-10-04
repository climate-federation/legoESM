#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE: unhandled command failure at run.sh line %s (exit %s)\n' \
    "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

# VORTEX kt=1..10 ACQUISITION -- USER-EXECUTED ONLY.
#
#   preflight (default):  run.sh
#   acquire:              run.sh --run
#
# The agent that wrote this file did not run NEMO: MPI is refused in its
# sandbox.  Without --run the script only CHECKS its inputs and prints the
# commands it would issue; nothing is built, nothing is written outside /tmp.
#
# WHAT THIS ACQUIRES.  tests/VORTEX is the baroclinic vortex on a beta-plane.
# Its namelist selects exactly the switch set already certified for
# LOCK_EXCHANGE and OVERFLOW -- flux-form UP3 momentum, FCT2 tracers, hpg_sco,
# split-explicit dynspg_ts with nn_bt_flt=3, constant zdf -- plus ONE thing the
# tanks structurally cannot exercise: a LIVE beta-plane Coriolis with
# ln_dynvor_een=.true.  The tanks have f=0 and one wet row, so their vorticity
# operator is dead.
#
# READ THIS BEFORE YOU RUN IT.  That last switch is why this case needed its own
# operator.  Under flux-form momentum NEMO hands the energy-and-enstrophy scheme
# Coriolis plus a metric term, and that metric term is built from differences of
# the mesh's scale factors -- which this Cartesian mesh makes exactly zero, so
# the scheme IS the Coriolis operator here.  Round 1 declared that as a gap and
# fixed the card closed; round 2 transcribed it, and the card now executes.
#
# Round 2 also re-acquires: the deck below no longer overrides the shipped
# equation of state (decision 69), so round 1's record is superseded from step 1
# onward.  The INITIAL STATE is unaffected -- nothing in it reads the equation of
# state -- so this run must reproduce round 1's initial-state comparison exactly,
# and a difference there is the finding rather than the ladder.
#
# TWO CONFIGURATIONS, DELIBERATELY.  A brand-new card has no un-instrumented
# reference to judge its writer against, so this script builds BOTH:
#
#   VORTEX_OMIP_L1       shipped MY_SRC only        -> the reference restart
#   VORTEX_OMIP_L1_P3    + the step-record writer   -> the kt=1..10 records
#
# and refuses unless the two restarts at step 10 are byte-identical.  That is
# the passivity criterion note AS makes binding; a stream-to-stream comparison
# between two differently instrumented builds is NOT a refusal criterion.
#
# THE ZOOM IS OUT OF SCOPE.  cpp_VORTEX.fcm compiles key_agrif for a 1:3 nest.
# legoESM has no nesting machinery, so both builds DROP key_agrif and the run
# is the PARENT grid alone.  The child deck files are removed from EXP00 below
# so nothing can silently pick them up.
#
# DELIBERATE DECK DEVIATIONS, all in the committed namelist patch and all
# visible in the diff it prints:
#   * nn_itend = nn_stock = 10 (the tanks' own kt1_10 cadence) instead of 3000.
#   * ln_meshmask = .true., so the geometry receipt is written.
#
# PATH.  FCM's extract step needs perl's Text::Balanced, which the system perl
# lacks; the conda build environment must come first, exactly as the tanks'
# round-33 acquisition documents.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly TEST_CASE=VORTEX
# The ladder length and the coordinate tag cn_exp carries.  Both are 10
# and ZCO for every certified variant; decision 88's VORTEX_SMT rungs
# override them (ZPS, and 3000 steps for the 100-day record).
steps=10
# How many kt=1.. records the admission must find.  Equal to the run length
# for every ladder rung, but the step-record writer only fires for the first
# SIXTY steps -- stprk3_step_record.patch:19 is
#   IF( lwp .AND. kstp >= nit000 .AND. kstp <= nit000 + 59 ) THEN
# -- so a 3000-step run has 60 records and a checker asked for 3000 would
# refuse a perfectly good acquisition.  The RESTART the admission compares
# is still the one at the full run length.
record_steps=
coord_tag=ZCO
smt_zgr=

# TWO CARDS, ONE SCRIPT (decision 73, operator note BJ).  The vector-EEN card is
# the SAME experiment -- same geometry, same simplified equation of state, same
# eddy, no forcing, no implicit vertical advection -- with ONE thing changed:
# the momentum scheme set becomes ORCA2's and GYRE's, vector-invariant advection
# with the energy-and-enstrophy vorticity.  Everything else in this script is
# shared deliberately, so the two records differ by exactly that one deck hunk
# and nothing about the build, the writer or the admission can drift between
# them.
#
#   --variant flux (default)  the round-2 card: flux-form UP3, EEN on Coriolis
#                             plus the (bitwise zero) metric term
#   --variant vec             the round-3 card: vector-invariant, EEN on
#                             Coriolis plus RELATIVE vorticity
#   --variant vecrhs          the round-4 acquisition: the SAME vector-EEN
#                             deck, with one EXTRA read-only writer that dumps
#                             the momentum right-hand side after each routine
#                             that contributes to it inside stp_2D.  Round 4
#                             proved, by substitution, that the whole of this
#                             card's stage-1 momentum error lives in that
#                             completed right-hand side (handing legoESM
#                             NEMO's own copy puts stage 1 at 1.1e-16, from
#                             1.7e-05); naming WHICH TERM needs the split,
#                             and no existing record carries it.
#
# Each variant writes its OWN evidence directory beside the other and builds its
# OWN pair of NEMO configurations.  Nothing is ever overwritten: the acquire arm
# refuses a target that already exists, and moving an old build aside is the
# operator's call, never this script's.
variant=flux
do_run=0
# Round 208 / VORTEX round 21 (decision 74, operator note BZ): the resolution
# rungs REUSE the certified executables instead of rebuilding.  NEMO derives
# the whole grid at RUN time -- usrdef_nam.f90:138-144 of VORTEX_OMIP_L1_P3
# sets kpi = NINT(1800e3/rn_dx)+3, kpj likewise and kpk = NINT(5000/rn_dz)+1
# from the namelist it has just read -- so a refined deck needs a new namelist
# and nothing else.  Rebuilding would produce the same binary and would destroy
# the control: reusing the certified one proves the rungs differ in the DECK
# alone.  The hashes are checked against the committed manifest below.
reuse_build=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) do_run=1 ;;
    --variant) shift; variant=${1:-} ;;
    --variant=*) variant=${1#--variant=} ;;
    *) printf 'Usage: %s [--run] [--variant flux|vec|vecrhs|stage23|spgts|stage123flx|...]\n' "$0" >&2 ; exit 64 ;;
  esac
  shift
done

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
round192_repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run.sh is not inside a git worktree\n' >&2; exit 66; }
if [[ -n "$(git -C "$round192_repo_root" status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: legoESM worktree is dirty; commit the exact acquisition tool first\n' >&2
  exit 66
fi
round192_git_sha=$(git -C "$round192_repo_root" rev-parse HEAD) \
  || { printf 'REFUSE: cannot resolve the legoESM commit stamp\n' >&2; exit 66; }
readonly round192_repo_root round192_git_sha
case "$variant" in
  flux)
    # Round 2 writes BESIDE round 1, never over it.  Round 1's record was
    # produced on a deck that selected a different equation of state, so it is
    # superseded from step 1 onward -- but its INITIAL STATE is the reference
    # the round-2 gate compares against, and overwriting it would destroy the
    # only control that can tell a deck change from a transcription defect.
    deck_basename=namelist_cfg_omip_l1.patch
    default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round2
    ref_name=VORTEX_OMIP_L1
    exp_name=VORTEX_OMIP_L1
    tag=round2
    ;;
  vec)
    deck_basename=namelist_cfg_vec_een.patch
    default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round3
    ref_name=VORTEX_VEC_OMIP_L1
    exp_name=VORTEX_VEC_OMIP_L1
    tag=round3_vec
    ;;
  vecrhs)
    # The SAME deck as the vec variant -- one hunk apart from the shipped
    # namelist, exactly as decision 73 requires -- with a SECOND instrument.
    # New target names, new evidence directory: nothing round 3 produced is
    # touched, and the acquire arm below refuses any target that exists.
    deck_basename=namelist_cfg_vec_een.patch
    default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round4_rhsterms
    ref_name=VORTEX_VEC_R4_OMIP_L1
    # The build name is new so nothing round 3 produced can be overwritten,
    # but the RUN's experiment name comes from the SHARED deck, so the file
    # NEMO writes still carries round 3's cn_exp.  Keep the two apart here
    # rather than have the admission look for a file that is never written.
    exp_name=VORTEX_VEC_OMIP_L1
    tag=round4_vec_rhsterms
    ;;
  stage23)
    # Round 192 / VORTEX round 8: same vector-EEN deck, new paired build,
    # additive stage-2/3 term writer.  The existing record stops at stage
    # inputs/outputs and cannot split HPG, VOR/EEN, KEG, ZAD, LDF and ZDF.
    deck_basename=namelist_cfg_vec_een.patch
    default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round192/oracle_stage23_terms
    ref_name=VORTEX_VEC_R8_OMIP_L1
    exp_name=VORTEX_VEC_OMIP_L1
    tag=round192_stage23
    ;;
  stage123flx)
    # Round 200 / VORTEX round 16: the FLUX card, new paired build, additive
    # stage-1/2/3 term writer.  Round 200's stage-local walk puts 4.4e-08 of
    # the flux card's kt=2 velocity error in STAGE 1, where the only momentum
    # statement NEMO runs is the flux-form advection call -- and no record
    # carries that call's operands (the advective transports zFu, zFv, zFw) or
    # the right-hand side on either side of it.  Round 192's record is the
    # VECTOR card's and starts at stage 2.
    deck_basename=namelist_cfg_omip_l1.patch
    default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round200/oracle_stage123_flux_terms
    ref_name=VORTEX_R16_OMIP_L1
    exp_name=VORTEX_OMIP_L1
    tag=round200_stage123flx
    ;;
  spgts)
    # Round 196 / VORTEX round 12: same vector-EEN deck, new paired build,
    # additive per-substep writer inside the split-explicit barotropic solve.
    # Round 195 named that solve as the owner of what the held two-solve
    # candidate leaves behind, and no existing record carries its SUBSTEP
    # operands -- round 192's record has the solve's OUTPUT frames only.
    deck_basename=namelist_cfg_vec_een.patch
    default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round196/oracle_spgts_substeps
    ref_name=VORTEX_VEC_R12_OMIP_L1
    exp_name=VORTEX_VEC_OMIP_L1
    tag=round196_spgts
    ;;
  res15flx | res15vec | res10flx | res10vec)
    # DECISION 74's resolution ladder.  The deck rule is NEMO's OWN: its VORTEX
    # ships an AGRIF zoom at refinement ratio 3 3 3
    # (tests/VORTEX/EXPREF/AGRIF_FixedGrids.in:2 "22 41 22 41 3 3 3"), and the
    # child namelist it ships for that zoom
    # (tests/VORTEX/EXPREF/1_namelist_cfg) changes exactly three things against
    # the parent: rn_dx and rn_dy 30000 -> 10000 (:21-22), rn_Dt 2880 -> 960
    # (:43) and nn_itend 3000 -> 6000 (:34); it also adds a &namagrif sponge
    # block (:103-108) which has no meaning in a non-nested run.  rn_dz (:23),
    # rn_ppgphi0, rn_ppumax, nn_rot, nn_e = 48 (:222) and every physics switch
    # are UNCHANGED, and both lateral diffusion operators are OFF in parent and
    # child alike, so there is no viscosity to rescale.  The rule is therefore
    # dx -> dx/r and dt -> dt/r with everything else held; the 10 km rung IS
    # NEMO's child deck, and the 15 km rung is the same rule at r = 2.
    # nn_itend is the RUN LENGTH: the child's 6000 is not 3*3000, so no ratio-2
    # analogue exists.  It is registered as DECISION_NEEDED in the receipt and
    # is inert here, because this acquisition pins nn_itend = 10 at EVERY
    # resolution, exactly as the 30 km records do.
    reuse_build=1
    case "$variant" in
      res15flx) deck_basename=namelist_cfg_omip_l1_15km.patch
                ref_name=VORTEX_OMIP_L1 ; exp_name=VORTEX_R21_15KM
                tag=round208_res15_flux ;;
      res15vec) deck_basename=namelist_cfg_vec_een_15km.patch
                ref_name=VORTEX_VEC_OMIP_L1 ; exp_name=VORTEX_VEC_R21_15KM
                tag=round208_res15_vec ;;
      res10flx) deck_basename=namelist_cfg_omip_l1_10km.patch
                ref_name=VORTEX_OMIP_L1 ; exp_name=VORTEX_R21_10KM
                tag=round208_res10_flux ;;
      res10vec) deck_basename=namelist_cfg_vec_een_10km.patch
                ref_name=VORTEX_VEC_OMIP_L1 ; exp_name=VORTEX_VEC_R21_10KM
                tag=round208_res10_vec ;;
    esac
    # phase3/vortex_ladder/<resolution>/<card>, per the round-208 brief.
    ladder_res=${variant:3:2}km ; ladder_card=${variant:5}
    default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_ladder/$ladder_res/$ladder_card
    ;;
  smtflx | smtvec | smtflx100d | smtvec100d | \
  smtflxr3 | smtvecr3 | smtflx100dr3 | smtvec100dr3 | \
  smtflxspgts | smtvecspgts | smtvecrhs)
    # DECISION 88 (user, 2026-10-03), operator note CC: VORTEX WITH TOPOGRAPHY.
    # The SAME 30 km VORTEX deck -- rn_dx 30000, rn_Dt 2880, rn_dz 500, ten
    # levels, every physics switch as the certified cards pin it -- with a
    # user-defined Gaussian seamount and z-coordinate PARTIAL STEPS, through
    # NEMO's own usrdef_zgr hook.  The shipped VORTEX_* builds and the four
    # certified VORTEX configurations are NOT touched: these are new ones.
    #
    # THE ONE cpp KEY THAT CHANGES, and it is forced by NEMO, not chosen:
    # src/OCE/DOM/domzgr.F90:259 refuses partial steps under key_vco_1d
    #   IF( l_zps ) CALL ctl_stop('STOP','domzgr: key_vco_1d and l_zps=T are
    #                             incompatible. Fix usrdef_zgr !')
    # and NEMO's own name for the partial-cell key is key_vco_1d3d
    # (domzgr.F90:242-243 "z-partial cells"; it is the key OVERFLOW's and
    # IWAVE's zps branches are guarded by).  key_qco and key_RK3 are
    # unchanged, and key_xios/key_agrif are dropped exactly as every other
    # variant drops them.
    smt_zgr=$here/vortex_smt_usrdef_zgr.F90
    coord_tag=ZPS
    case "$variant" in
      smtflx)
        deck_basename=namelist_cfg_smt_omip_l1.patch
        ref_name=VORTEX_SMT_OMIP_L1 ; exp_name=VORTEX_SMT_OMIP_L1
        tag=round211_smt_flux
        default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/VORTEX_SMT_OMIP_L1_P3/kt1_10 ;;
      smtvec)
        deck_basename=namelist_cfg_smt_vec_een.patch
        ref_name=VORTEX_SMT_VEC_R8_OMIP_L1 ; exp_name=VORTEX_SMT_VEC_OMIP_L1
        tag=round211_smt_vec
        default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/VORTEX_SMT_VEC_R8_OMIP_L1_P3/kt1_10 ;;
      smtflx100d)
        record_steps=60
        # NEMO's own shipped run length, nn_itend = 3000 steps of rn_Dt =
        # 2880 s = 100 days, with nn_stock = 30 (daily) restarts -- the same
        # cadence round 210 measured the flat cards over.  The builds are the
        # ones the kt=1..10 rung above produced, re-proved by hash.
        reuse_build=1 ; steps=3000
        deck_basename=namelist_cfg_smt_omip_l1_100d.patch
        ref_name=VORTEX_SMT_OMIP_L1 ; exp_name=VORTEX_SMT_OMIP_L1
        tag=round211_smt_flux_100d
        default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/VORTEX_SMT_OMIP_L1_P3/day100 ;;
      smtvec100d)
        record_steps=60
        reuse_build=1 ; steps=3000
        deck_basename=namelist_cfg_smt_vec_een_100d.patch
        ref_name=VORTEX_SMT_VEC_R8_OMIP_L1 ; exp_name=VORTEX_SMT_VEC_OMIP_L1
        tag=round211_smt_vec_100d
        default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/VORTEX_SMT_VEC_R8_OMIP_L1_P3/day100 ;;
      # ---- ROUND 213 / VORTEX_SMT round 3 (operator note CC addendum 3,
      # item 1).  Round 2 fixed a defect in the seamount hook's CONTROL PRINT
      # (rank-local MINVAL/MAXVAL -> global mpp_min/mpp_max) and could not
      # re-run, so the committed hook is not the one that produced the
      # admitted evidence.  These four variants re-acquire with the committed
      # hook into NEW configurations, VORTEX_SMT_R3_*; round 1's four build
      # directories and four evidence directories are NOT touched, so the
      # bit-identity of the two sets is a measurement rather than a claim.
      # The experiment name (hence cn_exp, hence the restart file name) is
      # the SAME, because the deck is the same deck: only the build directory
      # and the evidence directory are new.  Exactly the `vecrhs` pattern.
      smtflxr3)
        deck_basename=namelist_cfg_smt_omip_l1.patch
        ref_name=VORTEX_SMT_R3_OMIP_L1 ; exp_name=VORTEX_SMT_OMIP_L1
        tag=round213_smt_flux
        default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round3/VORTEX_SMT_R3_OMIP_L1_P3/kt1_10 ;;
      smtvecr3)
        deck_basename=namelist_cfg_smt_vec_een.patch
        ref_name=VORTEX_SMT_R3_VEC_R8_OMIP_L1 ; exp_name=VORTEX_SMT_VEC_OMIP_L1
        tag=round213_smt_vec
        default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round3/VORTEX_SMT_R3_VEC_R8_OMIP_L1_P3/kt1_10 ;;
      smtflx100dr3)
        record_steps=60
        reuse_build=1 ; steps=3000
        deck_basename=namelist_cfg_smt_omip_l1_100d.patch
        ref_name=VORTEX_SMT_R3_OMIP_L1 ; exp_name=VORTEX_SMT_OMIP_L1
        tag=round213_smt_flux_100d
        default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round3/VORTEX_SMT_R3_OMIP_L1_P3/day100 ;;
      smtvec100dr3)
        record_steps=60
        reuse_build=1 ; steps=3000
        deck_basename=namelist_cfg_smt_vec_een_100d.patch
        ref_name=VORTEX_SMT_R3_VEC_R8_OMIP_L1 ; exp_name=VORTEX_SMT_VEC_OMIP_L1
        tag=round213_smt_vec_100d
        default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round3/VORTEX_SMT_R3_VEC_R8_OMIP_L1_P3/day100 ;;
      # ---- ROUND 215 / VORTEX_SMT round 5 (operator note CC addendum 5).
      # The kt=2 sea-surface-height error is the SAME size under both
      # momentum programs (3.73e-07 flux, 3.66e-07 vector) while the flat
      # VORTEX pair is at the bar, so its owner is a statement the two
      # programs SHARE: the free-surface / split-explicit barotropic path
      # over partial cells.  No seamount record carries that solve's
      # SUBSTEP operands -- the R3 records hold its OUTPUT only -- so these
      # two variants compile the round-196 per-substep writer into the R3
      # seamount configurations.  NEW build directories (VORTEX_SMT_R5_*):
      # rounds 1 and 3 builds are never moved or rebuilt, and the admitted
      # R3 restart is what the additions-only proof compares to.
      smtflxspgts)
        deck_basename=namelist_cfg_smt_omip_l1.patch
        ref_name=VORTEX_SMT_R5_OMIP_L1 ; exp_name=VORTEX_SMT_OMIP_L1
        tag=round215_smt_flux_spgts
        default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round5/VORTEX_SMT_R5_OMIP_L1_P3/spgts ;;
      smtvecspgts)
        deck_basename=namelist_cfg_smt_vec_een.patch
        ref_name=VORTEX_SMT_R5_VEC_R8_OMIP_L1 ; exp_name=VORTEX_SMT_VEC_OMIP_L1
        tag=round215_smt_vec_spgts
        default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round5/VORTEX_SMT_R5_VEC_R8_OMIP_L1_P3/spgts ;;
      # The barotropic substep walk (round 215) names the loop-entry
      # depth-averaged slow forcing as the first non-bit operand, and NEMO's
      # own depth-average statement (stp2d.f90:176-178) accounts for only
      # part of it -- the rest is inside the 3-D right-hand side that
      # statement averages (hpg, ldf, vor, wzv, keg, zad; stp2d.f90:134-170).
      # No seamount record carries those per-term boundaries; this variant is
      # round 4's stp2d per-term writer on the seamount vector deck.
      smtvecrhs)
        deck_basename=namelist_cfg_smt_vec_een.patch
        ref_name=VORTEX_SMT_R5R_VEC_R8_OMIP_L1 ; exp_name=VORTEX_SMT_VEC_OMIP_L1
        tag=round215_smt_vec_rhs
        default_evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round5/VORTEX_SMT_R5R_VEC_R8_OMIP_L1_P3/rhs ;;
    esac
    ;;
  *)
    printf 'REFUSE: unknown variant %s; expected flux, vec, vecrhs, stage23, spgts, stage123flx, res15flx, res15vec, res10flx, res10vec, smtflx, smtvec, smtflx100d, smtvec100d, smtflxr3, smtvecr3, smtflx100dr3, smtvec100dr3, smtflxspgts, smtvecspgts or smtvecrhs\n' \
      "$variant" >&2
    exit 64
    ;;
esac
readonly EVIDENCE=${EVIDENCE:-$default_evidence}
readonly STEPS=$steps
readonly RECORD_STEPS=${record_steps:-$steps}
readonly COORD_TAG=$coord_tag
readonly SMT_ZGR=$smt_zgr
readonly REF_CFG=$ref_name
readonly RUN_CFG=${ref_name}_P3
readonly RESTART=$(printf '%s_%s_%08d_restart.nc' "$exp_name" "$COORD_TAG" "$STEPS")
readonly TAG=$tag
readonly INSTRUMENT=$here/stprk3_step_record.patch
# The second, round-4 instrument.  Empty for every variant but vecrhs.
if [[ "$variant" == "vecrhs" || "$variant" == "smtvecrhs" ]]; then
  RHS_INSTRUMENT=$here/stp2d_rhs_terms_record.patch
else
  RHS_INSTRUMENT=
fi
readonly RHS_INSTRUMENT
readonly SHIPPED_STP2D=$NEMO_ROOT/src/OCE/stp2d.F90
if [[ "$variant" == "spgts" || "$variant" == "smtflxspgts" \
   || "$variant" == "smtvecspgts" ]]; then
  SPGTS_INSTRUMENT=$here/dynspg_ts_substep_record.patch
  SPGTS_MODULE=$here/vortex_r12_spgts_terms.F90
  SPGTS_STUBS=$here/vortex_r12_spgts_terms_syntax_stubs.F90
else
  SPGTS_INSTRUMENT=
  SPGTS_MODULE=
  SPGTS_STUBS=
fi
readonly SPGTS_INSTRUMENT SPGTS_MODULE SPGTS_STUBS
readonly SHIPPED_SPGTS=$NEMO_ROOT/src/OCE/DYN/dynspg_ts.F90
if [[ "$variant" == "stage23" || "$variant" == "smtvec" || "$variant" == "smtvec100d" \
   || "$variant" == "smtvecr3" || "$variant" == "smtvec100dr3" \
   || "$variant" == "smtvecspgts" ]]; then
  # VORTEX_SMT_VEC_R8 is a copy of the certified VORTEX_VEC_R8_OMIP_L1_P3
  # instrumented build, so it carries the SAME stage-2/3 term writer.
  STAGE_INSTRUMENT=$here/stprk3_stage_terms_record.patch
  DYNADV_INSTRUMENT=$here/dynadv_stage_terms_record.patch
  STAGE_MODULE=$here/vortex_r8_stage_terms.F90
  STAGE_STUBS=$here/vortex_r8_stage_terms_syntax_stubs.F90
  STAGE_MODULE_NAME=vortex_r8_stage_terms.F90
  STAGE_SYMBOL=vortex_r8_stage
  STAGE_FLAG_NAME=--stage-terms
elif [[ "$variant" == "stage123flx" ]]; then
  # One patched file, not two: the flux-form advection trend is recorded at
  # its CALL SITE in stprk3_stg, so dynadv.F90 stays shipped.
  STAGE_INSTRUMENT=$here/stprk3_stage123_flux_record.patch
  DYNADV_INSTRUMENT=
  STAGE_MODULE=$here/vortex_r16_stage_terms.F90
  STAGE_STUBS=$here/vortex_r16_stage_terms_syntax_stubs.F90
  STAGE_MODULE_NAME=vortex_r16_stage_terms.F90
  STAGE_SYMBOL=vortex_r16_stage
  STAGE_FLAG_NAME=--stage-flux-terms
else
  STAGE_INSTRUMENT=
  DYNADV_INSTRUMENT=
  STAGE_MODULE=
  STAGE_STUBS=
  STAGE_MODULE_NAME=
  STAGE_SYMBOL=
  STAGE_FLAG_NAME=
fi
readonly STAGE_INSTRUMENT DYNADV_INSTRUMENT STAGE_MODULE STAGE_STUBS
readonly STAGE_MODULE_NAME STAGE_SYMBOL STAGE_FLAG_NAME
readonly SHIPPED_STG=$NEMO_ROOT/src/OCE/stprk3_stg.F90
readonly SHIPPED_DYNADV=$NEMO_ROOT/src/OCE/DYN/dynadv.F90
readonly DECK=$here/$deck_basename
readonly CHECKER=$here/check_records.py
readonly SHIPPED_STP=$NEMO_ROOT/src/OCE/stprk3.F90
readonly SHIPPED_CFG=$NEMO_ROOT/tests/$TEST_CASE/EXPREF/namelist_cfg
readonly SRC_CASE=$NEMO_ROOT/tests/$TEST_CASE

ref_cfg=$NEMO_ROOT/tests/$REF_CFG
run_cfg=$NEMO_ROOT/tests/$RUN_CFG

# ---------------------------------------------------------------- preflight
for path in "$INSTRUMENT" "$DECK" "$CHECKER" "$SHIPPED_STP" "$SHIPPED_CFG"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 66; }
done
if [[ -n "$RHS_INSTRUMENT" ]]; then
  [[ -f "$RHS_INSTRUMENT" ]] \
    || { printf 'REFUSE: missing %s\n' "$RHS_INSTRUMENT" >&2; exit 66; }
  [[ -f "$SHIPPED_STP2D" ]] \
    || { printf 'REFUSE: missing %s\n' "$SHIPPED_STP2D" >&2; exit 66; }
  # Same premise as the stprk3 writer: this case must not already override
  # the file the instrument patches.
  if [[ -e "$SRC_CASE/MY_SRC/stp2d.F90" ]]; then
    printf 'REFUSE: %s overrides stp2d.F90; the shared-writer premise is false\n' \
      "$TEST_CASE" >&2
    exit 66
  fi
fi
if [[ -n "$SPGTS_INSTRUMENT" ]]; then
  for path in "$SPGTS_INSTRUMENT" "$SPGTS_MODULE" "$SPGTS_STUBS" "$SHIPPED_SPGTS"; do
    [[ -f "$path" ]] \
      || { printf 'REFUSE: missing barotropic-record input %s\n' "$path" >&2; exit 66; }
  done
  for override in dynspg_ts.F90 vortex_r12_spgts_terms.F90; do
    if [[ -e "$SRC_CASE/MY_SRC/$override" ]]; then
      printf 'REFUSE: %s overrides %s; the shared-source premise is false\n' \
        "$TEST_CASE" "$override" >&2
      exit 66
    fi
  done
  if [[ $(grep -c '^-' "$SPGTS_INSTRUMENT") -ne $(grep -c '^---' "$SPGTS_INSTRUMENT") ]]; then
    printf 'REFUSE: %s deletes or changes a shipped line; it must only ADD\n' \
      "$SPGTS_INSTRUMENT" >&2
    exit 67
  fi
fi
if [[ -n "$STAGE_INSTRUMENT" ]]; then
  for path in "$STAGE_INSTRUMENT" ${DYNADV_INSTRUMENT:+"$DYNADV_INSTRUMENT"} \
              "$STAGE_MODULE" "$STAGE_STUBS" "$SHIPPED_STG" \
              ${DYNADV_INSTRUMENT:+"$SHIPPED_DYNADV"}; do
    [[ -f "$path" ]] \
      || { printf 'REFUSE: missing stage-term input %s\n' "$path" >&2; exit 66; }
  done
  for override in stprk3_stg.F90 ${DYNADV_INSTRUMENT:+dynadv.F90} "$STAGE_MODULE_NAME"; do
    if [[ -e "$SRC_CASE/MY_SRC/$override" ]]; then
      printf 'REFUSE: %s overrides %s; the shared-source premise is false\n' \
        "$TEST_CASE" "$override" >&2
      exit 66
    fi
  done
fi
if [[ -n "$SMT_ZGR" ]]; then
  # Decision 88's seamount hook.  It REPLACES tests/VORTEX/MY_SRC/usrdef_zgr.F90
  # in the new configuration only; the shipped file is never edited.
  [[ -f "$SMT_ZGR" ]] \
    || { printf 'REFUSE: missing seamount source %s\n' "$SMT_ZGR" >&2; exit 66; }
  [[ -f "$SRC_CASE/MY_SRC/usrdef_zgr.F90" ]] \
    || { printf 'REFUSE: the shipped case has no usrdef_zgr.F90 to replace\n' >&2; exit 66; }
  if cmp -s "$SMT_ZGR" "$SRC_CASE/MY_SRC/usrdef_zgr.F90"; then
    printf 'REFUSE: the seamount source is identical to the shipped flat-bottom one\n' >&2
    exit 67
  fi
  # The three things the configuration IS.  A copy that lost any of them
  # would silently run the flat box under the seamount's name.
  grep -q 'ld_zps    = .TRUE.' "$SMT_ZGR" \
    || { printf 'REFUSE: the seamount source does not select partial steps\n' >&2; exit 67; }
  for want in 'pp_smt_H0 =  5000._wp' 'pp_smt_A  =  1000._wp' \
              'pp_smt_L  =   150.e3_wp' 'pp_smt_x0 =  -300.e3_wp' \
              'pp_smt_y0 =     0.e3_wp'; do
    grep -qF "$want" "$SMT_ZGR" \
      || { printf 'REFUSE: the seamount source does not pin %s\n' "$want" >&2; exit 67; }
  done
  # NEMO refuses partial steps under key_vco_1d (domzgr.F90:259), so the
  # shipped key set CANNOT be carried unchanged here.  Prove that refusal
  # still exists in this tree rather than quoting a comment at it.
  grep -q "key_vco_1d and l_zps=T are incompatible" "$NEMO_ROOT/src/OCE/DOM/domzgr.F90" \
    || { printf 'REFUSE: domzgr no longer refuses zps under key_vco_1d; re-read the coordinate rules\n' >&2
         exit 66; }
  grep -q 'key_vco_1d' "$SRC_CASE/cpp_${TEST_CASE}.fcm" \
    || { printf 'REFUSE: cpp_%s.fcm no longer compiles key_vco_1d; re-read the case\n' "$TEST_CASE" >&2
         exit 66; }
fi
[[ -d "$SRC_CASE/MY_SRC" && -d "$SRC_CASE/EXPREF" ]] \
  || { printf 'REFUSE: %s is not the shipped test case\n' "$SRC_CASE" >&2; exit 66; }
# The card is the PARENT grid; refuse if the shipped case stopped being the
# AGRIF one this script was written against (its assumptions would be stale).
if ! grep -q 'key_agrif' "$SRC_CASE/cpp_${TEST_CASE}.fcm"; then
  printf 'REFUSE: cpp_%s.fcm no longer compiles key_agrif; re-read the case\n' \
    "$TEST_CASE" >&2
  exit 66
fi
# The writer premise: this case must NOT override stprk3 in its own MY_SRC.
if [[ -e "$SRC_CASE/MY_SRC/stprk3.F90" ]]; then
  printf 'REFUSE: %s overrides stprk3.F90; the shared-writer premise is false\n' \
    "$TEST_CASE" >&2
  exit 66
fi
# A WRITE-only instrument may ADD lines; it may not delete or change one.
# '^-[^-]' would miss a deleted BLANK line, so count every removal line and
# subtract only the '---' file header.
if [[ $(grep -c '^-' "$INSTRUMENT") -ne $(grep -c '^---' "$INSTRUMENT") ]]; then
  printf 'REFUSE: %s deletes or changes a shipped line; it must only ADD\n' \
    "$INSTRUMENT" >&2
  exit 67
fi
if [[ -n "$RHS_INSTRUMENT" ]]; then
  if [[ $(grep -c '^-' "$RHS_INSTRUMENT") -ne $(grep -c '^---' "$RHS_INSTRUMENT") ]]; then
    printf 'REFUSE: %s deletes or changes a shipped line; it must only ADD\n' \
      "$RHS_INSTRUMENT" >&2
    exit 67
  fi
fi
if [[ -n "$STAGE_INSTRUMENT" ]]; then
  for patch_file in "$STAGE_INSTRUMENT" ${DYNADV_INSTRUMENT:+"$DYNADV_INSTRUMENT"}; do
    if [[ $(grep -c '^-' "$patch_file") -ne $(grep -c '^---' "$patch_file") ]]; then
      printf 'REFUSE: %s deletes or changes a shipped line; it must only ADD\n' \
        "$patch_file" >&2
      exit 67
    fi
  done
fi
dry=$(mktemp -d /tmp/vortex-r1-dryrun.XXXXXX)
cp "$SHIPPED_STP" "$dry/stprk3.F90"
cp "$SHIPPED_CFG" "$dry/namelist_cfg"
patch -s "$dry/stprk3.F90" <"$INSTRUMENT" \
  || { printf 'REFUSE: the step-record instrument does not apply to the shipped stprk3\n' >&2
       rm -rf "$dry"; exit 67; }
patch -s "$dry/namelist_cfg" <"$DECK" \
  || { printf 'REFUSE: the deck patch does not apply to the shipped namelist_cfg\n' >&2
       rm -rf "$dry"; exit 67; }
grep -q 'NEMO_L1_ENTRY_1' "$dry/stprk3.F90" \
  || { printf 'REFUSE: the patched stprk3 carries no step-record writer\n' >&2
       rm -rf "$dry"; exit 67; }
if [[ -n "$RHS_INSTRUMENT" ]]; then
  cp "$SHIPPED_STP2D" "$dry/stp2d.F90"
  patch -s "$dry/stp2d.F90" <"$RHS_INSTRUMENT" \
    || { printf 'REFUSE: the per-term instrument does not apply to the shipped stp2d\n' >&2
         rm -rf "$dry"; exit 67; }
  grep -q 'NEMO_L1_RHSTRM1' "$dry/stp2d.F90" \
    || { printf 'REFUSE: the patched stp2d carries no per-term writer\n' >&2
         rm -rf "$dry"; exit 67; }
  # One dump per contributing routine, in NEMO's own order.  A boundary that
  # went missing would leave one term silently unmeasured, which is the whole
  # reason this record is being acquired.
  for term in hpg ldf vor wzv keg zad; do
    grep -q "l1_rhs_open_and_dump( kt, Kbb, Kmm, Kaa, Krhs, '$term' )" \
      "$dry/stp2d.F90" \
      || { printf 'REFUSE: the per-term instrument has no %s boundary\n' \
             "$term" >&2
           rm -rf "$dry"; exit 67; }
  done
fi
if [[ -n "$SPGTS_INSTRUMENT" ]]; then
  cp "$SHIPPED_SPGTS" "$dry/dynspg_ts.F90"
  patch -s "$dry/dynspg_ts.F90" <"$SPGTS_INSTRUMENT" \
    || { printf 'REFUSE: the substep instrument does not apply to the shipped dynspg_ts\n' >&2
         rm -rf "$dry"; exit 67; }
  grep -q 'spgts_r12_open' "$dry/dynspg_ts.F90" \
    || { printf 'REFUSE: the patched dynspg_ts carries no substep writer\n' >&2
         rm -rf "$dry"; exit 67; }
  # Every boundary the record claims, in NEMO's own order inside the loop.
  # A boundary that went missing would leave one operand unmeasured, which
  # is the whole reason this record is being acquired.
  for operand in ua_ext sshp2_mid zhU ssha_e un_adv sshu_a sshp2_bck \
                 zu_spg cor_u trd_u ua_new uub_sum ssh_aa; do
    grep -q "spgts_r12_w2( '$operand'" "$dry/dynspg_ts.F90" \
      || { printf 'REFUSE: the substep instrument has no %s boundary\n' \
             "$operand" >&2
           rm -rf "$dry"; exit 67; }
  done
  spgts_syntax_dir=$dry/spgts_syntax
  mkdir -p "$spgts_syntax_dir"
  round196_fc=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
  [[ -x "$round196_fc" ]] \
    || { printf 'REFUSE: gfortran syntax checker is missing at %s\n' "$round196_fc" >&2; exit 67; }
  "$round196_fc" -J "$spgts_syntax_dir" -c "$SPGTS_STUBS" -o "$spgts_syntax_dir/stubs.o"
  "$round196_fc" -I "$spgts_syntax_dir" -J "$spgts_syntax_dir" -fsyntax-only "$SPGTS_MODULE"
  printf 'GFORTRAN_SYNTAX_PASS %s\n' "$SPGTS_MODULE"
fi
if [[ -n "$STAGE_INSTRUMENT" ]]; then
  cp "$SHIPPED_STG" "$dry/stprk3_stg.F90"
  patch -s "$dry/stprk3_stg.F90" <"$STAGE_INSTRUMENT" \
    || { printf 'REFUSE: stage-term patch does not apply to stprk3_stg.F90\n' >&2
         exit 67; }
  if [[ -n "$DYNADV_INSTRUMENT" ]]; then
    cp "$SHIPPED_DYNADV" "$dry/dynadv.F90"
    patch -s "$dry/dynadv.F90" <"$DYNADV_INSTRUMENT" \
      || { printf 'REFUSE: stage-term patch does not apply to dynadv.F90\n' >&2
           exit 67; }
  fi
  for symbol in ${STAGE_SYMBOL}_begin ${STAGE_SYMBOL}_rhs \
                ${STAGE_SYMBOL}_state ${STAGE_SYMBOL}_finish; do
    grep -q "$symbol" "$dry/stprk3_stg.F90" \
         ${DYNADV_INSTRUMENT:+"$dry/dynadv.F90"} \
      || { printf 'REFUSE: patched sources do not call %s\n' "$symbol" >&2
           exit 67; }
  done
  # Every per-term boundary this record claims, in NEMO's own stage order.
  # A boundary that went missing would leave one term silently unmeasured.
  if [[ "$variant" == "stage123flx" ]]; then
    for boundary in adv hpg vor ldf; do
      grep -q "${STAGE_SYMBOL}_rhs( '$boundary'" "$dry/stprk3_stg.F90" \
        || { printf 'REFUSE: the flux stage instrument has no %s boundary\n' \
               "$boundary" >&2; exit 67; }
    done
    for boundary in update zdf; do
      grep -q "${STAGE_SYMBOL}_state( '$boundary'" "$dry/stprk3_stg.F90" \
        || { printf 'REFUSE: the flux stage instrument has no %s boundary\n' \
               "$boundary" >&2; exit 67; }
    done
  fi
  syntax_dir=$dry/syntax
  mkdir -p "$syntax_dir"
  round192_fc=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
  [[ -x "$round192_fc" ]] \
    || { printf 'REFUSE: gfortran syntax checker is missing at %s\n' "$round192_fc" >&2; exit 67; }
  "$round192_fc" -J "$syntax_dir" -c "$STAGE_STUBS" -o "$syntax_dir/stubs.o"
  "$round192_fc" -I "$syntax_dir" -J "$syntax_dir" -fsyntax-only "$STAGE_MODULE"
  printf 'GFORTRAN_SYNTAX_PASS %s\n' "$STAGE_MODULE"
fi
# Decision 69 (operator note BG): VORTEX runs its SHIPPED simplified equation
# of state, the one narrow exception to the campaign's TEOS-10.  The eddy's
# temperature is defined by inverting this law (usrdef_istate.F90:83-88), so a
# deck that switched it would not be this experiment.  Refuse any deck that
# leaves the shipped selection or its coefficients behind.
grep -q 'ln_seos     = .true.' "$dry/namelist_cfg" \
  || { printf 'REFUSE: the patched deck does not select S-EOS (decision 69)\n' >&2
       rm -rf "$dry"; exit 67; }
if grep -qE 'ln_teos10|ln_eos80' "$dry/namelist_cfg"; then
  printf 'REFUSE: the deck selects a second equation of state alongside S-EOS\n' >&2
  rm -rf "$dry"; exit 67
fi
grep -q 'rn_a0       =  0.28' "$dry/namelist_cfg" \
  || { printf 'REFUSE: rn_a0 was dropped; usrdef_istate and the S-EOS need it\n' >&2
       rm -rf "$dry"; exit 67; }
# The card transcribes ln_zad_Aimp = .false., which this deck gets by LEAVING
# IT UNSET.  The tanks' own campaign decks set it .true.; if anyone copies that
# line in here the card and the oracle stop agreeing on the vertical momentum
# scheme, silently.  Refuse instead.
if grep -q 'ln_zad_Aimp' "$dry/namelist_cfg"; then
  printf 'REFUSE: the deck now sets ln_zad_Aimp; the card transcribes the unset default\n' >&2
  rm -rf "$dry"; exit 67
fi
# THE ONE THING THE TWO CARDS DISAGREE ON (decision 73).  NEMO counts the
# advection-form switches and stops unless EXACTLY ONE is true
# (dynadv.F90:184-190), and the vorticity routine reads that count to decide
# what the vorticity operator is handed: flux form gets Coriolis plus the metric
# term, vector form gets Coriolis plus the RELATIVE vorticity
# (dynvor.F90:855-868).  So this pair of lines IS the experiment's identity, and
# a deck that silently carried the other card's pair would run the other card
# under this card's name.  Refuse rather than discover it in the ladder.
case "$variant" in
  flux | stage123flx | res15flx | res10flx | smtflx | smtflx100d \
  | smtflxr3 | smtflx100dr3 | smtflxspgts)
      want_vec='.false.' ; want_up3='.true.'  ;;
  vec | vecrhs | stage23 | spgts | res15vec | res10vec | smtvec | smtvec100d \
  | smtvecr3 | smtvec100dr3 | smtvecspgts | smtvecrhs)
      want_vec='.true.'  ; want_up3='.false.' ;;
esac
if ! grep -qE "^ *ln_dynadv_vec *= *${want_vec//./\.}" "$dry/namelist_cfg"; then
  printf 'REFUSE: variant %s needs ln_dynadv_vec = %s\n' "$variant" "$want_vec" >&2
  rm -rf "$dry"; exit 67
fi
if ! grep -qE "^ *ln_dynadv_up3 *= *${want_up3//./\.}" "$dry/namelist_cfg"; then
  printf 'REFUSE: variant %s needs ln_dynadv_up3 = %s\n' "$variant" "$want_up3" >&2
  rm -rf "$dry"; exit 67
fi
# NEMO's own rule, applied here so a two-form deck is refused before makenemo:
# count the advection forms the deck leaves true.
forms=$(grep -cE "^ *ln_dynadv_(vec|cen2|up3) *= *\.true\." "$dry/namelist_cfg")
if [[ "$forms" -ne 1 ]]; then
  printf 'REFUSE: the deck selects %s momentum advection forms; NEMO needs exactly one\n' \
    "$forms" >&2
  rm -rf "$dry"; exit 67
fi
# Both cards run the energy-and-enstrophy vorticity; only what it is HANDED
# differs.  A deck that changed the scheme would be a third card.
if ! grep -qE "^ *ln_dynvor_een *= *\.true\." "$dry/namelist_cfg"; then
  printf 'REFUSE: both VORTEX cards require ln_dynvor_een = .true.\n' >&2
  rm -rf "$dry"; exit 67
fi
rm -rf "$dry"
python "$CHECKER" --help >/dev/null \
  || { printf 'REFUSE: the record checker does not run\n' >&2; exit 67; }
printf 'PREFLIGHT_OK  variant %s: instrument and deck patches apply to the shipped sources\n' "$variant"
printf '  reference config : %s\n  instrumented cfg : %s\n  evidence         : %s\n' \
  "$ref_cfg" "$run_cfg" "$EVIDENCE"
printf '  deck deviations  :\n'
sed -n 's/^/    /p' "$DECK" | grep -E '^\s+[-+][^-+]' || true

if [[ "$do_run" -eq 0 ]]; then
  printf '\nDRY RUN.  Re-run with --run to build and acquire.\n'
  exit 0
fi

# ------------------------------------------------------------------ acquire
if [[ "$reuse_build" -eq 1 ]]; then
  # The configurations are the CERTIFIED ones and must already exist; only the
  # evidence directory may not.
  for target in "$ref_cfg" "$run_cfg"; do
    [[ -x "$target/BLD/bin/nemo.exe" ]] \
      || { printf 'REFUSE: certified build %s is absent\n' "$target" >&2; exit 64; }
  done
  [[ ! -e "$EVIDENCE" ]] \
    || { printf 'REFUSE: target already exists: %s\n' "$EVIDENCE" >&2; exit 64; }
else
  for target in "$ref_cfg" "$run_cfg" "$EVIDENCE"; do
    if [[ -e "$target" ]]; then
      printf 'REFUSE: target already exists: %s\n' "$target" >&2
      exit 64
    fi
  done
fi
for mount in /tmp "$(dirname "$EVIDENCE")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 2097152 ]]; then
    printf 'REFUSE: %s has %s kB free, under the 2 GB floor\n' "$mount" "$free_kb" >&2
    exit 68
  fi
done

manifest=$(mktemp -d /tmp/vortex-r1-provenance.XXXXXX)
printf 'provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$round192_git_sha" >"$manifest/legoesm_git_sha.txt"
(
  cd "$SRC_CASE"
  find EXPREF MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/shipped_case.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$SRC_CASE/cpp_${TEST_CASE}.fcm" "$SHIPPED_STP" "$SHIPPED_CFG" \
  "$INSTRUMENT" "$DECK" "$CHECKER" ${RHS_INSTRUMENT:+"$RHS_INSTRUMENT"} \
  ${RHS_INSTRUMENT:+"$SHIPPED_STP2D"} \
  ${SPGTS_INSTRUMENT:+"$SPGTS_INSTRUMENT"} \
  ${SPGTS_MODULE:+"$SPGTS_MODULE"} ${SPGTS_STUBS:+"$SPGTS_STUBS"} \
  ${SPGTS_INSTRUMENT:+"$SHIPPED_SPGTS"} \
  ${STAGE_INSTRUMENT:+"$STAGE_INSTRUMENT"} \
  ${DYNADV_INSTRUMENT:+"$DYNADV_INSTRUMENT"} \
  ${STAGE_MODULE:+"$STAGE_MODULE"} ${STAGE_STUBS:+"$STAGE_STUBS"} \
  ${STAGE_INSTRUMENT:+"$SHIPPED_STG"} \
  ${DYNADV_INSTRUMENT:+"$SHIPPED_DYNADV"} >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
build_one() {          # $1 = config name, $2 = 1 to apply the instrument
  local name=$1 instrumented=$2 cfg=$NEMO_ROOT/tests/$1
  if [[ -n "$SMT_ZGR" ]]; then
    # key_vco_1d -> key_vco_1d3d is NEMO's own requirement for partial
    # steps (domzgr.F90:259 refuses l_zps under key_vco_1d; :242-243 names
    # key_vco_1d3d the "z-partial cells" key).  key_qco and key_RK3 stay.
    ./makenemo -a "$TEST_CASE" -n "$name" -m conda-scalarmath \
      del_key 'key_xios key_agrif key_vco_1d' add_key 'key_vco_1d3d'
  else
    ./makenemo -a "$TEST_CASE" -n "$name" -m conda-scalarmath \
      del_key 'key_xios key_agrif'
  fi
  # cp -r, NOT cp -a: preserved mtimes let fcm skip a patched file.
  cp -r "$SRC_CASE/EXPREF/." "$cfg/EXP00/"
  cp -r "$SRC_CASE/MY_SRC/." "$cfg/MY_SRC/"
  # The AGRIF child deck is meaningless without key_agrif; remove it so it
  # cannot be read by accident.
  rm -f "$cfg/EXP00/1_"* "$cfg/EXP00/AGRIF_FixedGrids.in"
  patch "$cfg/EXP00/namelist_cfg" <"$DECK"
  if [[ -n "$SMT_ZGR" ]]; then
    cp -f "$SMT_ZGR" "$cfg/MY_SRC/usrdef_zgr.F90"
    # The resolved key set must be exactly the shipped one with vco_1d
    # swapped for vco_1d3d -- read it back, never assume makenemo obeyed.
    grep -q 'key_vco_1d3d' "$cfg/cpp_${name}.fcm" \
      || { printf 'REFUSE: %s did not take key_vco_1d3d\n' "$name" >&2; exit 69; }
    if grep -qE '(^| )key_vco_1d( |$)' "$cfg/cpp_${name}.fcm"; then
      printf 'REFUSE: %s still carries key_vco_1d\n' "$name" >&2; exit 69
    fi
    for key in key_qco key_RK3; do
      grep -q "$key" "$cfg/cpp_${name}.fcm" \
        || { printf 'REFUSE: %s lost %s\n' "$name" "$key" >&2; exit 69; }
    done
  fi
  if [[ "$instrumented" -eq 1 ]]; then
    [[ ! -e "$cfg/MY_SRC/stprk3.F90" ]]
    cp "$SHIPPED_STP" "$cfg/MY_SRC/stprk3.F90"
    patch "$cfg/MY_SRC/stprk3.F90" <"$INSTRUMENT"
    if [[ -n "$RHS_INSTRUMENT" ]]; then
      [[ ! -e "$cfg/MY_SRC/stp2d.F90" ]]
      cp "$SHIPPED_STP2D" "$cfg/MY_SRC/stp2d.F90"
      patch "$cfg/MY_SRC/stp2d.F90" <"$RHS_INSTRUMENT"
    fi
    if [[ -n "$SPGTS_INSTRUMENT" ]]; then
      [[ ! -e "$cfg/MY_SRC/dynspg_ts.F90" ]]
      cp "$SHIPPED_SPGTS" "$cfg/MY_SRC/dynspg_ts.F90"
      cp "$SPGTS_MODULE" "$cfg/MY_SRC/vortex_r12_spgts_terms.F90"
      patch "$cfg/MY_SRC/dynspg_ts.F90" <"$SPGTS_INSTRUMENT"
    fi
    if [[ -n "$STAGE_INSTRUMENT" ]]; then
      cp "$SHIPPED_STG" "$cfg/MY_SRC/stprk3_stg.F90"
      cp "$STAGE_MODULE" "$cfg/MY_SRC/$STAGE_MODULE_NAME"
      patch "$cfg/MY_SRC/stprk3_stg.F90" <"$STAGE_INSTRUMENT"
      if [[ -n "$DYNADV_INSTRUMENT" ]]; then
        cp "$SHIPPED_DYNADV" "$cfg/MY_SRC/dynadv.F90"
        patch "$cfg/MY_SRC/dynadv.F90" <"$DYNADV_INSTRUMENT"
      fi
    fi
  fi
  touch "$cfg/MY_SRC/"*.F90
  ./makenemo -n "$name" -m conda-scalarmath
  [[ -x "$cfg/BLD/bin/nemo.exe" ]]
  # The AGRIF root arm must be the compiled one, and the usrdef routines must
  # have been carried (a stale object would silently reuse another case's).
  if grep -q 'Agrif_Root' "$cfg/BLD/ppsrc/nemo/usrdef_nam.f90"; then
    printf 'REFUSE: %s still compiles an AGRIF branch\n' "$name" >&2; exit 69
  fi
  if ! grep -q 'VORTEX' "$cfg/BLD/ppsrc/nemo/usrdef_hgr.f90"; then
    printf 'REFUSE: %s did not compile the VORTEX usrdef_hgr\n' "$name" >&2; exit 69
  fi
  if [[ -n "$SMT_ZGR" ]]; then
    grep -q 'pp_smt_H0' "$cfg/BLD/ppsrc/nemo/usrdef_zgr.f90" \
      || { printf 'REFUSE: %s compiled a usrdef_zgr without the seamount\n' "$name" >&2; exit 69; }
    grep -q 'lk_vco_1d3d = .TRUE.' "$cfg/BLD/ppsrc/nemo/dom_oce.f90" \
      || { printf 'REFUSE: %s did not resolve lk_vco_1d3d\n' "$name" >&2; exit 69; }
  fi
  if grep -q 'NEMO_L1_ENTRY_1' "$cfg/BLD/ppsrc/nemo/stprk3.f90"; then
    if [[ "$instrumented" -ne 1 ]]; then
      printf 'REFUSE: the REFERENCE build carries the writer\n' >&2; exit 69
    fi
  elif [[ "$instrumented" -eq 1 ]]; then
    printf 'REFUSE: the writer is absent from %s ppsrc (stale build)\n' \
      "$name" >&2; exit 69
  fi
  if [[ -n "$SPGTS_INSTRUMENT" && "$instrumented" -eq 1 ]]; then
    grep -q 'spgts_r12_open' "$cfg/BLD/ppsrc/nemo/dynspg_ts.f90" \
      || { printf 'REFUSE: substep calls are absent from compiled dynspg_ts\n' >&2; exit 69; }
  fi
  if [[ -n "$STAGE_INSTRUMENT" && "$instrumented" -eq 1 ]]; then
    grep -q "${STAGE_SYMBOL}_begin" "$cfg/BLD/ppsrc/nemo/stprk3_stg.f90" \
      || { printf 'REFUSE: stage-term calls are absent from compiled stprk3_stg\n' >&2; exit 69; }
    if [[ -n "$DYNADV_INSTRUMENT" ]]; then
      grep -q "${STAGE_SYMBOL}_rhs" "$cfg/BLD/ppsrc/nemo/dynadv.f90" \
        || { printf 'REFUSE: stage-term calls are absent from compiled dynadv\n' >&2; exit 69; }
    fi
  fi
  # `nm | grep -q` would report CLEAN if nm itself failed, so capture first
  # and require nm to have succeeded before believing the grep.
  local symbols
  symbols=$(nm -D "$cfg/BLD/bin/nemo.exe") || {
    printf 'REFUSE: cannot read symbols from %s\n' "$name" >&2; exit 65; }
  if printf '%s' "$symbols" | grep -q '_ZGV'; then
    printf 'REFUSE: vector-math symbol present in %s\n' "$name" >&2
    exit 65
  fi
}

run_one() {            # $1 = config name, $2 = run directory
  local cfg=$NEMO_ROOT/tests/$1 dir=$2
  mkdir -p "$dir"
  cp -L "$cfg/EXP00/namelist_cfg" "$cfg/EXP00/namelist_ref" "$dir/"
  for xml in "$cfg"/EXP00/*.xml; do
    [[ -e "$xml" ]] && cp -L "$xml" "$dir/"
  done
  cp "$cfg/BLD/bin/nemo.exe" "$dir/nemo"
  (
    cd "$dir"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.log
    # No pipe: under `set -o pipefail` a failing mpirun aborts the script
    # before any ${PIPESTATUS} line could be read, so such a guard would be
    # unreachable and would prove nothing.  Redirect, then show the tail.
    mpirun -np 1 --oversubscribe ./nemo >>run.user.log 2>&1
    tail -n 20 run.user.log
    printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >>run.user.log
  )
  [[ -f "$dir/$RESTART" ]] \
    || { printf 'REFUSE: %s wrote no step-%d restart\n' "$1" "$STEPS" >&2; exit 71; }
  if [[ "$variant" == "stage23" || "$variant" == "stage123flx" ]] && ! grep -q 'ln_tile    =  F' "$dir/ocean.output"; then
    printf 'REFUSE: stage-term writer requires the resolved VORTEX non-tiled branch\n' >&2
    exit 71
  fi
}

run_one_reuse() {       # $1 = certified config name, $2 = run directory
  # The 30 km EXP00 inside the certified configuration is PART OF THE CERTIFIED
  # RECORD and is not touched.  The run directory is assembled from the shipped
  # EXPREF and the resolution deck patch, so the only difference between this
  # rung and the certified one is the namelist.
  local cfg=$NEMO_ROOT/tests/$1 dir=$2
  mkdir -p "$dir"
  cp -L "$SRC_CASE"/EXPREF/*.xml "$dir/"
  cp -L "$SRC_CASE/EXPREF/namelist_cfg" "$SRC_CASE/EXPREF/namelist_ref" "$dir/"
  rm -f "$dir"/1_* "$dir/AGRIF_FixedGrids.in"
  patch "$dir/namelist_cfg" <"$DECK"
  cp "$cfg/BLD/bin/nemo.exe" "$dir/nemo"
  (
    cd "$dir"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.log
    mpirun -np 1 --oversubscribe ./nemo >>run.user.log 2>&1
    tail -n 20 run.user.log
    printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >>run.user.log
  )
  [[ -f "$dir/$RESTART" ]] \
    || { printf 'REFUSE: %s wrote no step-%d restart\n' "$1" "$STEPS" >&2; exit 71; }
}

if [[ "$reuse_build" -eq 1 ]]; then
  # PROVE the reuse: these must be the very executables the certified 30 km
  # records were produced with, so the rung differs in the deck ALONE.
  certified_manifest=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex
  case "$variant" in
    res15flx | res10flx) certified_manifest=$certified_manifest/round2/binaries.sha256 ;;
    res15vec | res10vec) certified_manifest=$certified_manifest/round3/binaries.sha256 ;;
    smtflx100d) certified_manifest=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/VORTEX_SMT_OMIP_L1_P3/kt1_10/binaries.sha256 ;;
    smtvec100d) certified_manifest=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/VORTEX_SMT_VEC_R8_OMIP_L1_P3/kt1_10/binaries.sha256 ;;
    smtflx100dr3) certified_manifest=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round3/VORTEX_SMT_R3_OMIP_L1_P3/kt1_10/binaries.sha256 ;;
    smtvec100dr3) certified_manifest=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round3/VORTEX_SMT_R3_VEC_R8_OMIP_L1_P3/kt1_10/binaries.sha256 ;;
  esac
  [[ -f "$certified_manifest" ]] \
    || { printf 'REFUSE: certified binary manifest %s is absent\n' \
           "$certified_manifest" >&2; exit 65; }
  for exe in "$ref_cfg/BLD/bin/nemo.exe" "$run_cfg/BLD/bin/nemo.exe"; do
    # The manifest may record either spelling of this tree (/home/dbalwada is a
    # symlink to /data/abyssal/dbalwada), so match on the path BELOW tests/,
    # which is the same in both.
    want=$(awk -v t="/tests/${exe#*/tests/}" 'index($2, t) {print $1}' \
             "$certified_manifest")
    [[ -n "$want" ]] \
      || { printf 'REFUSE: %s is not named in %s\n' "$exe" "$certified_manifest" >&2
           exit 65; }
    got=$(sha256sum "$exe" | awk '{print $1}')
    [[ "$got" == "$want" ]] \
      || { printf 'REFUSE: %s is not the certified executable (%s != %s)\n' \
             "$exe" "$got" "$want" >&2; exit 65; }
    printf 'CERTIFIED_BUILD_REUSED %s %s\n' "$exe" "$got"
  done
  # The instrumented build must still carry the writer and the reference must
  # still not: the passivity premise is re-proved, never assumed.
  grep -q 'NEMO_L1_ENTRY_1' "$run_cfg/BLD/ppsrc/nemo/stprk3.f90" \
    || { printf 'REFUSE: the reused instrumented build carries no writer\n' >&2; exit 69; }
  if grep -q 'NEMO_L1_ENTRY_1' "$ref_cfg/BLD/ppsrc/nemo/stprk3.f90"; then
    printf 'REFUSE: the reused REFERENCE build carries the writer\n' >&2; exit 69
  fi
else
build_one "$REF_CFG" 0
build_one "$RUN_CFG" 1
fi
sha256sum "$ref_cfg/BLD/bin/nemo.exe" "$run_cfg/BLD/bin/nemo.exe" \
  >"$manifest/binaries.sha256"
if [[ "$reuse_build" -eq 1 ]]; then
  run_one_reuse "$REF_CFG" "$EVIDENCE/reference"
  run_one_reuse "$RUN_CFG" "$EVIDENCE"
else
  # The two decks must be the SAME deck; only the compiled writer may differ.
  cmp "$ref_cfg/EXP00/namelist_cfg" "$run_cfg/EXP00/namelist_cfg"

  run_one "$REF_CFG" "$EVIDENCE/reference"
  run_one "$RUN_CFG" "$EVIDENCE"
fi
# Both arms of the rung must have read the SAME deck; only the compiled writer
# may differ between them.
cmp "$EVIDENCE/reference/namelist_cfg" "$EVIDENCE/namelist_cfg"
cp "$manifest"/*.sha256 "$EVIDENCE/"
cp "$manifest/legoesm_git_sha.txt" "$EVIDENCE/"

# ADMISSION.  The checker parses every record's own header (note BD) and
# refuses unless the two restarts are byte-identical (note AS).  Its plant
# MUST turn it red, or it proves nothing.
if [[ -n "$RHS_INSTRUMENT" ]]; then RHS_FLAG=--rhs-terms; else RHS_FLAG=; fi
if [[ -n "$STAGE_INSTRUMENT" ]]; then STAGE_FLAG=$STAGE_FLAG_NAME; else STAGE_FLAG=; fi
if [[ -n "$SPGTS_INSTRUMENT" ]]; then SPGTS_FLAG=--spgts-terms; else SPGTS_FLAG=; fi
python "$CHECKER" --run-dir "$EVIDENCE" --reference-dir "$EVIDENCE/reference" \
  --restart "$RESTART" --steps "$RECORD_STEPS" ${RHS_FLAG:+$RHS_FLAG} \
  ${STAGE_FLAG:+$STAGE_FLAG} ${SPGTS_FLAG:+$SPGTS_FLAG} \
  --output "$EVIDENCE/vortex_${TAG}_admission.json"
# Every plant the checker offers must turn it red.  One plant proves one
# guard; the record is only admissible if each guard the round relies on is
# shown to be able to fail.
if [[ -n "$SPGTS_INSTRUMENT" ]]; then
  plants=(header field-name truncated missing-frame)
else
  plants=(header)
fi
for plant in "${plants[@]}"; do
  if python "$CHECKER" --run-dir "$EVIDENCE" --reference-dir "$EVIDENCE/reference" \
       --restart "$RESTART" --steps "$RECORD_STEPS" ${RHS_FLAG:+$RHS_FLAG} \
       ${STAGE_FLAG:+$STAGE_FLAG} ${SPGTS_FLAG:+$SPGTS_FLAG} --plant "$plant" \
       >"$EVIDENCE/vortex_${TAG}_admission_plant_${plant}.json" 2>&1; then
    printf 'REFUSE: the %s plant did not turn the checker red\n' "$plant" >&2
    exit 70
  fi
  printf 'PLANT_FIRED %s\n' "$plant"
done
(
  cd "$EVIDENCE"
  sha256sum oracle_*.bin vortex_${TAG}_admission.json "$RESTART" mesh_mask.nc \
    legoesm_git_sha.txt >vortex_${TAG}_outputs.sha256
)
printf 'VORTEX_%s_KT1_10_ORACLE_READY %s\n' "$TAG" "$EVIDENCE"
