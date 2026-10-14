#!/usr/bin/env bash
# Grid ladder G4 ACQUISITION (DECISION 117) -- OPERATOR-EXECUTED ONLY.
#
#   preflight (default):  run.sh
#   acquire:              run.sh --run
#   re-admit:             run.sh --admit-existing
#
# The agent that wrote this ran neither makenemo nor NEMO.
#
# WHAT THIS ACQUIRES.  The certified G3 program (ORCA2's shipped 31-level
# partial-cell domain, ORCA2's initial T/S climatology without damping,
# TSUNAMI's ssh bump at (ji,jj) = (51,74), no ice, advection / lateral mixing
# / drag OFF, split-explicit free surface, EEN, constant vertical mixing,
# rn_Dt = 10800) with ONE switch: ORCA2's surface boundary condition
# (&namsbc :80-92 with nn_ice = 0, and ORCA2's &namsbc_blk, &namtra_qsr,
# &namsbc_ssr, &namsbc_rnf verbatim).  Every forcing file NEMO opens under
# this deck is ORCA2's, copied byte for byte and pinned by SHA-256.
#
# NO BUILD.  ORCA2_OMIP_L4 compiles key_si3 key_qco key_vco_1d3d key_RK3; the
# G1 binaries compile the same keys without key_si3, and nn_ice = 0 never
# reaches the ice code.  The bulk, runoff, restoring, light and fldread
# sources are compiled in every build (no cpp key), and no G1 MY_SRC file
# touches them (usrdef_sbc is called only under ln_usr = T).  So the two
# round-4 G1 binaries are reused, pinned:
#   ORCA2G1_OMIP_L1_R4REF  the reference binary (no writer);
#   ORCA2G1_OMIP_L1_P5     the record binary (stage frames at kt = 1..50).
# The frames hold no surface flux.  NEMO's RK3 restarts hold the surface
# tracer fluxes (trasbc.f90:175-176 sbc_hc_b/sbc_sc_b, traqsr.f90:246
# fraqsr_1lev) and the mean surface fields (sbcssm.f90:185-192), not the
# wind stress (sbcmod.f90:590 writes it only under lk_SWE).
#
# THE RECORD LENGTH IS NEMO'S.  G3 stopped at kt = 11 (stp_ctl, |V| > 10
# m/s); G4 adds forcing and no dissipation, so the reference (100-step
# deck) is run FIRST and the record is the steps it completes: STEPS =
# min(10, completed), rounded down to a multiple of nn_fsbc = 2.  A stop
# anywhere but stp_ctl is refused.  Twins (record binary, nn_itend = STEPS):
# a (restart STEPS), b (even restarts 2..STEPS), c (odd restarts 3..STEPS-1;
# G2 round 3: a list of CONSECUTIVE steps never opens, restart.f90:119,
# :188, :199-202).  The reference's own list is 2,4,..,20 so it holds the
# step-STEPS restart whatever STEPS is.
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE: unhandled command failure at run.sh line %s (exit %s)\n' \
    "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_NAME=ORCA2G1_OMIP_L1_R4REF RUN_NAME=ORCA2G1_OMIP_L1_P5
readonly REF_SHA=5ed4b6e543f32b827ee42e818704f836dd322768a17bbe4f40e8ce5a7ff523b3
readonly RUN_SHA=9b96f1427503da2773ce6c17621dd431c3261792d11a39c218f64d88e3f00778
readonly WANT_KEYS='key_RK3 key_qco key_vco_1d3d'
readonly ORCA2_KEYS='key_RK3 key_qco key_si3 key_vco_1d3d'
readonly STAGE_STEPS=50 SPGTS_STEPS=1,2,17,31,41     # the record binary's own gates
readonly RECORD_SPGTS=1,2                             # what a >= 2-step record holds
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round17/oracle_g4_r17}
readonly SHIPPED_DOMCFG=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/ORCA_R2_zps_domcfg.nc
readonly SHIPPED_DOMCFG_SHA=7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839
readonly INPUTS=$(dirname "$SHIPPED_DOMCFG")
readonly NAMELIST_REF_SHA=b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd
readonly G3_DECK_SHA=5b2699e8e20c49bd0b8cce6a8dfcf3286821001a2079a3318a6a0f77fed215a6
# Every file NEMO opens under this deck besides the domain file, as name:SHA-256.
readonly PINNED_FILES=(
  data_1m_potential_temperature_nomask.nc:ca00905c27078305e80130ea6dd07fc575de2ae3257fad38003b329458cf7f7a
  data_1m_salinity_nomask.nc:ad648d972f0631bde7e1b598d98470a979b7f2649271fc9cf5ccfca158e31d0c
  u_10.15JUNE2009_fill.nc:219bf3a6891dfe25871ba9d739e8ad465e12934b62f6524422a985b3858ae96b
  v_10.15JUNE2009_fill.nc:b3913b212b6a194fa8f79214ea56819728cbef94a06c6a82e94d65337d6e7b3a
  ncar_rad.15JUNE2009_fill.nc:050ee263198d49ce285070bc4d3e75953091c61c8f6a0d992484e79974fc37a7
  t_10.15JUNE2009_fill.nc:bee5df15a69d757f3911660910efc54c936555ec8bce742215ec047a16a4338b
  q_10.15JUNE2009_fill.nc:4c46f4527e8e5c1f3cc12942a87d0938c10431fcdb56cff7d25a9b5db2e69806
  ncar_precip.15JUNE2009_fill.nc:ed17b5069ddb13cd2a596079a0102e7792899e5dbe4ce1ba5a20e29802b5337a
  slp.15JUNE2009_fill.nc:19dea3069caa8f513218c7cbc11b3dce83d7cb9363e44b787e892917625c81fb
  weights_core2_orca2_bicub.nc:2cfefd58b0231f355cf999f943a48586f63e4e9264f0cbcc7197334f12b3e7e8
  weights_core2_orca2_bilin.nc:e5fe31582615262d84b06d01aeaa1893f4d942dc978aebc3e13f56cda484d766
  chlorophyll.nc:9178f44245f9465c584168eeb47b28e83c358a9ef78ed614cd623c4456ad7890
  sss_data.nc:ea4609d6deca4ccf95b6307a7b7eed1bff3b767146eb69c2a82762824bf1ed49
  runoff_core_monthly.nc:b92ff46e4c406096a1e1fe3815dfdcce12b234e84541a3b6dffb1037e777d4eb
)

mode=--preflight-only
case "${1:---preflight-only}" in
  --run|--preflight-only|--admit-existing) mode=${1:---preflight-only} ;;
  *) printf 'Usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run.sh is not inside a git worktree\n' >&2; exit 66; }
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: legoESM worktree is dirty; commit the exact acquisition tool first\n' >&2
  exit 66
fi
readonly COMMIT=$(git -C "$repo_root" rev-parse HEAD)

readonly DECK=$here/namelist_cfg
readonly G3_DECK=$here/../nemo_testcase_g3_orca2ts/namelist_cfg
readonly G1_DIR=$here/../nemo_testcase_g1_orca2grid
readonly CHECKER=$G1_DIR/check_records.py
readonly STEP_PATCH=$G1_DIR/stprk3_g1_step_record_stage50.patch   # the record binary's writer
readonly REF_BIN=$NEMO_ROOT/cfgs/$REF_NAME/BLD/bin/nemo.exe
readonly RUN_BIN=$NEMO_ROOT/cfgs/$RUN_NAME/BLD/bin/nemo.exe
readonly ORCA2=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4

for f in "$DECK" "$G3_DECK" "$CHECKER" "$STEP_PATCH" "$SHIPPED_DOMCFG" "$REF_BIN" "$RUN_BIN" \
         "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$ORCA2/EXP00/namelist_cfg" "$ORCA2/cpp_ORCA2_OMIP_L4.fcm"; do
  [[ -f "$f" ]] || { printf 'REFUSE: missing input %s\n' "$f" >&2; exit 67; }
done
sha_of() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(sha_of "$SHIPPED_DOMCFG")" == "$SHIPPED_DOMCFG_SHA" ]] \
  || { printf 'REFUSE: the shipped ORCA2 domain file changed\n' >&2; exit 67; }
for entry in "${PINNED_FILES[@]}"; do
  [[ -f "$INPUTS/${entry%%:*}" && "$(sha_of "$INPUTS/${entry%%:*}")" == "${entry#*:}" ]] \
    || { printf 'REFUSE: ORCA2 input %s is absent or changed\n' "${entry%%:*}" >&2; exit 67; }
done
[[ "$(sha_of "$REF_BIN")" == "$REF_SHA" && "$(sha_of "$RUN_BIN")" == "$RUN_SHA" ]] \
  || { printf 'REFUSE: a G1 binary is not the round-4 one (rebuilt?)\n' >&2; exit 67; }
[[ "$(sha_of "$NEMO_ROOT/cfgs/SHARED/namelist_ref")" == "$NAMELIST_REF_SHA" ]] \
  || { printf 'REFUSE: namelist_ref changed (every unstated forcing value resolves there)\n' >&2; exit 67; }
[[ "$(sha_of "$G3_DECK")" == "$G3_DECK_SHA" ]] \
  || { printf 'REFUSE: the G3 deck is not the certified one\n' >&2; exit 67; }

sorted_keys() { sed -E 's/.*fppkeys *//' "$1" | xargs -n1 | LC_ALL=C sort | xargs; }
[[ "$(sorted_keys "$ORCA2/cpp_ORCA2_OMIP_L4.fcm")" == "$ORCA2_KEYS" ]] \
  || { printf 'REFUSE: ORCA2_OMIP_L4 keys are not "%s"\n' "$ORCA2_KEYS" >&2; exit 68; }
# The binaries are G1's program: same keys (ORCA2's minus key_si3), the bump,
# and the writer only in the record binary with the gates named above.
for name in "$REF_NAME" "$RUN_NAME"; do
  pp=$NEMO_ROOT/cfgs/$name/BLD/ppsrc/nemo
  [[ "$(sorted_keys "$NEMO_ROOT/cfgs/$name/cpp_$name.fcm")" == "$WANT_KEYS" ]] \
    || { printf 'REFUSE: %s keys are not "%s"\n' "$name" "$WANT_KEYS" >&2; exit 68; }
  grep -q 'ORCA2G1: the ssh bump' "$pp/restart.f90" \
    && grep -q 'np_g1_ji0 = 51' "$pp/usrdef_istate.f90" \
    && grep -q 'np_g1_jj0 = 74' "$pp/usrdef_istate.f90" && grep -q 'CALL stp_RK3' "$pp/nemogcm.f90" \
    || { printf 'REFUSE: %s is not the G1 program\n' "$name" >&2; exit 68; }
  # no MY_SRC file reaches the forcing path
  for src in "$NEMO_ROOT/cfgs/$name/MY_SRC"/*.F90; do
    case "$(basename "$src")" in
      restart.F90|usrdef_istate.F90|usrdef_sbc.F90|stprk3.F90|dynspg_ts.F90|g1_step_record.F90|g1_spgts_record.F90) ;;
      *) printf 'REFUSE: %s carries an unexpected MY_SRC file %s\n' "$name" "$src" >&2; exit 68 ;;
    esac
  done
done
[[ "$(grep -c 'g1r_open' "$NEMO_ROOT/cfgs/$REF_NAME/BLD/ppsrc/nemo/stprk3.f90" || true)" -eq 0 ]] \
  || { printf 'REFUSE: the reference binary carries the writer\n' >&2; exit 68; }
pp=$NEMO_ROOT/cfgs/$RUN_NAME/BLD/ppsrc/nemo
[[ "$(grep -c "kstp <= nit000 + $((STAGE_STEPS - 1)) )" "$pp/stprk3.f90" || true)" -eq 5 ]] \
  || { printf 'REFUSE: the record binary does not write stage frames at kt = 1..%s\n' "$STAGE_STEPS" >&2; exit 68; }
gate=$(grep -oE 'kstp /= [0-9]+' "$pp/g1_spgts_record.f90" | awk '{print $3}' | paste -sd, -)
[[ "$gate" == "$SPGTS_STEPS" ]] \
  || { printf 'REFUSE: the substep writer opens at "%s", not "%s"\n' "$gate" "$SPGTS_STEPS" >&2; exit 68; }

# ONE switch.  Outside the five forcing groups the G4 deck's assignments are
# G3's; the four data groups are ORCA2's assignments; &namsbc is ORCA2's
# switches with nn_ice = 0 and every other key at namelist_ref's value (the
# committed test checks that resolution against namelist_ref).
readonly FORCING_GROUPS='namsbc namsbc_blk namtra_qsr namsbc_ssr namsbc_rnf'
assignments() { sed -e 's/!.*//' -e 's/[[:space:]]*$//' "$1" | grep -v '^$'; }
group() { assignments "$1" | awk -v g="&$2" '$1==g{p=1} p{print} p&&/^\//{p=0}' | tr -d ' \t'; }
outside_forcing() {
  local text g
  text=$(assignments "$1")
  for g in $FORCING_GROUPS; do
    text=$(awk -v g="&$g" '$1==g{p=1} !p{print} p&&/^\//{p=0}' <<<"$text")
  done
  printf '%s\n' "$text"
}
cmp -s <(outside_forcing "$DECK") <(outside_forcing "$G3_DECK") \
  || { printf 'REFUSE: the G4 deck assigns something outside the forcing groups that G3 does not\n' >&2; exit 68; }
for g in namsbc_blk namtra_qsr namsbc_ssr namsbc_rnf; do
  [[ -n "$(group "$DECK" "$g")" && "$(group "$DECK" "$g")" == "$(group "$ORCA2/EXP00/namelist_cfg" "$g")" ]] \
    || { printf 'REFUSE: the G4 &%s is not ORCA2s own\n' "$g" >&2; exit 68; }
  [[ -z "$(group "$G3_DECK" "$g")" ]] \
    || { printf 'REFUSE: the G3 deck states &%s\n' "$g" >&2; exit 68; }
done
readonly WANT_NAMSBC='&namsbc|nn_fsbc=2|ln_usr=.false.|ln_flx=.false.|ln_blk=.true.|ln_abl=.false.|ln_cpl=.false.|ln_mixcpl=.false.|nn_components=0|nn_ice=0|ln_ice_embd=.false.|ln_traqsr=.true.|ln_dm2dc=.false.|ln_ssr=.true.|nn_fwb=2|ln_rnf=.true.|ln_apr_dyn=.false.|ln_wave=.false.|nn_lsm=0|/'
[[ "$(group "$DECK" namsbc | paste -sd'|' -)" == "$WANT_NAMSBC" ]] \
  || { printf 'REFUSE: the G4 &namsbc is not ORCA2s switches with nn_ice = 0\n' >&2; exit 68; }

render_deck() {        # $1 = target, $2 = nn_itend, $3 = nn_stocklist
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
         -e "s/^(   nn_stock    = ).*/\1    $2/" \
         -e "s/^(   nn_stocklist = ).*/\1$3/" "$DECK" >"$1"
  local moved
  moved=$(diff "$DECK" "$1" | grep -c '^>' || true)
  [[ "$moved" -eq 0 || "$moved" -eq 3 || "$moved" -eq 1 ]] \
    || { printf 'REFUSE: rendering changed %s deck lines, not the run-length lines\n' "$moved" >&2; exit 68; }
}
dry=$(mktemp -d)
trap 'rm -rf "$dry"' EXIT
readonly L_SMOKE='2,0,0,0,0,0,0,0,0,0' L_FULL='10,20,30,40,50,60,70,80,90,100'
readonly L_REF='2,4,6,8,10,12,14,16,18,20'
# the three twin lists for a record of $1 steps (even, >= 2)
list_a() { printf '%s,0,0,0,0,0,0,0,0,0' "$1"; }
pad10() { local l=("$@"); while ((${#l[@]} < 10)); do l+=(0); done; local IFS=,; printf '%s' "${l[*]}"; }
list_b() { pad10 $(seq 2 2 "$1"); }
list_c() { (($1 >= 4)) && pad10 $(seq 3 2 $(($1 - 1))) || pad10 0; }
render_deck "$dry/full" 100 "$L_FULL"
cmp -s "$dry/full" "$DECK" \
  || { printf 'REFUSE: the committed deck is not the 100-step deck\n' >&2; exit 68; }
render_deck "$dry/smoke" 2 "$L_SMOKE"
render_deck "$dry/ref" 100 "$L_REF"
for steps in 2 4 6 8 10; do
  render_deck "$dry/a$steps" "$steps" "$(list_a "$steps")"
  render_deck "$dry/b$steps" "$steps" "$(list_b "$steps")"
  render_deck "$dry/c$steps" "$steps" "$(list_c "$steps")"
done
[[ "$(list_b 10)" == '2,4,6,8,10,0,0,0,0,0' && "$(list_c 10)" == '3,5,7,9,0,0,0,0,0,0' ]] \
  || { printf 'REFUSE: the twin restart lists are not G3s at 10 steps\n' >&2; exit 68; }
python3 -I -m py_compile "$CHECKER"
printf 'PREFLIGHT: binaries, domain and %s ORCA2 forcing/climatology files pinned; keys "%s" (ORCA2 minus key_si3); deck = G3 plus ORCA2 forcing, nn_ice = 0; commit %s\n' \
  "${#PINNED_FILES[@]}" "$WANT_KEYS" "$COMMIT"

readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly TWIN_A=$EVIDENCE/p3_a TWIN_B=$EVIDENCE/p3_b TWIN_C=$EVIDENCE/p3_c

# NEMO's own log: T/S read with damping off (dtatsd.f90), and the forcing
# switches as resolved (sbcmod.f90:183-215 prints).
logged() {
  local o=$1/ocean.output
  grep -Eq 'ln_tsd_init *= *T' "$o" && grep -Eq 'ln_tsd_dmp *= *F' "$o" \
    && grep -q 'dta_tsd: deallocte T & S arrays' "$o" \
    && grep -Eq 'ln_usr +=  F' "$o" && grep -Eq 'ln_blk +=  T' "$o" \
    && grep -Eq 'nn_ice +=  +0$' "$o" && grep -Eq 'ln_traqsr +=  T' "$o" \
    && grep -Eq 'ln_ssr +=  T' "$o" && grep -Eq 'nn_fwb +=  +2$' "$o" \
    && grep -Eq 'ln_rnf +=  T' "$o" && grep -Eq 'jpkglo *= *31' "$o"
}

# The reference's completed steps: NEMO's own boundary.  Finished = 100; a
# stop is admitted only from stp_ctl (stpctl.f90), with its abort state.
completed_steps() {
  local t
  t=$(tr -d ' \n' <"$REF/time.step")
  if grep -qx 'RUN_DONE' "$REF/run.user.log"; then
    [[ "$t" == 100 ]] && ! grep -q 'E R R O R' "$REF/ocean.output" \
      || { printf 'REFUSE: the reference finished at %s, not 100, or with an error\n' "$t" >&2; exit 73; }
    printf '100'; return
  fi
  grep -q 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$REF/ocean.output" && [[ -f "$REF/output.abort.nc" ]] \
    || { printf 'REFUSE: the reference stopped at kt = %s outside stp_ctl\n' "$t" >&2; exit 73; }
  printf '%s' $((t - 1))
}
record_steps() {       # min(10, completed), down to a multiple of nn_fsbc = 2
  local c s
  c=$(completed_steps)
  s=$(( c < 10 ? c : 10 )); s=$(( s - s % 2 ))
  ((s >= 2)) || { printf 'REFUSE: NEMO completed %s steps; no record of 2 steps exists\n' "$c" >&2; exit 73; }
  printf '%s' "$s"
}

# one run directory, judged: pinned domain, binary and inputs; the committed
# deck at its length and list; NEMO's log; every listed restart present
judge_dir() {          # $1 = dir, $2 = binary sha, $3 = list
  local dir=$1 step entry
  [[ -f "$dir/domain_cfg.nc" && "$(sha_of "$dir/domain_cfg.nc")" == "$SHIPPED_DOMCFG_SHA" ]] \
    || { printf 'REFUSE: %s does not hold the shipped ORCA2 domain file\n' "$dir" >&2; exit 73; }
  [[ -f "$dir/namelist_ref" && "$(sha_of "$dir/namelist_ref")" == "$NAMELIST_REF_SHA" ]] \
    || { printf 'REFUSE: %s ran another namelist_ref\n' "$dir" >&2; exit 73; }
  [[ -f "$dir/nemo" && "$(sha_of "$dir/nemo")" == "$2" ]] \
    || { printf 'REFUSE: %s was not run with its pinned binary\n' "$dir" >&2; exit 73; }
  for entry in "${PINNED_FILES[@]}"; do
    [[ -f "$dir/${entry%%:*}" && "$(sha_of "$dir/${entry%%:*}")" == "${entry#*:}" ]] \
      || { printf 'REFUSE: %s does not hold the pinned %s\n' "$dir" "${entry%%:*}" >&2; exit 73; }
  done
  logged "$dir" || { printf 'REFUSE: %s ocean.output does not state the G4 switches\n' "$dir" >&2; exit 73; }
  for step in ${3//,/ }; do
    [[ "$step" == 0 || "$step" -gt "$(tr -d ' \n' <"$dir/time.step")" \
       || -f "$dir/ORCA2G1_$(printf '%08d' "$step")_restart.nc" ]] \
      || { printf 'REFUSE: %s has no restart at listed step %s\n' "$dir" "$step" >&2; exit 73; }
  done
}

admit() {
  local steps plant dir bin len list
  steps=$(record_steps)
  printf 'G4_BOUNDARY reference completed %s steps; record = kt 1..%s\n' "$(completed_steps)" "$steps"
  grep -E '^ *kt +[0-9]+ \|' "$REF/ocean.output" | tail -6 || true      # reported, not judged
  [[ -f "$REF/ORCA2G1_$(printf '%08d' "$steps")_restart.nc" ]] \
    || { printf 'REFUSE: the reference has no step-%s restart\n' "$steps" >&2; exit 73; }
  for dir in "$SMOKE:$REF_SHA:2:$L_SMOKE" "$REF:$REF_SHA:100:$L_REF" \
             "$TWIN_A:$RUN_SHA:$steps:$(list_a "$steps")" "$TWIN_B:$RUN_SHA:$steps:$(list_b "$steps")" \
             "$TWIN_C:$RUN_SHA:$steps:$(list_c "$steps")"; do
    IFS=: read -r dir bin len list <<<"$dir"
    judge_dir "$dir" "$bin" "$list"
    render_deck "$dry/admit_deck" "$len" "$list"
    cmp -s "$dir/namelist_cfg" "$dry/admit_deck" \
      || { printf 'REFUSE: %s ran a deck that is not the committed one at its length and list\n' "$dir" >&2; exit 73; }
    [[ "$dir" == "$REF" ]] && continue
    grep -qx 'RUN_DONE' "$dir/run.user.log" && [[ "$(tr -d ' \n' <"$dir/time.step")" == "$len" ]] \
      && ! grep -q 'E R R O R' "$dir/ocean.output" \
      || { printf 'REFUSE: %s is not a finished %s-step run\n' "$dir" "$len" >&2; exit 73; }
  done
  for plant in truncation nonfinite missing-frame twin-ulp missing-step missing-substep \
               twin-substep-inner; do
    if python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" \
         --steps "$steps" --stage-steps "$steps" --plant "$plant" \
         --substep-steps "$RECORD_SPGTS" >"$EVIDENCE/manifest/plant_${plant}.log" 2>&1; then
      printf 'REFUSE: record plant %s stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/manifest/plant_${plant}.log" \
      || { printf 'REFUSE: record plant %s did not fire\n' "$plant" >&2; exit 72; }
  done
  python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" --reference "$REF" \
    --steps "$steps" --stage-steps "$steps" --restart-steps "$steps" \
    --substep-steps "$RECORD_SPGTS" --json "$EVIDENCE/g4_round17_admission.json"
  python3 -I - "$G1_DIR" "$TWIN_A" "$TWIN_B" "$STEP_PATCH" "$EVIDENCE/g4_round17_admission.json" \
      "$TWIN_C" "$SMOKE" "$steps" <<'PY'
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import check_records
steps = int(sys.argv[8])
try:
    frame, stage = None, set()
    for line in Path(sys.argv[4]).read_text().splitlines():
        if not line.startswith("+"):
            continue
        hit = re.search(r"CALL g1r_frame\( '(.)' \)", line)
        frame = hit.group(1) if hit else frame
        hit = re.search(r"CALL g1r_w2\( '(\w+)'", line)
        if hit:
            stage.add(f"{frame}_{hit.group(1)}")
    admitted = json.loads(Path(sys.argv[5]).read_text())
    check_records.require(len(stage) > 40, "the writer patch was not parsed")
    check_records.require(admitted["fields_stage"] == sorted(stage),
                          "a stage step lacks a field the writer writes")
    check_records.require(admitted["fields_final_only"] == [],
                          "a record within kt <= 50 holds a final-only step")
    check_records.restarts_identical(Path(sys.argv[2]), Path(sys.argv[3]), [steps])
    # the record binary's step 2 is the reference binary's smoke step 2
    check_records.restarts_identical(Path(sys.argv[7]), Path(sys.argv[3]), [2])
    for step in range(2, steps + 1):
        twin = Path(sys.argv[3] if step % 2 == 0 else sys.argv[6])
        check_records.require((twin / f"ORCA2G1_{step:08d}_restart.nc").is_file(),
                              f"no twin restart at step {step}")
except check_records.Refuse as exc:
    sys.exit(f"REFUSE: {exc}")
print("G4_FRAME_FIELDS_COMPLETE", len(stage), "per stage step")
print(f"G4_TWIN_RESTARTS_PASSIVE step {steps} b = a, step 2 b = smoke; every step 2..{steps} present")
PY
  local kt twin
  for twin in "$TWIN_B" "$TWIN_C"; do
    for kt in $(seq 1 "$steps"); do
      cmp -s "$TWIN_A/oracle_g1step_kt$(printf '%08d' "$kt").bin" \
             "$twin/oracle_g1step_kt$(printf '%08d' "$kt").bin" \
        || { printf 'REFUSE: %s step frame kt=%s differs from twin a\n' "$twin" "$kt" >&2; exit 73; }
    done
  done
  printf 'G4_TWIN_FRAMES_IDENTICAL kt=1..%s (twins b, c)\n' "$steps"
  shopt -s nullglob
  ( cd "$EVIDENCE" && sha256sum g4_round17_admission.json p3_a/mesh_mask.nc \
      p3_a/ORCA2G1_$(printf '%08d' "$steps")_restart.nc p3_b/ORCA2G1_*_restart.nc \
      p3_c/ORCA2G1_*_restart.nc smoke_2step/ORCA2G1_00000002_restart.nc \
      ref_100step/ORCA2G1_*_restart.nc ref_100step/ocean.output \
      p3_a/oracle_*.bin p3_b/oracle_*.bin p3_c/oracle_*.bin ) >"$EVIDENCE/manifest/outputs.sha256"
  [[ ! -f "$REF/output.abort.nc" ]] || sha256sum "$REF/output.abort.nc" >>"$EVIDENCE/manifest/outputs.sha256"
  printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
}

if [[ "$mode" == --admit-existing ]]; then
  for target in "$SMOKE" "$REF" "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
    [[ -e "$target" ]] || { printf 'REFUSE: nothing to admit, %s is absent\n' "$target" >&2; exit 69; }
  done
  mkdir -p "$EVIDENCE/manifest"
  admit
  exit 0
fi
[[ ! -e "$EVIDENCE" ]] || {
  printf 'REFUSE: evidence directory %s already exists; use --admit-existing\n' "$EVIDENCE" >&2; exit 69; }
if [[ "$mode" != --run ]]; then
  printf 'WOULD: mpirun -np 1 ./nemo: smoke (2 steps, %s), reference (100-step deck, %s, NEMO stops it), then twins a, b, c (%s, nn_itend = record length) in %s; then %s\n' \
    "$REF_NAME" "$REF_NAME" "$RUN_NAME" "$EVIDENCE" "$CHECKER"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

mkdir -p "$EVIDENCE/manifest"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
( sha256sum "$SHIPPED_DOMCFG" "$DECK" "$G3_DECK" "$CHECKER" "$REF_BIN" "$RUN_BIN" \
    "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$ORCA2/EXP00/namelist_cfg"
  for entry in "${PINNED_FILES[@]}"; do sha256sum "$INPUTS/${entry%%:*}"; done ) \
  >"$EVIDENCE/manifest/inputs.sha256"

run_one() {            # $1 = binary, $2 = run dir, $3 = nn_itend, $4 = stocklist, $5 = expect: done|stop
  local dir=$2 entry rc=0
  mkdir -p "$dir"
  render_deck "$dir/namelist_cfg" "$3" "$4"
  cp -L "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$dir/namelist_ref"
  cp "$SHIPPED_DOMCFG" "$dir/domain_cfg.nc"
  for entry in "${PINNED_FILES[@]}"; do cp "$INPUTS/${entry%%:*}" "$dir/"; done
  cp "$1" "$dir/nemo"
  (
    cd "$dir"
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.log
    # No pipe; the exit status is kept, so an abort is reported, not trapped.
    mpirun -np 1 --oversubscribe ./nemo >>run.user.log 2>&1 || rc=$?
    printf 'RUN_FINISHED_UTC=%s\nRUN_EXIT=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>run.user.log
    [[ "$rc" -ne 0 ]] || printf 'RUN_DONE\n' >>run.user.log
  )
  [[ -f "$dir/ocean.output" && -f "$dir/time.step" ]] \
    || { printf 'REFUSE: %s wrote no ocean.output/time.step\n' "$dir" >&2; exit 71; }
  logged "$dir" || { printf 'REFUSE: %s did not resolve the G4 switches (see ocean.output)\n' "$dir" >&2; exit 71; }
  grep -E 'nn_e *=|ln_bt_auto|jpiglo|jpjglo|jpkglo|l_NFold|c_NFtype|G1_BUMP|ln_NCAR|nn_iter_algo|ln_qsr_rgb|nn_chldta|nn_sssr|rn_deds|ln_rnf_mouth|nkrnf|w-levels|nn_fwb|sbc_fwb|fld_|dta_tsd' \
    "$dir/ocean.output" >"$dir/resolved_facts.log" || true
  if [[ "$5" == done ]] && { ! grep -qx 'RUN_DONE' "$dir/run.user.log" || grep -q 'E R R O R' "$dir/ocean.output" \
       || [[ "$(tr -d ' \n' <"$dir/time.step")" != "$3" ]]; }; then
    grep -E '^ *kt +[0-9]+ \|' "$dir/ocean.output" >&2 || true
    printf 'REFUSE: %s stopped at kt = %s before nn_itend = %s (the reference completed it)\n' \
      "$dir" "$(tr -d ' \n' <"$dir/time.step")" "$3" >&2
    exit 71
  fi
}

# ---- smoke first, then NEMO's boundary, then the twins at its length --------
run_one "$REF_BIN" "$SMOKE" 2 "$L_SMOKE" done
[[ -f "$SMOKE/ORCA2G1_00000002_restart.nc" ]] || { printf 'REFUSE: the smoke wrote no step-2 restart\n' >&2; exit 71; }
printf 'G4_SMOKE_OK %s\n' "$SMOKE"
run_one "$REF_BIN" "$REF" 100 "$L_REF" stop
steps=$(record_steps)
printf 'G4_RECORD_LENGTH %s (reference completed %s)\n' "$steps" "$(completed_steps)"
run_one "$RUN_BIN" "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$RUN_BIN" "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$RUN_BIN" "$TWIN_C" "$steps" "$(list_c "$steps")" done

# ---- admission -------------------------------------------------------------
admit
