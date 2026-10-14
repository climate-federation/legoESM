#!/usr/bin/env bash
# Grid ladder G3 ACQUISITION (DECISION 117) -- OPERATOR-EXECUTED ONLY.
#
#   preflight (default):  run.sh
#   acquire:              run.sh --run
#   re-admit:             run.sh --admit-existing
#
# The agent that wrote this ran neither makenemo nor NEMO.
#
# WHAT THIS ACQUIRES.  The certified G2 program (ORCA2's shipped 31-level
# partial-cell domain, TSUNAMI's ssh bump at (ji,jj) = (51,74), zero forcing,
# no ice, advection / lateral mixing / drag OFF, split-explicit free surface,
# EEN, constant vertical mixing, rn_Dt = 10800) with ONE switch: &namtsd
# reads ORCA2's initial T/S climatology with ORCA2's own lines
# (ORCA2_OMIP_L4/EXP00/namelist_cfg:47,50,54-55), damping OFF.  The two
# climatology files are ORCA2's, copied byte for byte and pinned by SHA-256.
#
# NO BUILD.  The compiled keys and every MY_SRC file are G1's; ln_tsd_init
# is a namelist switch (istate.f90:117-118) and dtatsd/fldread are compiled
# in every build, so the two round-4 G1 binaries are reused, pinned:
#   ORCA2G1_OMIP_L1_R4REF  the reference binary (no writer);
#   ORCA2G1_OMIP_L1_P5     the record binary (stage frames at kt = 1..50, the
#                          final frame at every kt, solver substeps at
#                          kt = 1, 2, 17, 31, 41).
# The record binary writes 2-D fields and LEVEL 1 of u, v, T, S.  The other
# 29 levels are in NEMO's own restart files: steps 10, 20, .., 100 (twin a),
# steps 2, 4, 6, 8, 10 (twin b) and steps 3, 5, 7, 9 (twin c).  G2 round 3
# found a list of CONSECUTIVE steps never opens (restart.f90:119, :188,
# :199-202); even and odd lists each open, so b and c together give every
# step 2..10.
#
# ROUND 2 (gridladder_r15).  Round 1's acquisition (round14/oracle_g3_r14)
# ran the 2-step SMOKE (finished) and the 100-step reference, which NEMO
# itself stopped at kt = 11: stp_ctl's |V| > 10 m/s check
# (stpctl.f90:181, :245), |V| = 10.24 m/s at (ji,jj,jk) = (22,84,27).  That
# is NEMO's boundary for this program (no dissipation), so G3's record is
# kt = 1..10 and nothing here runs past it.  Both round-1 runs are ADMITTED
# IN PLACE, never re-run: the smoke as a finished 2-step run, the stopped
# run as the boundary (stopped at kt = 11 by |V| alone, its listed step-10
# restart written) and as the REFERENCE binary's step-10 restart.
#
# New runs (mpirun -np 1), all with the record binary, all nn_itend = 10 (a
# multiple of nn_fsbc = 2): twin a (restart 10), twin b (restarts 2, 4, 6,
# 8, 10), twin c (restarts 3, 5, 7, 9).  A run that stops before kt = 10 is
# REFUSED (the reference completed kt = 10, so it is not the boundary); a
# stop after kt = 10 is unreachable.  Admission is G1's check_records.py,
# unchanged, at 10 steps, plus G2's judgments, plus: twins b and c write
# step frames byte-identical to twin a's at kt = 1..10, twin a's step-10
# restart equals the reference binary's, twin b's step-2 restart equals the
# smoke's, and NEMO's log states the T/S read with damping off.
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
readonly STAGE_STEPS=50 SPGTS_STEPS=1,2,17,31,41     # the record binary's own gates
readonly STEPS=10 RECORD_SPGTS=1,2                    # what a 10-step record holds
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round15/oracle_g3_r15}
readonly PRIOR=${PRIOR:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round14/oracle_g3_r14}
readonly SHIPPED_DOMCFG=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/ORCA_R2_zps_domcfg.nc
readonly SHIPPED_DOMCFG_SHA=7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839
readonly INPUTS=$(dirname "$SHIPPED_DOMCFG")
readonly TEM_FILE=data_1m_potential_temperature_nomask.nc SAL_FILE=data_1m_salinity_nomask.nc
readonly TEM_SHA=ca00905c27078305e80130ea6dd07fc575de2ae3257fad38003b329458cf7f7a
readonly SAL_SHA=ad648d972f0631bde7e1b598d98470a979b7f2649271fc9cf5ccfca158e31d0c

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
readonly G2_DECK=$here/../nemo_testcase_g2_orca2bathy/namelist_cfg
readonly G1_DIR=$here/../nemo_testcase_g1_orca2grid
readonly G1_DECK=$G1_DIR/namelist_cfg CHECKER=$G1_DIR/check_records.py
readonly STEP_PATCH=$G1_DIR/stprk3_g1_step_record_stage50.patch   # the record binary's writer
readonly REF_BIN=$NEMO_ROOT/cfgs/$REF_NAME/BLD/bin/nemo.exe
readonly RUN_BIN=$NEMO_ROOT/cfgs/$RUN_NAME/BLD/bin/nemo.exe

for f in "$DECK" "$G2_DECK" "$G1_DECK" "$INPUTS/$TEM_FILE" "$INPUTS/$SAL_FILE" "$CHECKER" "$STEP_PATCH" "$SHIPPED_DOMCFG" "$REF_BIN" "$RUN_BIN" \
         "$NEMO_ROOT/cfgs/SHARED/namelist_ref"; do
  [[ -f "$f" ]] || { printf 'REFUSE: missing input %s\n' "$f" >&2; exit 67; }
done
sha_of() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(sha_of "$SHIPPED_DOMCFG")" == "$SHIPPED_DOMCFG_SHA" ]] \
  || { printf 'REFUSE: the shipped ORCA2 domain file changed\n' >&2; exit 67; }
[[ "$(sha_of "$INPUTS/$TEM_FILE")" == "$TEM_SHA" && "$(sha_of "$INPUTS/$SAL_FILE")" == "$SAL_SHA" ]] \
  || { printf 'REFUSE: an ORCA2 T/S climatology file changed\n' >&2; exit 67; }
[[ "$(sha_of "$REF_BIN")" == "$REF_SHA" && "$(sha_of "$RUN_BIN")" == "$RUN_SHA" ]] \
  || { printf 'REFUSE: a G1 binary is not the round-4 one (rebuilt?)\n' >&2; exit 67; }

# The binaries are G1's program: same keys, the bump statement, the bump
# centre, and the writer only in the record binary with the gates named above.
for name in "$REF_NAME" "$RUN_NAME"; do
  pp=$NEMO_ROOT/cfgs/$name/BLD/ppsrc/nemo
  keys=$(sed -E 's/.*fppkeys *//' "$NEMO_ROOT/cfgs/$name/cpp_$name.fcm" | xargs -n1 | LC_ALL=C sort | xargs)
  [[ "$keys" == "$WANT_KEYS" ]] \
    || { printf 'REFUSE: %s compiled keys "%s", want "%s"\n' "$name" "$keys" "$WANT_KEYS" >&2; exit 68; }
  grep -q 'ORCA2G1: the ssh bump' "$pp/restart.f90" \
    && grep -q 'np_g1_ji0 = 51' "$pp/usrdef_istate.f90" \
    && grep -q 'np_g1_jj0 = 74' "$pp/usrdef_istate.f90" \
    && grep -q 'ORCA2G1' "$pp/usrdef_sbc.f90" && grep -q 'CALL stp_RK3' "$pp/nemogcm.f90" \
    || { printf 'REFUSE: %s is not the G1 program\n' "$name" >&2; exit 68; }
done
[[ "$(grep -c 'g1r_open' "$NEMO_ROOT/cfgs/$REF_NAME/BLD/ppsrc/nemo/stprk3.f90" || true)" -eq 0 ]] \
  || { printf 'REFUSE: the reference binary carries the writer\n' >&2; exit 68; }
pp=$NEMO_ROOT/cfgs/$RUN_NAME/BLD/ppsrc/nemo
[[ "$(grep -c "kstp <= nit000 + $((STAGE_STEPS - 1)) )" "$pp/stprk3.f90" || true)" -eq 5 ]] \
  || { printf 'REFUSE: the record binary does not write stage frames at kt = 1..%s\n' "$STAGE_STEPS" >&2; exit 68; }
gate=$(grep -oE 'kstp /= [0-9]+' "$pp/g1_spgts_record.f90" | awk '{print $3}' | paste -sd, -)
[[ "$gate" == "$SPGTS_STEPS" ]] \
  || { printf 'REFUSE: the substep writer opens at "%s", not "%s"\n' "$gate" "$SPGTS_STEPS" >&2; exit 68; }

# ONE switch: outside &namtsd the G3 deck's assignments are G2's, and G2's
# are G1's; inside it, ln_tsd_init moves and ORCA2's &namtsd data lines
# (cn_dir, sn_tem, sn_sal) appear.  cn_dir is also a key of other groups,
# so the comparison cuts the &namtsd block, never a key name.
assignments() { sed -e 's/!.*//' -e 's/[[:space:]]*$//' "$1" | grep -v '^$'; }
outside_tsd() { assignments "$1" | sed '/^&namtsd/,/^\//d'; }
inside_tsd() { assignments "$1" | sed -n '/^&namtsd/,/^\//p' | tr -d ' \t'; }
orca2=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg
[[ -f "$orca2" ]] || { printf 'REFUSE: missing ORCA2 deck %s\n' "$orca2" >&2; exit 67; }
cmp -s <(assignments "$G2_DECK") <(assignments "$G1_DECK") \
  || { printf 'REFUSE: the G2 deck is not the certified G1 deck\n' >&2; exit 68; }
cmp -s <(outside_tsd "$DECK") <(outside_tsd "$G2_DECK") \
  || { printf 'REFUSE: the G3 deck assigns something outside namtsd the G2 deck does not\n' >&2; exit 68; }
[[ "$(inside_tsd "$G2_DECK")" == $'&namtsd\nln_tsd_init=.false.\nln_tsd_dmp=.false.\n/' ]] \
  || { printf 'REFUSE: the G2 namtsd block is not ln_tsd_init = F, ln_tsd_dmp = F\n' >&2; exit 68; }
[[ "$(inside_tsd "$DECK")" == \
   "$(printf '&namtsd\nln_tsd_init=.true.\nln_tsd_dmp=.false.\n%s\n/' \
        "$(inside_tsd "$orca2" | grep -E '^(cn_dir|sn_tem|sn_sal)=')")" ]] \
  || { printf 'REFUSE: the G3 namtsd block is not ln_tsd_init = T, ln_tsd_dmp = F and ORCA2s own data lines\n' >&2; exit 68; }

render_deck() {        # $1 = target, $2 = nn_itend, $3 = nn_stocklist
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
         -e "s/^(   nn_stock    = ).*/\1    $2/" \
         -e "s/^(   nn_stocklist = ).*/\1$3/" "$DECK" >"$1"
  local moved
  moved=$(diff "$DECK" "$1" | grep -c '^>' || true)
  [[ "$moved" -eq 0 || "$moved" -eq 3 ]] \
    || { printf 'REFUSE: rendering changed %s deck lines, not the 3 run-length lines\n' "$moved" >&2; exit 68; }
}
dry=$(mktemp -d)
trap 'rm -rf "$dry"' EXIT
readonly L_SMOKE='2,0,0,0,0,0,0,0,0,0' L_FULL='10,20,30,40,50,60,70,80,90,100'
readonly L_TEN='10,0,0,0,0,0,0,0,0,0'
readonly L_EVEN='2,4,6,8,10,0,0,0,0,0' L_ODD='3,5,7,9,0,0,0,0,0,0'
render_deck "$dry/smoke" 2 "$L_SMOKE"
render_deck "$dry/a" 10 "$L_TEN"
render_deck "$dry/ten" 10 "$L_EVEN"
render_deck "$dry/odd" 10 "$L_ODD"
render_deck "$dry/full" 100 "$L_FULL"
cmp -s "$dry/full" "$DECK" \
  || { printf 'REFUSE: the committed deck is not the 100-step deck\n' >&2; exit 68; }
python3 -I -m py_compile "$CHECKER"
printf 'PREFLIGHT: binaries and T/S files pinned; keys "%s"; deck = G2 plus namtsd; commit %s\n' "$WANT_KEYS" "$COMMIT"

readonly SMOKE=$PRIOR/smoke_2step REF=$PRIOR/ref_100step
readonly TWIN_A=$EVIDENCE/p3_a_10step TWIN_B=$EVIDENCE/p3_b_10step TWIN_C=$EVIDENCE/p3_c_10step

# NEMO's own log, dtatsd.f90 dta_tsd_init / dta_tsd: the read is on, damping
# off, and the arrays are released after the initial read.
tsd_logged() {
  grep -Eq 'ln_tsd_init *= *T' "$1/ocean.output" && grep -Eq 'ln_tsd_dmp *= *F' "$1/ocean.output" \
    && grep -q 'dta_tsd: deallocte T & S arrays' "$1/ocean.output"
}

# The round-1 smoke, judged as a finished run before anything new runs.
admit_prior() {
  [[ -d "$SMOKE" && -d "$REF" ]] \
    || { printf 'REFUSE: round 1 smoke or stopped run absent under %s\n' "$PRIOR" >&2; exit 69; }
  [[ "$(sha_of "$SMOKE/domain_cfg.nc")" == "$SHIPPED_DOMCFG_SHA" && "$(sha_of "$SMOKE/nemo")" == "$REF_SHA" ]] \
    && [[ "$(sha_of "$SMOKE/$TEM_FILE")" == "$TEM_SHA" && "$(sha_of "$SMOKE/$SAL_FILE")" == "$SAL_SHA" ]] \
    && tsd_logged "$SMOKE" && grep -qx 'RUN_DONE' "$SMOKE/run.user.log" \
    && [[ "$(tr -d ' \n' <"$SMOKE/time.step")" == 2 ]] && ! grep -q 'E R R O R' "$SMOKE/ocean.output" \
    && [[ -f "$SMOKE/ORCA2G1_00000002_restart.nc" ]] \
    || { printf 'REFUSE: %s is not the finished pinned 2-step smoke\n' "$SMOKE" >&2; exit 73; }
  render_deck "$dry/admit_smoke" 2 "$L_SMOKE"
  cmp -s "$SMOKE/namelist_cfg" "$dry/admit_smoke" \
    || { printf 'REFUSE: the smoke ran another deck\n' >&2; exit 73; }
  admit_boundary
}

# NEMO's boundary: the 100-step deck, the reference binary, stopped by
# stp_ctl at kt = 11 on |V| > 10 m/s ALONE (stpctl.f90:243-250), after its
# listed step-10 restart.  Any other stop (step, condition) is refused.
admit_boundary() {
  [[ "$(sha_of "$REF/domain_cfg.nc")" == "$SHIPPED_DOMCFG_SHA" && "$(sha_of "$REF/nemo")" == "$REF_SHA" ]] \
    && [[ "$(sha_of "$REF/$TEM_FILE")" == "$TEM_SHA" && "$(sha_of "$REF/$SAL_FILE")" == "$SAL_SHA" ]] \
    && tsd_logged "$REF" && grep -Eq 'jpkglo *= *31' "$REF/ocean.output" \
    || { printf 'REFUSE: %s is not the pinned G3 reference run\n' "$REF" >&2; exit 73; }
  render_deck "$dry/admit_ref" 100 "$L_FULL"
  cmp -s "$REF/namelist_cfg" "$dry/admit_ref" \
    || { printf 'REFUSE: the stopped run ran another deck\n' >&2; exit 73; }
  [[ "$(tr -d ' \n' <"$REF/time.step")" == 11 ]] && ! grep -qx 'RUN_DONE' "$REF/run.user.log" \
    && grep -q 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$REF/ocean.output" \
    && [[ -f "$REF/ORCA2G1_00000010_restart.nc" && ! -e "$REF/ORCA2G1_00000020_restart.nc" ]] \
    || { printf 'REFUSE: %s did not stop in stp_ctl at kt = 11 after its step-10 restart\n' "$REF" >&2; exit 73; }
  # which check fired, from NEMO's own kt = 11 lines: |V| alone
  awk '/^ *kt 11 \|ssh\| max/ {s=$5} /^ *kt 11 \|U\| +max/ {u=$5} /^ *kt 11 \|V\| +max/ {v=$5}
       /^ *kt 11 Sal +min/ {n=$5} /^ *kt 11 Sal +max/ {x=$5}
       END { exit !(s != "" && u != "" && v != "" && n != "" && x != "" &&
                    s+0 <= 20 && u+0 <= 10 && v+0 > 10 && n+0 > 0 && x+0 < 100) }' "$REF/ocean.output" \
    || { printf 'REFUSE: %s did not stop on |V| > 10 m/s alone at kt = 11\n' "$REF" >&2; exit 73; }
  printf 'G3_BOUNDARY_ADMITTED kt=11 |V|>10 m/s (stp_ctl), step-10 restart present\n'
}

admit() {
  local plant dir bin steps
  # Provenance of what is admitted (a fresh --run and --admit-existing alike):
  # every run directory holds the pinned domain file and its pinned binary,
  # finished, at its own length; the smoke is one of them.
  local list step
  admit_boundary
  for dir in "$SMOKE:$REF_SHA:2:$L_SMOKE" "$TWIN_A:$RUN_SHA:10:$L_TEN" \
             "$TWIN_B:$RUN_SHA:10:$L_EVEN" "$TWIN_C:$RUN_SHA:10:$L_ODD"; do
    IFS=: read -r dir bin steps list <<<"$dir"
    [[ -f "$dir/domain_cfg.nc" && "$(sha_of "$dir/domain_cfg.nc")" == "$SHIPPED_DOMCFG_SHA" ]] \
      || { printf 'REFUSE: %s does not hold the shipped ORCA2 domain file\n' "$dir" >&2; exit 73; }
    [[ -f "$dir/nemo" && "$(sha_of "$dir/nemo")" == "$bin" ]] \
      || { printf 'REFUSE: %s was not run with its pinned binary\n' "$dir" >&2; exit 73; }
    [[ "$(sha_of "$dir/$TEM_FILE")" == "$TEM_SHA" && "$(sha_of "$dir/$SAL_FILE")" == "$SAL_SHA" ]] \
      || { printf 'REFUSE: %s does not hold the pinned T/S climatology\n' "$dir" >&2; exit 73; }
    tsd_logged "$dir" \
      || { printf 'REFUSE: %s ocean.output does not state the T/S read with damping off\n' "$dir" >&2; exit 73; }
    grep -qx 'RUN_DONE' "$dir/run.user.log" && [[ "$(tr -d ' \n' <"$dir/time.step")" == "$steps" ]] \
      && grep -Eq 'jpkglo *= *31' "$dir/ocean.output" && ! grep -q 'E R R O R' "$dir/ocean.output" \
      || { printf 'REFUSE: %s is not a finished %s-step 31-level run\n' "$dir" "$steps" >&2; exit 73; }
    # The whole deck, run length and restart list included, is the committed
    # one rendered for this run; every listed restart exists.
    render_deck "$dry/admit_deck" "$steps" "$list"
    cmp -s "$dir/namelist_cfg" "$dry/admit_deck" \
      || { printf 'REFUSE: %s ran a deck that is not the committed one at its length and list\n' "$dir" >&2; exit 73; }
    for step in ${list//,/ }; do
      [[ "$step" == 0 || -f "$dir/ORCA2G1_$(printf '%08d' "$step")_restart.nc" ]] \
        || { printf 'REFUSE: %s has no restart at listed step %s\n' "$dir" "$step" >&2; exit 73; }
    done
  done
  for plant in truncation nonfinite missing-frame twin-ulp missing-step missing-substep \
               twin-substep-inner; do
    if python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" \
         --steps "$STEPS" --stage-steps "$STEPS" --plant "$plant" \
         --substep-steps "$RECORD_SPGTS" >"$EVIDENCE/manifest/plant_${plant}.log" 2>&1; then
      printf 'REFUSE: record plant %s stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/manifest/plant_${plant}.log" \
      || { printf 'REFUSE: record plant %s did not fire\n' "$plant" >&2; exit 72; }
  done
  # every step of a 10-step record is a stage step; the reference is the
  # stopped run's step-10 restart (reference binary, same deck to kt = 10)
  python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" --reference "$REF" \
    --steps "$STEPS" --stage-steps "$STEPS" --restart-steps 10 \
    --substep-steps "$RECORD_SPGTS" --json "$EVIDENCE/g3_round15_admission.json"
  # Twin b writes restarts at every second step; that it changes nothing is
  # judged here: its step-10 restart is twin a's, variable by variable.
  # The checker does not know which fields a frame must hold; the committed
  # writer patch (what the record binary was built from) does.
  python3 -I - "$G1_DIR" "$TWIN_A" "$TWIN_B" "$STEP_PATCH" "$EVIDENCE/g3_round15_admission.json" \
      "$TWIN_C" "$SMOKE" <<'PY'
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import check_records
try:
    frame, stage, last = None, set(), set()
    for line in Path(sys.argv[4]).read_text().splitlines():
        if not line.startswith("+"):
            continue
        hit = re.search(r"CALL g1r_frame\( '(.)' \)", line)
        frame = hit.group(1) if hit else frame
        hit = re.search(r"CALL g1r_w2\( '(\w+)'", line)
        if hit:
            stage.add(f"{frame}_{hit.group(1)}")
            if frame == "f":
                last.add(f"f_{hit.group(1)}")
    admitted = json.loads(Path(sys.argv[5]).read_text())
    check_records.require(len(stage) > 40 and len(last) > 5, "the writer patch was not parsed")
    check_records.require(admitted["fields_stage"] == sorted(stage),
                          "a stage step lacks a field the writer writes")
    # a 10-step record has no final-only step (stage frames run to kt = 50)
    check_records.require(admitted["fields_final_only"] == [],
                          "a 10-step record holds a final-only step")
    check_records.restarts_identical(Path(sys.argv[2]), Path(sys.argv[3]), [10])
    # the record binary's step 2 is the reference binary's smoke step 2
    check_records.restarts_identical(Path(sys.argv[7]), Path(sys.argv[3]), [2])
    for step in (2, 4, 6, 8):
        check_records.require(
            (Path(sys.argv[3]) / f"ORCA2G1_{step:08d}_restart.nc").is_file(),
            f"twin b has no restart at step {step}")
    for step in (3, 5, 7, 9):
        check_records.require(
            (Path(sys.argv[6]) / f"ORCA2G1_{step:08d}_restart.nc").is_file(),
            f"twin c has no restart at step {step}")
except check_records.Refuse as exc:
    sys.exit(f"REFUSE: {exc}")
print("G2_FRAME_FIELDS_COMPLETE", len(stage), "per stage step; no final-only step")
print("G3_TWIN_B_RESTARTS_PASSIVE step 10 equals twin a, step 2 equals the smoke; "
      "steps 2,4,6,8 present; twin c 3,5,7,9 present")
PY
  # The extra restart writes are passive: each 10-step twin's step frames are
  # twin a's, byte for byte, at kt = 1..10.
  local kt twin
  for twin in "$TWIN_B" "$TWIN_C"; do
    for kt in $(seq 1 10); do
      cmp -s "$TWIN_A/oracle_g1step_kt$(printf '%08d' "$kt").bin" \
             "$twin/oracle_g1step_kt$(printf '%08d' "$kt").bin" \
        || { printf 'REFUSE: %s step frame kt=%s differs from twin a\n' "$twin" "$kt" >&2; exit 73; }
    done
  done
  printf 'G3_TWIN_FRAMES_IDENTICAL kt=1..10 (twins b, c)\n'
  ( cd "$EVIDENCE" && sha256sum g3_round15_admission.json p3_a_10step/domain_cfg.nc \
      p3_a_10step/mesh_mask.nc p3_a_10step/ORCA2G1_00000010_restart.nc \
      p3_b_10step/ORCA2G1_0000000{2,4,6,8}_restart.nc p3_b_10step/ORCA2G1_00000010_restart.nc \
      p3_c_10step/ORCA2G1_0000000{3,5,7,9}_restart.nc \
      "$SMOKE/ORCA2G1_00000002_restart.nc" "$REF/ORCA2G1_00000010_restart.nc" \
      "$REF/ocean.output" ) >"$EVIDENCE/manifest/outputs.sha256"
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
  printf 'WOULD: admit %s/{smoke_2step,ref_100step} in place; mpirun -np 1 ./nemo (%s) in %s/{p3_a_10step,p3_b_10step,p3_c_10step}; then %s\n' \
    "$PRIOR" "$RUN_NAME" "$EVIDENCE" "$CHECKER"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

mkdir -p "$EVIDENCE/manifest"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
sha256sum "$SHIPPED_DOMCFG" "$INPUTS/$TEM_FILE" "$INPUTS/$SAL_FILE" "$DECK" "$G2_DECK" "$G1_DECK" "$CHECKER" "$REF_BIN" "$RUN_BIN" \
  "$NEMO_ROOT/cfgs/SHARED/namelist_ref" >"$EVIDENCE/manifest/inputs.sha256"

run_one() {            # $1 = binary, $2 = run dir, $3 = nn_itend, $4 = stocklist, $5 = last listed step
  local dir=$2
  mkdir -p "$dir"
  render_deck "$dir/namelist_cfg" "$3" "$4"
  cp -L "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$dir/namelist_ref"
  cp "$SHIPPED_DOMCFG" "$dir/domain_cfg.nc"
  cp "$INPUTS/$TEM_FILE" "$INPUTS/$SAL_FILE" "$dir/"
  cp "$1" "$dir/nemo"
  (
    cd "$dir"
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.log
    # No pipe; the exit status is kept, so an abort is reported, not trapped.
    rc=0
    mpirun -np 1 --oversubscribe ./nemo >>run.user.log 2>&1 || rc=$?
    printf 'RUN_FINISHED_UTC=%s\nRUN_EXIT=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>run.user.log
    [[ "$rc" -ne 0 ]] || printf 'RUN_DONE\n' >>run.user.log
  )
  # EXPECTED-ABORT HANDLING.  The reference completed kt = 1..10, so any
  # stop here (NEMO's stp_ctl or otherwise) is NOT the registered kt = 11
  # boundary: it is refused, with NEMO's own stop lines quoted.
  if ! grep -qx 'RUN_DONE' "$dir/run.user.log" || grep -q 'E R R O R' "$dir/ocean.output"; then
    grep -E '^ *kt +[0-9]+ \|' "$dir/ocean.output" >&2 || true
    printf 'REFUSE: %s stopped at kt = %s before nn_itend = %s; not the registered kt = 11 boundary\n' \
      "$dir" "$(tr -d ' \n' <"$dir/time.step" 2>/dev/null || echo '?')" "$3" >&2
    exit 71
  fi
  # The one switch, read from NEMO's own log: 31 levels, partial cells.
  grep -Eq 'jpkglo *= *31' "$dir/ocean.output" \
    || { printf 'REFUSE: %s did not read a 31-level domain\n' "$dir" >&2; exit 71; }
  tsd_logged "$dir" \
    || { printf 'REFUSE: %s did not read the T/S climatology with damping off\n' "$dir" >&2; exit 71; }
  [[ "$(tr -d ' \n' <"$dir/time.step")" == "$3" ]] \
    || { printf 'REFUSE: %s stopped before step %s\n' "$dir" "$3" >&2; exit 71; }
  # The last LISTED step: twin c's odd list ends at 9, and after its last
  # entry nitrst is 0, so NEMO writes no step-10 restart (restart.f90:199-202).
  [[ -f "$dir/ORCA2G1_$(printf '%08d' "$5")_restart.nc" ]] \
    || { printf 'REFUSE: %s wrote no restart at its last listed step %s\n' "$dir" "$5" >&2; exit 71; }
  # Facts the receipt quotes, read from NEMO's own log (reported, not judged).
  grep -E 'nn_e *=|ln_bt_auto|jpiglo|jpjglo|jpkglo|l_Iperio|l_NFold|c_NFtype|G1_BUMP|ourant|l_zps|strait|mbku|tsd|ln_eos|ln_hpg|dta_tsd|fld_' \
    "$dir/ocean.output" >"$dir/resolved_facts.log" || true
}

# ---- the smoke first: round 1's finished smoke and the boundary, admitted ----
admit_prior
printf 'G3_SMOKE_OK %s (admitted, not re-run)\n' "$SMOKE"
run_one "$RUN_BIN" "$TWIN_A" 10 "$L_TEN" 10
run_one "$RUN_BIN" "$TWIN_B" 10 "$L_EVEN" 10
run_one "$RUN_BIN" "$TWIN_C" 10 "$L_ODD" 9

# ---- admission -------------------------------------------------------------
admit
