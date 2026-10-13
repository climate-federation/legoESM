#!/usr/bin/env bash
# Grid ladder G2 ACQUISITION (DECISION 117) -- OPERATOR-EXECUTED ONLY.
#
#   preflight (default):  run.sh
#   acquire:              run.sh --run
#   re-admit:             run.sh --admit-existing
#
# The agent that wrote this ran neither makenemo nor NEMO.
#
# WHAT THIS ACQUIRES.  The certified G1 program (uniform T/S, TSUNAMI's ssh
# bump at (ji,jj) = (51,74), zero forcing, no ice, advection / lateral mixing
# / drag OFF, split-explicit free surface, EEN, constant vertical mixing,
# rn_Dt = 10800) with ONE switch: the run directory's domain_cfg.nc is
# ORCA2's SHIPPED ORCA_R2_zps_domcfg.nc (31 levels, partial bottom cells,
# ORCA2's bathymetry), copied byte for byte.  No DOMAINcfg run.
#
# NO BUILD.  The compiled keys and every MY_SRC file are G1's, and NEMO sizes
# the domain from the file (domain.f90:581), so the two round-4 G1 binaries
# are reused, pinned by SHA-256:
#   ORCA2G1_OMIP_L1_R4REF  the reference binary (no writer);
#   ORCA2G1_OMIP_L1_P5     the record binary (stage frames at kt = 1..50, the
#                          final frame at every kt, solver substeps at
#                          kt = 1, 2, 17, 31, 41).
# The record binary writes 2-D fields and LEVEL 1 of u, v, T, S.  The other
# 29 levels are in NEMO's own restart files: at steps 10, 20, .., 100 (twin a,
# as G1) and at steps 2, 4, 6, 8, 10 (twin b; the one added output).
#
# Runs (mpirun -np 1): a 2-step SMOKE of the reference binary (judged before
# anything else runs), the 100-step reference, the 100-step record (twin a)
# and a 10-step twin b.  Admission is G1's check_records.py, unchanged, plus
# twin b's step-10 restart equal to twin a's.
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
readonly STEPS=100 STAGE_STEPS=50 SPGTS_STEPS=1,2,17,31,41
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round10/oracle_g2_r10}
readonly SHIPPED_DOMCFG=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/ORCA_R2_zps_domcfg.nc
readonly SHIPPED_DOMCFG_SHA=7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839

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
readonly G1_DIR=$here/../nemo_testcase_g1_orca2grid
readonly G1_DECK=$G1_DIR/namelist_cfg CHECKER=$G1_DIR/check_records.py
readonly STEP_PATCH=$G1_DIR/stprk3_g1_step_record_stage50.patch   # the record binary's writer
readonly REF_BIN=$NEMO_ROOT/cfgs/$REF_NAME/BLD/bin/nemo.exe
readonly RUN_BIN=$NEMO_ROOT/cfgs/$RUN_NAME/BLD/bin/nemo.exe

for f in "$DECK" "$G1_DECK" "$CHECKER" "$STEP_PATCH" "$SHIPPED_DOMCFG" "$REF_BIN" "$RUN_BIN" \
         "$NEMO_ROOT/cfgs/SHARED/namelist_ref"; do
  [[ -f "$f" ]] || { printf 'REFUSE: missing input %s\n' "$f" >&2; exit 67; }
done
sha_of() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(sha_of "$SHIPPED_DOMCFG")" == "$SHIPPED_DOMCFG_SHA" ]] \
  || { printf 'REFUSE: the shipped ORCA2 domain file changed\n' >&2; exit 67; }
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

# ONE switch: the G2 deck's assignments are G1's (comments may differ).
assignments() { sed -e 's/!.*//' -e 's/[[:space:]]*$//' "$1" | grep -v '^$'; }
cmp -s <(assignments "$DECK") <(assignments "$G1_DECK") \
  || { printf 'REFUSE: the G2 deck assigns something the certified G1 deck does not\n' >&2; exit 68; }

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
render_deck "$dry/smoke" 2 '2,0,0,0,0,0,0,0,0,0'
render_deck "$dry/ten" 10 '2,4,6,8,10,0,0,0,0,0'
render_deck "$dry/full" 100 '10,20,30,40,50,60,70,80,90,100'
cmp -s "$dry/full" "$DECK" \
  || { printf 'REFUSE: the committed deck is not the 100-step deck\n' >&2; exit 68; }
python3 -I -m py_compile "$CHECKER"
printf 'PREFLIGHT: binaries pinned; keys "%s"; deck assignments equal G1; commit %s\n' "$WANT_KEYS" "$COMMIT"

readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly TWIN_A=$EVIDENCE/p3_a_100step TWIN_B=$EVIDENCE/p3_b_10step

admit() {
  local plant dir bin steps
  # Provenance of what is admitted (a fresh --run and --admit-existing alike):
  # every run directory holds the pinned domain file and its pinned binary,
  # finished, at its own length; the smoke is one of them.
  for dir in "$SMOKE:$REF_SHA:2" "$REF:$REF_SHA:100" "$TWIN_A:$RUN_SHA:100" "$TWIN_B:$RUN_SHA:10"; do
    IFS=: read -r dir bin steps <<<"$dir"
    [[ -f "$dir/domain_cfg.nc" && "$(sha_of "$dir/domain_cfg.nc")" == "$SHIPPED_DOMCFG_SHA" ]] \
      || { printf 'REFUSE: %s does not hold the shipped ORCA2 domain file\n' "$dir" >&2; exit 73; }
    [[ -f "$dir/nemo" && "$(sha_of "$dir/nemo")" == "$bin" ]] \
      || { printf 'REFUSE: %s was not run with its pinned binary\n' "$dir" >&2; exit 73; }
    grep -qx 'RUN_DONE' "$dir/run.user.log" && [[ "$(tr -d ' \n' <"$dir/time.step")" == "$steps" ]] \
      && grep -Eq 'jpkglo *= *31' "$dir/ocean.output" && ! grep -q 'E R R O R' "$dir/ocean.output" \
      || { printf 'REFUSE: %s is not a finished %s-step 31-level run\n' "$dir" "$steps" >&2; exit 73; }
    cmp -s <(assignments "$dir/namelist_cfg" | grep -Ev 'nn_itend|nn_stock') \
           <(assignments "$DECK" | grep -Ev 'nn_itend|nn_stock') \
      || { printf 'REFUSE: %s ran a deck that is not the committed one\n' "$dir" >&2; exit 73; }
  done
  for plant in truncation nonfinite missing-frame twin-ulp missing-step missing-substep; do
    if python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" \
         --steps "$STEPS" --stage-steps "$STAGE_STEPS" --plant "$plant" \
         --substep-steps "$SPGTS_STEPS" >"$EVIDENCE/manifest/plant_${plant}.log" 2>&1; then
      printf 'REFUSE: record plant %s stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/manifest/plant_${plant}.log" \
      || { printf 'REFUSE: record plant %s did not fire\n' "$plant" >&2; exit 72; }
  done
  python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" --reference "$REF" \
    --steps "$STEPS" --stage-steps "$STAGE_STEPS" --restart-steps 10,100 \
    --substep-steps "$SPGTS_STEPS" --json "$EVIDENCE/g2_round10_admission.json"
  # Twin b writes restarts at every second step; that it changes nothing is
  # judged here: its step-10 restart is twin a's, variable by variable.
  # The checker does not know which fields a frame must hold; the committed
  # writer patch (what the record binary was built from) does.
  python3 -I - "$G1_DIR" "$TWIN_A" "$TWIN_B" "$STEP_PATCH" "$EVIDENCE/g2_round10_admission.json" <<'PY'
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
    check_records.require(admitted["fields_final_only"] == sorted(last),
                          "a final-only step lacks a field the writer writes")
    check_records.restarts_identical(Path(sys.argv[2]), Path(sys.argv[3]), [10])
    for step in (2, 4, 6, 8):
        check_records.require(
            (Path(sys.argv[3]) / f"ORCA2G1_{step:08d}_restart.nc").is_file(),
            f"twin b has no restart at step {step}")
except check_records.Refuse as exc:
    sys.exit(f"REFUSE: {exc}")
print("G2_FRAME_FIELDS_COMPLETE", len(stage), "per stage step,", len(last), "per final-only step")
print("G2_TWIN_B_RESTARTS_PASSIVE step 10 equals twin a; steps 2,4,6,8 present")
PY
  ( cd "$EVIDENCE" && sha256sum g2_round10_admission.json p3_a_100step/domain_cfg.nc \
      p3_a_100step/mesh_mask.nc ref_100step/ORCA2G1_00000100_restart.nc \
      p3_a_100step/ORCA2G1_00000100_restart.nc p3_b_10step/ORCA2G1_0000000{2,4,6,8}_restart.nc \
      p3_b_10step/ORCA2G1_00000010_restart.nc ) >"$EVIDENCE/manifest/outputs.sha256"
  printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
}

if [[ "$mode" == --admit-existing ]]; then
  for target in "$SMOKE" "$REF" "$TWIN_A" "$TWIN_B"; do
    [[ -e "$target" ]] || { printf 'REFUSE: nothing to admit, %s is absent\n' "$target" >&2; exit 69; }
  done
  mkdir -p "$EVIDENCE/manifest"
  admit
  exit 0
fi
[[ ! -e "$EVIDENCE" ]] || {
  printf 'REFUSE: evidence directory %s already exists; use --admit-existing\n' "$EVIDENCE" >&2; exit 69; }
if [[ "$mode" != --run ]]; then
  printf 'WOULD: mpirun -np 1 ./nemo (%s, %s) in %s/{smoke_2step,ref_100step,p3_a_100step,p3_b_10step}; then %s\n' \
    "$REF_NAME" "$RUN_NAME" "$EVIDENCE" "$CHECKER"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

mkdir -p "$EVIDENCE/manifest"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
sha256sum "$SHIPPED_DOMCFG" "$DECK" "$G1_DECK" "$CHECKER" "$REF_BIN" "$RUN_BIN" \
  "$NEMO_ROOT/cfgs/SHARED/namelist_ref" >"$EVIDENCE/manifest/inputs.sha256"

run_one() {            # $1 = binary, $2 = run dir, $3 = nn_itend, $4 = stocklist
  local dir=$2
  mkdir -p "$dir"
  render_deck "$dir/namelist_cfg" "$3" "$4"
  cp -L "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$dir/namelist_ref"
  cp "$SHIPPED_DOMCFG" "$dir/domain_cfg.nc"
  cp "$1" "$dir/nemo"
  (
    cd "$dir"
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.log
    # No pipe: a failing mpirun must abort under set -e, not hide in a pipe.
    mpirun -np 1 --oversubscribe ./nemo >>run.user.log 2>&1
    printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.log
  )
  if grep -q 'E R R O R' "$dir/ocean.output"; then
    printf 'REFUSE: %s ocean.output reports an error\n' "$dir" >&2; exit 71
  fi
  # The one switch, read from NEMO's own log: 31 levels, partial cells.
  grep -Eq 'jpkglo *= *31' "$dir/ocean.output" \
    || { printf 'REFUSE: %s did not read a 31-level domain\n' "$dir" >&2; exit 71; }
  [[ "$(tr -d ' \n' <"$dir/time.step")" == "$3" ]] \
    || { printf 'REFUSE: %s stopped before step %s\n' "$dir" "$3" >&2; exit 71; }
  [[ -f "$dir/ORCA2G1_$(printf '%08d' "$3")_restart.nc" ]] \
    || { printf 'REFUSE: %s wrote no final restart\n' "$dir" >&2; exit 71; }
  # Facts the receipt quotes, read from NEMO's own log (reported, not judged).
  grep -E 'nn_e *=|ln_bt_auto|jpiglo|jpjglo|jpkglo|l_Iperio|l_NFold|c_NFtype|G1_BUMP|ourant|l_zps|strait|mbku' \
    "$dir/ocean.output" >"$dir/resolved_facts.log" || true
}

# ---- runs: the mandatory 2-step smoke first --------------------------------
run_one "$REF_BIN" "$SMOKE" 2 '2,0,0,0,0,0,0,0,0,0'
printf 'G2_SMOKE_OK %s\n' "$SMOKE"
run_one "$REF_BIN" "$REF" 100 '10,20,30,40,50,60,70,80,90,100'
run_one "$RUN_BIN" "$TWIN_A" 100 '10,20,30,40,50,60,70,80,90,100'
run_one "$RUN_BIN" "$TWIN_B" 10 '2,4,6,8,10,0,0,0,0,0'

# ---- admission -------------------------------------------------------------
admit
