#!/usr/bin/env bash
# Grid ladder G1b ACQUISITION (user decision 2026-10-14) -- OPERATOR-EXECUTED ONLY.
#
#   preflight (default):  run.sh
#   acquire:              run.sh --run
#   re-admit:             run.sh --admit-existing
#   resume (round 24):    run.sh --resume   -- admits the smoke, the reference
#                         and every finished twin IN PLACE (never re-run) and
#                         runs only the twins whose directory is absent: after
#                         round 23's resume, the three extra twins r11..r13.
#
# The agent that wrote this ran neither makenemo nor NEMO.
#
# WHAT THIS ACQUIRES.  The certified G1 program (ORCA2's grid, coast and
# fold, flat 100 m, one wet level, TSUNAMI's bump at (ji,jj) = (51,74), no
# forcing, no ice, rn_Dt = 10800) with ONE switch: &namdyn_adv is ORCA2's
# (ORCA2_OMIP_L4/EXP00/namelist_cfg:346-347, ln_dynadv_vec = T, nn_dynkeg = 0).
#
# NO BUILD.  ln_dynadv_vec is a namelist switch (dynadv.f90:162,186); the
# vector-form routines are compiled in every build, so G1's two round-4
# binaries are reused, pinned (the same pair the G3 acquisition used):
#   ORCA2G1_OMIP_L1_R4REF  the reference binary (no writer);
#   ORCA2G1_OMIP_L1_P5     the record binary (stage frames kt = 1..50, the
#                          final frame at every kt, solver substeps at
#                          kt = 1, 2, 17, 31, 41).
# G1's DOMAINcfg-built domain file is reused, pinned by SHA-256.
#
# RUNS (mpirun -np 1):
#   smoke     reference binary, 2 steps, restart at 2;
#   ref       reference binary, the committed 100-step deck (restarts 10..100).
#             NEMO MAY STOP EARLY (no dissipation).  N = 100 if it finishes;
#             if stp_ctl stops it at kt = K, N = the largest even step < K
#             (nn_itend must be a multiple of nn_fsbc = 2) and the boundary
#             is admitted as such.  Any other stop is refused.  N < 10 is
#             refused (the plan needs at least one 10-step restart).
#   twins r01..r10, record binary, nn_itend = N, restart list
#             j, j+10, .., <= N for twin rj: NEMO's list holds 10 entries
#             (in_out_manager.f90:51) and a list of consecutive steps never
#             opens (G2 round 3), so ten stride-10 lists give a restart at
#             EVERY step 1..N.  r10 is twin a (restarts 10, 20, ..);
#             r01 is the determinism twin.
#   ROUND 24 (measured on round 23's resume): the stride-10 plan does NOT
#             give every step.  A listed file opens one step early
#             (restart.f90:119); the pointer advances only after a write
#             (restart.f90:188, :200-201).  So (a) a list starting at 1 never
#             opens (there is no step 0): r01 wrote nothing, and its one file,
#             named for step 1, is an empty shell made by :119's end-of-run
#             clause; (b) a full 10-entry list ending below N leaves the
#             pointer on its last entry, and the same clause re-opens and
#             BLANKS that file at kt = N: r02..r09's restarts 92..99 are empty
#             shells.  No step-1 restart can exist on this deck (G2 round 3,
#             G4 round 5).  Extra twins (N = 100), each list ending on a 0
#             (nitrst = 0 never opens, :120): r11 = 11,21,..,91;
#             r12 = 92,94,96,98; r13 = 93,95,97,99 (consecutive entries
#             never open, :119).
# ADMISSION: G1's check_records.py, unchanged, on twin a with twin r01 and
# the reference; every other twin's step frames byte-identical to twin a's
# at kt = 1..N (extra restart writes are passive); a restart HOLDING A
# RECORD AT ITS STEP (a time record, kt = step) at every step 2..N;
# NEMO's own log states the vector form and total vorticity.
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
readonly STAGE_GATE=50 SPGTS_GATE=1,2,17,31,41      # the record binary's own gates
readonly G1_DOMCFG=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round4/oracle_g1_r4/domaincfg/domain_cfg.nc
readonly G1_DOMCFG_SHA=73801c6ff5c78183b85b5e654c0a8c152355007d85cea1ef3ae8e6b151deabdd
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round22/oracle_g1b_r22}

mode=--preflight-only
case "${1:---preflight-only}" in
  --run|--preflight-only|--admit-existing|--resume) mode=${1:---preflight-only} ;;
  *) printf 'Usage: %s [--run|--preflight-only|--admit-existing|--resume]\n' "$0" >&2; exit 64 ;;
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
readonly ORCA2_DECK=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg

for f in "$DECK" "$G1_DECK" "$CHECKER" "$STEP_PATCH" "$G1_DOMCFG" "$REF_BIN" "$RUN_BIN" \
         "$ORCA2_DECK" "$NEMO_ROOT/cfgs/SHARED/namelist_ref"; do
  [[ -f "$f" ]] || { printf 'REFUSE: missing input %s\n' "$f" >&2; exit 67; }
done
sha_of() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(sha_of "$G1_DOMCFG")" == "$G1_DOMCFG_SHA" ]] \
  || { printf 'REFUSE: the G1 domain file changed\n' >&2; exit 67; }
[[ "$(sha_of "$REF_BIN")" == "$REF_SHA" && "$(sha_of "$RUN_BIN")" == "$RUN_SHA" ]] \
  || { printf 'REFUSE: a G1 binary is not the round-4 one (rebuilt?)\n' >&2; exit 67; }

# The binaries are G1's program: same keys, the bump statement and centre,
# the writer only in the record binary with the gates named above, and the
# vector-form routines compiled (they are reached by the namelist alone).
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
  grep -q 'n_dynadv = np_VEC_c2' "$pp/dynadv.f90" && grep -q 'ntot = np_CRV' "$pp/dynvor.f90" \
    && grep -q 'CALL dyn_keg' "$pp/stp2d.f90" \
    || { printf 'REFUSE: %s does not compile the vector-form path\n' "$name" >&2; exit 68; }
done
[[ "$(grep -c 'g1r_open' "$NEMO_ROOT/cfgs/$REF_NAME/BLD/ppsrc/nemo/stprk3.f90" || true)" -eq 0 ]] \
  || { printf 'REFUSE: the reference binary carries the writer\n' >&2; exit 68; }
pp=$NEMO_ROOT/cfgs/$RUN_NAME/BLD/ppsrc/nemo
[[ "$(grep -c "kstp <= nit000 + $((STAGE_GATE - 1)) )" "$pp/stprk3.f90" || true)" -eq 5 ]] \
  || { printf 'REFUSE: the record binary does not write stage frames at kt = 1..%s\n' "$STAGE_GATE" >&2; exit 68; }
gate=$(grep -oE 'kstp /= [0-9]+' "$pp/g1_spgts_record.f90" | awk '{print $3}' | paste -sd, -)
[[ "$gate" == "$SPGTS_GATE" ]] \
  || { printf 'REFUSE: the substep writer opens at "%s", not "%s"\n' "$gate" "$SPGTS_GATE" >&2; exit 68; }

# ONE switch: outside &namdyn_adv the G1b deck's assignments are G1's; inside
# it, ORCA2's two lines replace G1's OFF selection.
assignments() { sed -e 's/!.*//' -e 's/[[:space:]]*$//' "$1" | grep -v '^$'; }
outside_adv() { assignments "$1" | sed '/^&namdyn_adv/,/^\//d'; }
inside_adv() { assignments "$1" | sed -n '/^&namdyn_adv/,/^\//p' | tr -d ' \t'; }
cmp -s <(outside_adv "$DECK") <(outside_adv "$G1_DECK") \
  || { printf 'REFUSE: the G1b deck assigns something outside namdyn_adv the G1 deck does not\n' >&2; exit 68; }
[[ "$(inside_adv "$G1_DECK")" == $'&namdyn_adv\nln_dynadv_OFF=.true.\nln_dynadv_vec=.false.\nln_dynadv_cen2=.false.\nln_dynadv_up3=.false.\n/' ]] \
  || { printf 'REFUSE: the G1 namdyn_adv block is not linear dynamics\n' >&2; exit 68; }
[[ "$(inside_adv "$ORCA2_DECK")" == $'&namdyn_adv\nln_dynadv_vec=.true.\nnn_dynkeg=0\n/' ]] \
  || { printf 'REFUSE: ORCA2s namdyn_adv block is not ln_dynadv_vec = T, nn_dynkeg = 0\n' >&2; exit 68; }
[[ "$(inside_adv "$DECK")" == $'&namdyn_adv\nln_dynadv_OFF=.false.\nln_dynadv_vec=.true.\nnn_dynkeg=0\nln_dynadv_cen2=.false.\nln_dynadv_up3=.false.\n/' ]] \
  || { printf 'REFUSE: the G1b namdyn_adv block is not ORCA2s two lines with the other forms off\n' >&2; exit 68; }

render_deck() {        # $1 = target, $2 = nn_itend, $3 = nn_stocklist
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
         -e "s/^(   nn_stock    = ).*/\1    $2/" \
         -e "s/^(   nn_stocklist = ).*/\1$3/" "$DECK" >"$1"
  only_run_length "$DECK" "$1" || exit 68
}
# Every line the rendering changed is one of the three run-length lines (at
# N = the deck's own 100 only nn_stocklist moves, so the count is 0..3).
only_run_length() {    # $1 = committed deck, $2 = rendered deck
  local other
  other=$(diff "$1" "$2" | grep -E '^[<>]' \
          | grep -Ev '^[<>]    nn_(itend|stock) +=  |^[<>]    nn_stocklist = ' || true)
  [[ $(wc -l <"$1") == $(wc -l <"$2") && -z "$other" ]] \
    || { printf 'REFUSE: rendering changed a deck line that is not a run-length line: %s\n' "${other:-line count}" >&2; return 1; }
}
twin_list() {          # $1 = j, $2 = N: j, j+10, .. <= N, zero-padded to 10 entries
  local j=$1 n=$2 out=() k
  for ((k = j; k <= n; k += 10)); do out+=("$k"); done
  while ((${#out[@]} < 10)); do out+=(0); done
  local IFS=,; printf '%s' "${out[*]}"
}
readonly L_SMOKE='2,0,0,0,0,0,0,0,0,0' L_FULL='10,20,30,40,50,60,70,80,90,100'
readonly N_TWINS=13 L_R11='11,21,31,41,51,61,71,81,91,0' \
         L_R12='92,94,96,98,0,0,0,0,0,0' L_R13='93,95,97,99,0,0,0,0,0,0'
twin_list_of() {       # $1 = j, $2 = N: r01..r10 stride 10; r11..r13 the extra lists (N = 100 only)
  case "$1" in
    11|12|13)
      [[ "$2" == 100 ]] || { printf 'REFUSE: the extra twins are planned for N = 100, not %s\n' "$2" >&2; exit 73; }
      case "$1" in 11) printf '%s' "$L_R11" ;; 12) printf '%s' "$L_R12" ;; *) printf '%s' "$L_R13" ;; esac ;;
    *) twin_list "$1" "$2" ;;
  esac
}
dry=$(mktemp -d)
trap 'rm -rf "$dry"' EXIT
render_deck "$dry/smoke" 2 "$L_SMOKE"
render_deck "$dry/full" 100 "$L_FULL"
cmp -s "$dry/full" "$DECK" \
  || { printf 'REFUSE: the committed deck is not the 100-step deck\n' >&2; exit 68; }
[[ "$(twin_list 10 100)" == "$L_FULL" && "$(twin_list 3 100)" == '3,13,23,33,43,53,63,73,83,93' \
   && "$(twin_list 7 30)" == '7,17,27,0,0,0,0,0,0,0' ]] \
  || { printf 'REFUSE: twin_list is wrong\n' >&2; exit 68; }
render_deck "$dry/one" 100 "$(twin_list 1 100)"      # the twin r01 deck: one line moves
for j in 11 12 13; do render_deck "$dry/x$j" 100 "$(twin_list_of "$j" 100)"; done
[[ "$(diff "$DECK" "$dry/one" | grep -c '^>')" -eq 1 ]] \
  || { printf 'REFUSE: the N = 100 twin deck should move nn_stocklist only\n' >&2; exit 68; }
sed 's/^\(   rn_Dt *= *\).*/\1 5400./' "$dry/one" >"$dry/plant"          # plant: a non-run-length line
if only_run_length "$DECK" "$dry/plant" 2>/dev/null || cmp -s "$dry/one" "$dry/plant"; then
  printf 'REFUSE: the rendering plant (rn_Dt moved) stayed green\n' >&2; exit 72
fi
python3 -I -m py_compile "$CHECKER"
printf 'PREFLIGHT: binaries and domain pinned; keys "%s"; deck = G1 plus ORCA2 namdyn_adv; commit %s\n' "$WANT_KEYS" "$COMMIT"

readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
twin_dir() { printf '%s/r%02d\n' "$EVIDENCE" "$1"; }   # newline: $(for ..; twin_dir) must split into words

# NEMO's own log: the vector form and the total vorticity it derives
# (dynadv.f90:197, dynvor.f90 dyn_vor_init "vector form dynamics").
vec_logged() {
  grep -q 'vector form : keg + zad + vor is used' "$1/ocean.output" \
    && grep -q 'vector form dynamics : total vorticity = Coriolis + relative vorticity' "$1/ocean.output" \
    && grep -Eq 'jpkglo *= *2\b' "$1/ocean.output"
}

# The record length N, from the reference run: 100 if it finished, else the
# largest even step before stp_ctl's stop.  Refuses any other outcome.
record_steps() {
  local k
  [[ "$(sha_of "$REF/domain_cfg.nc")" == "$G1_DOMCFG_SHA" && "$(sha_of "$REF/nemo")" == "$REF_SHA" ]] && vec_logged "$REF" \
    && cmp -s "$REF/namelist_ref" "$NEMO_ROOT/cfgs/SHARED/namelist_ref" \
    || { printf 'REFUSE: %s is not the pinned G1b reference run\n' "$REF" >&2; exit 73; }
  cmp -s "$REF/namelist_cfg" "$DECK" || { printf 'REFUSE: the reference ran another deck\n' >&2; exit 73; }
  k=$(tr -d ' \n' <"$REF/time.step")
  if grep -qx 'RUN_DONE' "$REF/run.user.log" && [[ "$k" == 100 ]] && ! grep -q 'E R R O R' "$REF/ocean.output"; then
    printf '100'; return
  fi
  grep -q 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$REF/ocean.output" && [[ -f "$REF/output.abort.nc" ]] \
    && [[ "$k" =~ ^[0-9]+$ ]] && (( k > 10 && k <= 100 )) \
    || { printf 'REFUSE: %s neither finished 100 steps nor stopped in stp_ctl after kt = 10 (kt = %s)\n' "$REF" "$k" >&2; exit 73; }
  printf '%s' $(( (k - 1) / 2 * 2 ))
}

# $1 = first step, $2 = last step, rest = twin dirs.  Every step in the range
# needs a restart holding a time record with kt = that step; an empty shell
# (restart.f90:119's end-of-run clause) does not count.  Exit 1 if any is missing.
restart_cover() {
  python3 -I -c '
import sys
from pathlib import Path
import netCDF4
lo, hi, dirs = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
full = set()
for d in dirs:
    for f in sorted(Path(d).glob("ORCA2G1_*_restart.nc")):
        step = int(f.name.split("_")[1])
        with netCDF4.Dataset(f) as nc:
            if nc.dimensions["time_counter"].size >= 1 and int(nc["kt"][...]) == step:
                full.add(step)
missing = [s for s in range(lo, hi + 1) if s not in full]
print("RESTART_FULL", len(full), "MISSING", ",".join(map(str, missing)) or "none")
sys.exit(1 if missing else 0)
' "$@"
}

admit() {
  local n stage_steps spgts restarts dir j list step kt plant want
  n=$(record_steps)
  printf 'G1B_RECORD_STEPS %s (reference time.step %s)\n' "$n" "$(tr -d ' \n' <"$REF/time.step")"
  stage_steps=$(( n < STAGE_GATE ? n : STAGE_GATE ))
  spgts=$(tr , '\n' <<<"$SPGTS_GATE" | awk -v n="$n" '$1 <= n' | paste -sd, -)
  restarts=$(seq 10 10 "$n" | paste -sd, -)
  # provenance: every run directory holds the pinned domain file and binary,
  # finished at its own length, with the committed deck rendered for it
  for dir in "$SMOKE:$REF_SHA:2:$L_SMOKE" \
             $(for j in $(seq 1 "$N_TWINS"); do printf '%s:%s:%s:%s ' "$(twin_dir "$j")" "$RUN_SHA" "$n" "$(twin_list_of "$j" "$n")"; done); do
    IFS=: read -r dir bin steps list <<<"$dir"
    [[ -f "$dir/domain_cfg.nc" && "$(sha_of "$dir/domain_cfg.nc")" == "$G1_DOMCFG_SHA" ]] \
      && [[ -f "$dir/nemo" && "$(sha_of "$dir/nemo")" == "$bin" ]] && vec_logged "$dir" \
      && grep -qx 'RUN_DONE' "$dir/run.user.log" && [[ "$(tr -d ' \n' <"$dir/time.step")" == "$steps" ]] \
      && ! grep -q 'E R R O R' "$dir/ocean.output" \
      && cmp -s "$dir/namelist_ref" "$NEMO_ROOT/cfgs/SHARED/namelist_ref" \
      || { printf 'REFUSE: %s is not a finished pinned %s-step G1b run (with the pinned namelist_ref)\n' "$dir" "$steps" >&2; exit 73; }
    render_deck "$dry/admit_deck" "$steps" "$list"
    cmp -s "$dir/namelist_cfg" "$dry/admit_deck" \
      || { printf 'REFUSE: %s ran a deck that is not the committed one at its length and list\n' "$dir" >&2; exit 73; }
  done
  readonly TWIN_A=$(twin_dir 10) TWIN_B=$(twin_dir 1)
  for plant in truncation nonfinite missing-frame twin-ulp missing-step missing-substep twin-substep-inner; do
    if python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" \
         --steps "$n" --stage-steps "$stage_steps" --plant "$plant" \
         --substep-steps "$spgts" >"$EVIDENCE/manifest/plant_${plant}.log" 2>&1; then
      printf 'REFUSE: record plant %s stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/manifest/plant_${plant}.log" \
      || { printf 'REFUSE: record plant %s did not fire\n' "$plant" >&2; exit 72; }
  done
  python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" --reference "$REF" \
    --steps "$n" --stage-steps "$stage_steps" --restart-steps "$restarts" \
    --substep-steps "$spgts" --json "$EVIDENCE/g1b_round22_admission.json"
  # the frame fields are the writer patch's (what the record binary was built from)
  python3 -I - "$G1_DIR" "$STEP_PATCH" "$EVIDENCE/g1b_round22_admission.json" "$n" "$STAGE_GATE" <<'PY'
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import check_records
try:
    frame, stage, last = None, set(), set()
    for line in Path(sys.argv[2]).read_text().splitlines():
        if not line.startswith("+"):
            continue
        hit = re.search(r"CALL g1r_frame\( '(.)' \)", line)
        frame = hit.group(1) if hit else frame
        hit = re.search(r"CALL g1r_w2\( '(\w+)'", line)
        if hit:
            stage.add(f"{frame}_{hit.group(1)}")
            if frame == "f":
                last.add(f"f_{hit.group(1)}")
    admitted = json.loads(Path(sys.argv[3]).read_text())
    check_records.require(len(stage) > 40 and len(last) > 5, "the writer patch was not parsed")
    check_records.require(admitted["fields_stage"] == sorted(stage),
                          "a stage step lacks a field the writer writes")
    want_final = sorted(last) if int(sys.argv[4]) > int(sys.argv[5]) else []
    check_records.require(admitted["fields_final_only"] == want_final,
                          "the final-only steps do not hold the writer's final frame")
except check_records.Refuse as exc:
    sys.exit(f"REFUSE: {exc}")
print("G1B_FRAME_FIELDS_COMPLETE", len(stage), "per stage step")
PY
  # extra restart writes are passive: every twin's step frames are twin a's
  for j in $(seq 1 9) $(seq 11 "$N_TWINS"); do
    for kt in $(seq 1 "$n"); do
      cmp -s "$TWIN_A/oracle_g1step_kt$(printf '%08d' "$kt").bin" \
             "$(twin_dir "$j")/oracle_g1step_kt$(printf '%08d' "$kt").bin" \
        || { printf 'REFUSE: twin r%02d step frame kt=%s differs from twin a\n' "$j" "$kt" >&2; exit 73; }
    done
  done
  printf 'G1B_TWIN_FRAMES_IDENTICAL kt=1..%s (twins r01..r09, r11..r%s against r10)\n' "$n" "$N_TWINS"
  # plants (N = 100): NEMO's own empty shells at steps 1 (r01) and 92 (r02)
  # must not count, and the ten stride-10 twins alone must leave steps uncovered
  if [[ "$n" == 100 ]]; then
    for plant in "1 $(twin_dir 1)" "92 $(twin_dir 2)"; do
      read -r step dir <<<"$plant"
      [[ -f "$dir/ORCA2G1_$(printf '%08d' "$step")_restart.nc" ]] \
        || { printf 'REFUSE: restart plant file for step %s is absent\n' "$step" >&2; exit 72; }
      if restart_cover "$step" "$step" "$dir" >/dev/null; then
        printf 'REFUSE: restart plant (empty shell, step %s) stayed green\n' "$step" >&2; exit 72
      fi
    done
    # it must miss EXACTLY the steps r11..r13 were planned for (not "everything": a
    # glob over one concatenated nonexistent path also misses every step)
    want=$(printf '%s,%s,%s\n' "$L_R11" "$L_R12" "$L_R13" | tr , '\n' | grep -vx 0 | sort -n | paste -sd, -)
    plant=$(restart_cover 2 "$n" $(for j in $(seq 1 10); do twin_dir "$j"; done)) \
      && { printf 'REFUSE: restart plant (stride-10 twins only) stayed green\n' >&2; exit 72; }
    [[ "$plant" == *" MISSING $want" ]] \
      || { printf 'REFUSE: restart plant (stride-10 twins only) failed for the wrong reason: %s (want MISSING %s)\n' "$plant" "$want" >&2; exit 72; }
    printf 'G1B_RESTART_PLANT_STRIDE10 %s\n' "$plant"
  fi
  restart_cover 2 "$n" $(for j in $(seq 1 "$N_TWINS"); do twin_dir "$j"; done) \
    || { printf 'REFUSE: a step in 2..%s has no restart holding its record\n' "$n" >&2; exit 73; }
  printf 'G1B_RESTART_EVERY_STEP 2..%s (no step-1 restart can be written on this deck: restart.f90:119)\n' "$n"
  ( cd "$EVIDENCE" && sha256sum g1b_round22_admission.json r10/domain_cfg.nc r10/mesh_mask.nc \
      ref_100step/ocean.output ref_100step/namelist_ref r??/ORCA2G1_*_restart.nc ) >"$EVIDENCE/manifest/outputs.sha256"
  printf 'ACQUISITION_DONE %s N=%s\n' "$EVIDENCE" "$n"
}

admit_smoke_in_place() {
  [[ "$(sha_of "$SMOKE/domain_cfg.nc")" == "$G1_DOMCFG_SHA" && "$(sha_of "$SMOKE/nemo")" == "$REF_SHA" ]] \
    && vec_logged "$SMOKE" && grep -qx 'RUN_DONE' "$SMOKE/run.user.log" \
    && [[ "$(tr -d ' \n' <"$SMOKE/time.step")" == 2 && -f "$SMOKE/ORCA2G1_00000002_restart.nc" ]] \
    && ! grep -q 'E R R O R' "$SMOKE/ocean.output" \
    && cmp -s "$SMOKE/namelist_ref" "$NEMO_ROOT/cfgs/SHARED/namelist_ref" \
    || { printf 'REFUSE: %s is not the finished pinned 2-step smoke\n' "$SMOKE" >&2; exit 73; }
  render_deck "$dry/smoke_admit" 2 "$L_SMOKE"
  cmp -s "$SMOKE/namelist_cfg" "$dry/smoke_admit" \
    || { printf 'REFUSE: the smoke ran another deck\n' >&2; exit 73; }
}
if [[ "$mode" == --admit-existing ]]; then
  for target in "$SMOKE" "$REF" "$(twin_dir 1)" "$(twin_dir 10)"; do
    [[ -e "$target" ]] || { printf 'REFUSE: nothing to admit, %s is absent\n' "$target" >&2; exit 69; }
  done
  mkdir -p "$EVIDENCE/manifest"
  admit
  exit 0
fi
if [[ "$mode" == --preflight-only && -e "$EVIDENCE" ]]; then
  admit_smoke_in_place
  printf 'RESUME_READY smoke and %s-step reference admissible in place; run.sh --resume runs the absent twins of r01..r13\n' "$(record_steps)"
  exit 0
fi
[[ "$mode" == --resume || ! -e "$EVIDENCE" ]] || {
  printf 'REFUSE: evidence directory %s already exists; use --admit-existing or --resume\n' "$EVIDENCE" >&2; exit 69; }
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD: mpirun -np 1 ./nemo (%s) in %s/{smoke_2step,ref_100step}; then (%s) in %s/r01..r10 to the reference length N; then %s\n' \
    "$REF_NAME" "$EVIDENCE" "$RUN_NAME" "$EVIDENCE" "$CHECKER"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

if [[ "$mode" == --run ]]; then
mkdir -p "$EVIDENCE/manifest"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
sha256sum "$G1_DOMCFG" "$DECK" "$G1_DECK" "$ORCA2_DECK" "$CHECKER" "$STEP_PATCH" "$REF_BIN" "$RUN_BIN" \
  "$NEMO_ROOT/cfgs/SHARED/namelist_ref" >"$EVIDENCE/manifest/inputs.sha256"
fi

run_one() {            # $1 = binary, $2 = run dir, $3 = nn_itend, $4 = stocklist; returns NEMO's status
  local dir=$2 rc=0
  mkdir -p "$dir"
  render_deck "$dir/namelist_cfg" "$3" "$4"
  cp -L "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$dir/namelist_ref"
  cp "$G1_DOMCFG" "$dir/domain_cfg.nc"
  cp "$1" "$dir/nemo"
  (
    cd "$dir"
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.log
    mpirun -np 1 --oversubscribe ./nemo >>run.user.log 2>&1 || rc=$?
    printf 'RUN_FINISHED_UTC=%s\nRUN_EXIT=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>run.user.log
    [[ "$rc" -ne 0 ]] || printf 'RUN_DONE\n' >>run.user.log
  )
  vec_logged "$dir" \
    || { printf 'REFUSE: %s did not log the vector form and total vorticity\n' "$dir" >&2; exit 71; }
  # facts the receipt quotes, read from NEMO's own log (reported, not judged)
  grep -E 'nn_e *=|ln_bt_auto|jpiglo|jpjglo|jpkglo|ln_dynadv|nn_dynkeg|vector form|total vorticity|ln_dynvor|ln_zad_Aimp|G1_BUMP|stp_ctl' \
    "$dir/ocean.output" >"$dir/resolved_facts.log" || true
}
finished() {           # $1 = dir, $2 = steps
  grep -qx 'RUN_DONE' "$1/run.user.log" && ! grep -q 'E R R O R' "$1/ocean.output" \
    && [[ "$(tr -d ' \n' <"$1/time.step")" == "$2" ]] \
    || { grep -E '^ *kt +[0-9]+ \|' "$1/ocean.output" >&2 || true
         printf 'REFUSE: %s did not finish %s steps\n' "$1" "$2" >&2; exit 71; }
}

if [[ "$mode" == --resume ]]; then
  # the smoke and the reference must be the finished pinned runs of the
  # committed deck; a twin directory may hold only the deck the refused
  # rendering wrote (NEMO never started there) and is re-rendered
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit_resume"
  admit_smoke_in_place
  N=$(record_steps)
  printf 'G1B_RESUME smoke and reference admitted in place; N=%s\n' "$N"
  for j in $(seq 1 "$N_TWINS"); do    # every twin directory checked before any runs
    d=$(twin_dir "$j")
    [[ ! -e "$d" ]] || finished "$d" "$N"     # present = a finished run, admitted in place
  done
  for j in $(seq 1 "$N_TWINS"); do
    d=$(twin_dir "$j")
    if [[ -e "$d" ]]; then printf 'G1B_TWIN_ADMITTED_IN_PLACE %s\n' "$d"; continue; fi
    run_one "$RUN_BIN" "$d" "$N" "$(twin_list_of "$j" "$N")"
    finished "$d" "$N"
  done
  admit
  exit 0
fi
# ---- the smoke first ---------------------------------------------------------
run_one "$REF_BIN" "$SMOKE" 2 "$L_SMOKE"
finished "$SMOKE" 2
[[ -f "$SMOKE/ORCA2G1_00000002_restart.nc" ]] || { printf 'REFUSE: smoke wrote no step-2 restart\n' >&2; exit 71; }
printf 'G1B_SMOKE_OK %s\n' "$SMOKE"
# ---- the reference: it decides the record length N ---------------------------
run_one "$REF_BIN" "$REF" 100 "$L_FULL"
N=$(record_steps)
grep -E '^ *kt +[0-9]+ \|' "$REF/ocean.output" | tail -6 || true
printf 'G1B_REFERENCE_STEPS N=%s\n' "$N"
for j in $(seq 1 "$N_TWINS"); do
  run_one "$RUN_BIN" "$(twin_dir "$j")" "$N" "$(twin_list_of "$j" "$N")"
  finished "$(twin_dir "$j")" "$N"
done

# ---- admission -----------------------------------------------------------------
admit
