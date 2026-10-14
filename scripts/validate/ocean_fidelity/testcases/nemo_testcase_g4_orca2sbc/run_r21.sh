#!/usr/bin/env bash
# Grid ladder G3 + G4 RE-ACQUISITION on the Decision-120 decks (gridladder_r21)
# -- OPERATOR-EXECUTED ONLY.
#
#   preflight (default):  run_r21.sh
#   acquire:              run_r21.sh --run
#   re-admit:             run_r21.sh --admit-existing
#
# The agent that wrote this ran neither makenemo nor NEMO.
#
# WHAT CHANGED.  Decision 120: the G3 and G4 decks set ln_tsd_dmp = .true.
# with ln_tradmp = .false., so NEMO applies the ORCA_R2 hand edits to the
# initial T/S (dtatsd.f90:219) and never restores (dta_tsd's only per-step
# caller is tradmp.f90:181).  Each deck is its admitted as-built deck
# (commit AS_BUILT) with that ONE assignment moved; preflight checks it.
#
# NO BUILD.  Three admitted binaries, pinned:
#   ORCA2G1_OMIP_L1_R4REF  reference binary (no writer): smokes, references;
#   ORCA2G1_OMIP_L1_P5     record binary (stage frames, substeps kt = 1, 2);
#   ORCA2G1_OMIP_L1_P6     P5 + G4 surface-forcing writer (round 19, admitted
#                          round 20).
# Inputs: each run copies its rung's files from the as-built admitted run
# (G3: round 15 twin a; G4: round 17 twin a) -- the decks are the only change.
#
# (a) G3, as round 15: smoke (2 steps, R4REF), the 100-step reference (R4REF)
#     whose stp_ctl stop is NEMO's boundary, twins a/b/c (P5, STEPS steps:
#     restart STEPS / even 2..STEPS / odd 3..STEPS-1).  A step-1 restart
#     cannot exist on this deck: a listed step opens at step-1 = 0
#     (restart.f90:119), and nn_stock = 1 or a 1-step run is stopped by
#     sbc_init with nn_fsbc = 2 (sbcmod.f90:341-349).
# (b) G4, as rounds 17 + 19: the P5 set (smoke, reference, p5_a/b/c) and the
#     P6 set (smoke6, p6_a/b/c) at the same STEPS; admission = round 17's
#     checks on the P5 set, round 19's forcing checks on the P6 set, and
#     round 19's passivity (P6 against P5, on the consumed support).
# STEPS = min(10, steps the reference completed), down to a multiple of
# nn_fsbc = 2, per rung; a stop outside stp_ctl is refused.
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE: unhandled command failure at run_r21.sh line %s (exit %s)\n' "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_SHA=5ed4b6e543f32b827ee42e818704f836dd322768a17bbe4f40e8ce5a7ff523b3
readonly P5_SHA=9b96f1427503da2773ce6c17621dd431c3261792d11a39c218f64d88e3f00778
readonly P6_SHA=a32c5b8026f03bdfb576887fd085b4b4e5d002607a970f3b70257fd39e8c9979
readonly REF_BIN=$NEMO_ROOT/cfgs/ORCA2G1_OMIP_L1_R4REF/BLD/bin/nemo.exe
readonly P5_BIN=$NEMO_ROOT/cfgs/ORCA2G1_OMIP_L1_P5/BLD/bin/nemo.exe
readonly P6_BIN=$NEMO_ROOT/cfgs/ORCA2G1_OMIP_L1_P6/BLD/bin/nemo.exe
readonly BULK_ITERATIONS=5 FSBC=2 RECORD_SPGTS=1,2
readonly ROUNDS=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds
readonly R15_RUN=$ROUNDS/round15/oracle_g3_r15/p3_a_10step R17_RUN=$ROUNDS/round17/oracle_g4_r17/p3_a
readonly R19_BIN_SHA_FILE=$ROUNDS/round19/oracle_g4_r19/manifest/binary.sha256
readonly EVIDENCE=${EVIDENCE:-$ROUNDS/round21}
readonly G3_EV=$EVIDENCE/oracle_g3_r21 G4_EV=$EVIDENCE/oracle_g4_r21
readonly AS_BUILT=ad1ba285e37a
readonly G3_AS_BUILT_SHA=5b2699e8e20c49bd0b8cce6a8dfcf3286821001a2079a3318a6a0f77fed215a6
readonly G4_AS_BUILT_SHA=3a45cc6cd3bbe776424c47ba02700a08b8ac576162307de5819d8e539a989353
readonly G3_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc namelist_ref)
readonly G4_FILES=(data_1m_potential_temperature_nomask.nc data_1m_salinity_nomask.nc
  u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc
  q_10.15JUNE2009_fill.nc ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc sss_data.nc
  runoff_core_monthly.nc domain_cfg.nc namelist_ref)

mode=--preflight-only
case "${1:---preflight-only}" in
  --run|--preflight-only|--admit-existing) mode=${1:---preflight-only} ;;
  *) printf 'Usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run_r21.sh is not inside a git worktree\n' >&2; exit 66; }
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] \
  || { printf 'REFUSE: legoESM worktree is dirty; commit the exact acquisition tool first\n' >&2; exit 66; }
readonly COMMIT=$(git -C "$repo_root" rev-parse HEAD)

readonly G3_REL=scripts/validate/ocean_fidelity/testcases/nemo_testcase_g3_orca2ts/namelist_cfg
readonly G4_REL=scripts/validate/ocean_fidelity/testcases/nemo_testcase_g4_orca2sbc/namelist_cfg
readonly G3_DECK=$repo_root/$G3_REL G4_DECK=$repo_root/$G4_REL
readonly G1_DIR=$here/../nemo_testcase_g1_orca2grid
readonly CHECKER=$G1_DIR/check_records.py
readonly STEP_PATCH=$G1_DIR/stprk3_g1_step_record_stage50.patch
readonly ORCA2_DECK=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg

for f in "$G3_DECK" "$G4_DECK" "$CHECKER" "$STEP_PATCH" "$REF_BIN" "$P5_BIN" "$P6_BIN" \
         "$R19_BIN_SHA_FILE" "$ORCA2_DECK"; do
  [[ -f "$f" ]] || { printf 'REFUSE: missing input %s\n' "$f" >&2; exit 67; }
done
for f in "${G3_FILES[@]}"; do [[ -f "$R15_RUN/$f" ]] || { printf 'REFUSE: missing %s/%s\n' "$R15_RUN" "$f" >&2; exit 67; }; done
for f in "${G4_FILES[@]}"; do [[ -f "$R17_RUN/$f" ]] || { printf 'REFUSE: missing %s/%s\n' "$R17_RUN" "$f" >&2; exit 67; }; done
sha_of() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(sha_of "$REF_BIN")" == "$REF_SHA" && "$(sha_of "$P5_BIN")" == "$P5_SHA" \
   && "$(sha_of "$P6_BIN")" == "$P6_SHA" && "$(awk '{print $1}' "$R19_BIN_SHA_FILE")" == "$P6_SHA" ]] \
  || { printf 'REFUSE: a binary is not the admitted one (rebuilt?)\n' >&2; exit 67; }

# ---- the decks: the as-built deck with ONE assignment moved, ln_tsd_dmp -----
assignments() { sed -e 's/!.*//' -e 's/[[:space:]]//g' "$1" | grep -v '^$'; }
block() { assignments "$1" | sed -n "/^&$2\$/,/^\\/\$/p"; }
dry=$(mktemp -d)
trap 'rm -rf "$dry"' EXIT
for pair in "$G3_REL:$G3_AS_BUILT_SHA" "$G4_REL:$G4_AS_BUILT_SHA"; do
  rel=${pair%%:*}
  git -C "$repo_root" show "$AS_BUILT:$rel" >"$dry/as_built"
  [[ "$(sha_of "$dry/as_built")" == "${pair#*:}" ]] \
    || { printf 'REFUSE: %s at %s is not the admitted as-built deck\n' "$rel" "$AS_BUILT" >&2; exit 68; }
  [[ "$(diff <(assignments "$dry/as_built") <(assignments "$repo_root/$rel") | grep -E '^[<>]' | xargs)" \
     == '< ln_tsd_dmp=.false. > ln_tsd_dmp=.true.' ]] \
    || { printf 'REFUSE: %s moves more than ln_tsd_dmp against the as-built deck\n' "$rel" >&2; exit 68; }
  [[ "$(block "$repo_root/$rel" namtsd)" == "$(block "$ORCA2_DECK" namtsd)" ]] \
    || { printf 'REFUSE: %s &namtsd is not ORCA2s own block\n' "$rel" >&2; exit 68; }
  assignments "$repo_root/$rel" | grep -qx 'ln_tradmp=.false.' \
    || { printf 'REFUSE: %s does not state ln_tradmp = .false.\n' "$rel" >&2; exit 68; }
done
grep -Eq '^ *nn_fsbc *= *'"$FSBC"'( |$)' "$G4_DECK" && ! grep -q 'nn_iter_algo' "$G4_DECK" \
  && grep -Eq '^ *nn_iter_algo *= *'"$BULK_ITERATIONS"' ' "$R17_RUN/namelist_ref" \
  || { printf 'REFUSE: nn_fsbc / nn_iter_algo do not resolve to %s / %s\n' "$FSBC" "$BULK_ITERATIONS" >&2; exit 68; }

render_deck() {        # $1 = target, $2 = deck, $3 = nn_itend, $4 = nn_stocklist
  sed -E -e "s/^(   nn_itend    = ).*/\1    $3/" \
         -e "s/^(   nn_stock    = ).*/\1    $3/" \
         -e "s/^(   nn_stocklist = ).*/\1$4/" "$2" >"$1"
  [[ -z "$(diff "$2" "$1" | grep '^>' | grep -vE '^>    (nn_itend    =|nn_stock    =|nn_stocklist =)' || true)" ]] \
    || { printf 'REFUSE: rendering changed a deck line other than the 3 run-length lines\n' >&2; exit 68; }
}
pad10() { local l=("$@"); while ((${#l[@]} < 10)); do l+=(0); done; local IFS=,; printf '%s' "${l[*]}"; }
readonly L_SMOKE=$(pad10 2) L_REF='2,4,6,8,10,12,14,16,18,20'
list_a() { pad10 "$1"; }
list_b() { pad10 $(seq 2 2 "$1"); }
list_c() { (($1 >= 4)) && pad10 $(seq 3 2 $(($1 - 1))) || pad10 0; }
for deck in "$G3_DECK" "$G4_DECK"; do
  render_deck "$dry/full" "$deck" 100 '10,20,30,40,50,60,70,80,90,100'
  cmp -s "$dry/full" "$deck" || { printf 'REFUSE: %s is not the 100-step deck\n' "$deck" >&2; exit 68; }
done
python3 -I -m py_compile "$CHECKER"
printf 'PREFLIGHT: decks = as-built (%s) + ln_tsd_dmp = T; namtsd = ORCA2s; three admitted binaries; commit %s\n' \
  "$AS_BUILT" "$COMMIT"

rung_deck() { [[ "$1" == g3 ]] && printf '%s' "$G3_DECK" || printf '%s' "$G4_DECK"; }
rung_src() { [[ "$1" == g3 ]] && printf '%s' "$R15_RUN" || printf '%s' "$R17_RUN"; }
rung_files() { if [[ "$1" == g3 ]]; then printf '%s\n' "${G3_FILES[@]}"; else printf '%s\n' "${G4_FILES[@]}"; fi; }

# NEMO's own log: T/S read WITH the hand-edit branch live (cn_cfg/nn_cfg and
# ln_tsd_dmp, dtatsd.f90:218-219), no restoring, 31 levels; G4 also its
# forcing switches as resolved (sbcmod.f90 prints).
logged() {             # $1 = dir, $2 = rung
  local o=$1/ocean.output
  grep -Eq 'ln_tsd_init *= *T' "$o" && grep -Eq 'ln_tsd_dmp *= *T' "$o" \
    && grep -Eq 'ln_tradmp *= *F' "$o" && grep -Eq 'cn_cfg = ORCA +nn_cfg = +2$' "$o" \
    && ! grep -q 'dta_tsd: deallocte T & S arrays' "$o" && grep -Eq 'jpkglo *= *31' "$o" \
    || return 1
  [[ "$2" == g3 ]] && return 0
  grep -Eq 'ln_usr +=  F' "$o" && grep -Eq 'ln_blk +=  T' "$o" \
    && grep -Eq 'nn_ice +=  +0$' "$o" && grep -Eq 'ln_traqsr +=  T' "$o" \
    && grep -Eq 'ln_ssr +=  T' "$o" && grep -Eq 'nn_fwb +=  +2$' "$o" && grep -Eq 'ln_rnf +=  T' "$o"
}

# The reference's completed steps: NEMO's boundary.  Finished = 100; a stop
# is admitted only from stp_ctl (stpctl.f90), with its abort state.
completed_steps() {    # $1 = reference dir
  local t
  t=$(tr -d ' \n' <"$1/time.step")
  if grep -qx 'RUN_DONE' "$1/run.user.log"; then
    [[ "$t" == 100 ]] && ! grep -q 'E R R O R' "$1/ocean.output" \
      || { printf 'REFUSE: %s finished at %s, not 100, or with an error\n' "$1" "$t" >&2; exit 73; }
    printf '100'; return
  fi
  grep -q 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$1/ocean.output" && [[ -f "$1/output.abort.nc" ]] \
    || { printf 'REFUSE: %s stopped at kt = %s outside stp_ctl\n' "$1" "$t" >&2; exit 73; }
  printf '%s' $((t - 1))
}
record_steps() {       # $1 = reference dir
  local c s
  c=$(completed_steps "$1")
  s=$(( c < 10 ? c : 10 )); s=$(( s - s % FSBC ))
  ((s >= 2)) || { printf 'REFUSE: NEMO completed %s steps; no record of 2 steps exists\n' "$c" >&2; exit 73; }
  printf '%s' "$s"
}

run_one() {            # $1 = binary, $2 = dir, $3 = rung, $4 = nn_itend, $5 = list, $6 = done|stop
  local dir=$2 f rc=0
  mkdir -p "$dir"
  render_deck "$dir/namelist_cfg" "$(rung_deck "$3")" "$4" "$5"
  while read -r f; do cp "$(rung_src "$3")/$f" "$dir/$f"; done < <(rung_files "$3")
  cp "$1" "$dir/nemo"
  (
    cd "$dir"
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.log
    mpirun -np 1 --oversubscribe ./nemo >>run.user.log 2>&1 || rc=$?
    printf 'RUN_FINISHED_UTC=%s\nRUN_EXIT=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>run.user.log
    [[ "$rc" -ne 0 ]] || printf 'RUN_DONE\n' >>run.user.log
  )
  [[ -f "$dir/ocean.output" && -f "$dir/time.step" ]] \
    || { printf 'REFUSE: %s wrote no ocean.output/time.step\n' "$dir" >&2; exit 71; }
  logged "$dir" "$3" || { printf 'REFUSE: %s did not resolve the deck (see ocean.output)\n' "$dir" >&2; exit 71; }
  grep -E 'nn_e *=|jpkglo|cn_cfg|tsd|tradmp|fld_|ln_NCAR|nn_sssr|ln_rnf_mouth|nn_fwb' \
    "$dir/ocean.output" >"$dir/resolved_facts.log" || true
  if [[ "$6" == done ]] && { ! grep -qx 'RUN_DONE' "$dir/run.user.log" || grep -q 'E R R O R' "$dir/ocean.output" \
       || [[ "$(tr -d ' \n' <"$dir/time.step")" != "$4" ]]; }; then
    grep -E '^ *kt +[0-9]+ \|' "$dir/ocean.output" >&2 || true
    printf 'REFUSE: %s stopped at kt = %s before nn_itend = %s\n' "$dir" "$(tr -d ' \n' <"$dir/time.step")" "$4" >&2
    exit 71
  fi
}

# one run directory, judged: its rung's admitted inputs, its pinned binary,
# the committed deck at its length and list, NEMO's log, every listed restart
judge_dir() {          # $1 = dir, $2 = rung, $3 = binary sha, $4 = nn_itend, $5 = list, $6 = done|stop
  local dir=$1 f step
  while read -r f; do
    cmp -s "$dir/$f" "$(rung_src "$2")/$f" || { printf 'REFUSE: %s/%s is not the admitted input\n' "$dir" "$f" >&2; exit 73; }
  done < <(rung_files "$2")
  [[ -f "$dir/nemo" && "$(sha_of "$dir/nemo")" == "$3" ]] \
    || { printf 'REFUSE: %s was not run with its pinned binary\n' "$dir" >&2; exit 73; }
  render_deck "$dry/admit_deck" "$(rung_deck "$2")" "$4" "$5"
  cmp -s "$dir/namelist_cfg" "$dry/admit_deck" \
    || { printf 'REFUSE: %s ran a deck that is not the committed one at its length and list\n' "$dir" >&2; exit 73; }
  logged "$dir" "$2" || { printf 'REFUSE: %s ocean.output does not state the deck\n' "$dir" >&2; exit 73; }
  if [[ "$6" == done ]]; then
    grep -qx 'RUN_DONE' "$dir/run.user.log" && [[ "$(tr -d ' \n' <"$dir/time.step")" == "$4" ]] \
      && ! grep -q 'E R R O R' "$dir/ocean.output" \
      || { printf 'REFUSE: %s is not a finished %s-step run\n' "$dir" "$4" >&2; exit 73; }
  fi
  for step in ${5//,/ }; do
    [[ "$step" == 0 || "$step" -gt "$(tr -d ' \n' <"$dir/time.step")" \
       || -f "$dir/ORCA2G1_$(printf '%08d' "$step")_restart.nc" ]] \
      || { printf 'REFUSE: %s has no restart at listed step %s\n' "$dir" "$step" >&2; exit 73; }
  done
}

plants() {             # $1 = manifest dir, $2 = twin a, $3 = twin b, $4 = steps, $5.. = checker args then plants after --
  local man=$1 a=$2 b=$3 steps=$4 plant args=()
  shift 4
  while [[ "$1" != -- ]]; do args+=("$1"); shift; done
  shift
  for plant in "$@"; do
    if python3 -I "$CHECKER" --evidence "$a" --twin "$b" --steps "$steps" --stage-steps "$steps" \
         "${args[@]}" --plant "$plant" >"$man/plant_${plant}.log" 2>&1; then
      printf 'REFUSE: record plant %s stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$man/plant_${plant}.log" \
      || { printf 'REFUSE: record plant %s did not fire\n' "$plant" >&2; exit 72; }
  done
}

# round 15 / round 17: the writer's stage fields complete, twin restarts and
# frames passive, every step 2..STEPS restart present
p5_checks() {          # $1 = a, $2 = b, $3 = c, $4 = smoke, $5 = admission json, $6 = steps
  python3 -I - "$G1_DIR" "$1" "$2" "$STEP_PATCH" "$5" "$3" "$4" "$6" <<'PY'
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
    check_records.restarts_identical(Path(sys.argv[7]), Path(sys.argv[3]), [2])
    for step in range(2, steps + 1):
        twin = Path(sys.argv[3] if step % 2 == 0 else sys.argv[6])
        check_records.require((twin / f"ORCA2G1_{step:08d}_restart.nc").is_file(),
                              f"no twin restart at step {step}")
except check_records.Refuse as exc:
    sys.exit(f"REFUSE: {exc}")
print("FRAME_FIELDS_COMPLETE", len(stage), "per stage step")
print(f"TWIN_RESTARTS_PASSIVE step {steps} b = a, step 2 b = smoke; every step 2..{steps} present")
PY
  local kt twin
  for twin in "$2" "$3"; do
    for kt in $(seq 1 "$6"); do
      cmp -s "$1/oracle_g1step_kt$(printf '%08d' "$kt").bin" "$twin/oracle_g1step_kt$(printf '%08d' "$kt").bin" \
        || { printf 'REFUSE: %s step frame kt=%s differs from twin a\n' "$twin" "$kt" >&2; exit 73; }
    done
  done
  printf 'TWIN_FRAMES_IDENTICAL kt=1..%s (%s, %s)\n' "$6" "$(basename "$2")" "$(basename "$3")"
}

P5_PLANTS=(truncation nonfinite missing-frame twin-ulp missing-step missing-substep twin-substep-inner)
P6_PLANTS=(truncation nonfinite missing-frame twin-ulp missing-step
  forcing-missing-iteration forcing-nonfinite forcing-twin-inner forcing-missing-field)

admit_g3() {
  local ev=$G3_EV steps
  steps=$(record_steps "$ev/ref_100step")
  printf 'G3_BOUNDARY reference completed %s steps; record = kt 1..%s\n' "$(completed_steps "$ev/ref_100step")" "$steps"
  grep -E '^ *kt +[0-9]+ \|' "$ev/ref_100step/ocean.output" | tail -6 || true
  mkdir -p "$ev/manifest"
  judge_dir "$ev/smoke_2step" g3 "$REF_SHA" 2 "$L_SMOKE" done
  judge_dir "$ev/ref_100step" g3 "$REF_SHA" 100 "$L_REF" stop
  judge_dir "$ev/p3_a" g3 "$P5_SHA" "$steps" "$(list_a "$steps")" done
  judge_dir "$ev/p3_b" g3 "$P5_SHA" "$steps" "$(list_b "$steps")" done
  judge_dir "$ev/p3_c" g3 "$P5_SHA" "$steps" "$(list_c "$steps")" done
  plants "$ev/manifest" "$ev/p3_a" "$ev/p3_b" "$steps" --substep-steps "$RECORD_SPGTS" -- "${P5_PLANTS[@]}"
  python3 -I "$CHECKER" --evidence "$ev/p3_a" --twin "$ev/p3_b" --reference "$ev/ref_100step" \
    --steps "$steps" --stage-steps "$steps" --restart-steps "$steps" \
    --substep-steps "$RECORD_SPGTS" --json "$ev/g3_round21_admission.json"
  p5_checks "$ev/p3_a" "$ev/p3_b" "$ev/p3_c" "$ev/smoke_2step" "$ev/g3_round21_admission.json" "$steps"
  ( cd "$ev" && sha256sum g3_round21_admission.json p3_a/mesh_mask.nc p3_*/ORCA2G1_*_restart.nc \
      smoke_2step/ORCA2G1_00000002_restart.nc ref_100step/ORCA2G1_*_restart.nc ref_100step/ocean.output \
      p3_a/oracle_*.bin ) >"$ev/manifest/outputs.sha256"
  [[ ! -f "$ev/ref_100step/output.abort.nc" ]] || sha256sum "$ev/ref_100step/output.abort.nc" >>"$ev/manifest/outputs.sha256"
  printf 'G3_ACQUISITION_DONE %s\n' "$ev"
}

admit_g4() {
  local ev=$G4_EV steps twin
  steps=$(record_steps "$ev/ref_100step")
  printf 'G4_BOUNDARY reference completed %s steps; record = kt 1..%s\n' "$(completed_steps "$ev/ref_100step")" "$steps"
  grep -E '^ *kt +[0-9]+ \|' "$ev/ref_100step/ocean.output" | tail -6 || true
  mkdir -p "$ev/manifest"
  [[ "$(sha_of "$ev/p6_a/nemo")" == "$P6_SHA" ]] || { printf 'REFUSE: p6_a is not the P6 binary\n' >&2; exit 73; }
  judge_dir "$ev/smoke_2step" g4 "$REF_SHA" 2 "$L_SMOKE" done
  judge_dir "$ev/ref_100step" g4 "$REF_SHA" 100 "$L_REF" stop
  judge_dir "$ev/smoke6_2step" g4 "$P6_SHA" 2 "$L_SMOKE" done
  for twin in a b c; do
    judge_dir "$ev/p5_$twin" g4 "$P5_SHA" "$steps" "$(list_$twin "$steps")" done
    judge_dir "$ev/p6_$twin" g4 "$P6_SHA" "$steps" "$(list_$twin "$steps")" done
  done
  # round 17's checks on the P5 set
  plants "$ev/manifest" "$ev/p5_a" "$ev/p5_b" "$steps" --substep-steps "$RECORD_SPGTS" -- \
    "${P5_PLANTS[@]}" substep-inner-nonfinite
  python3 -I "$CHECKER" --evidence "$ev/p5_a" --twin "$ev/p5_b" --reference "$ev/ref_100step" \
    --steps "$steps" --stage-steps "$steps" --restart-steps "$steps" \
    --substep-steps "$RECORD_SPGTS" --json "$ev/g4_round21_p5_admission.json"
  p5_checks "$ev/p5_a" "$ev/p5_b" "$ev/p5_c" "$ev/smoke_2step" "$ev/g4_round21_p5_admission.json" "$steps"
  # round 19's checks on the P6 set
  plants "$ev/manifest" "$ev/p6_a" "$ev/p6_b" "$steps" --bulk-iterations "$BULK_ITERATIONS" --fsbc "$FSBC" -- \
    "${P6_PLANTS[@]}"
  python3 -I "$CHECKER" --evidence "$ev/p6_a" --twin "$ev/p6_b" --reference "$ev/ref_100step" \
    --steps "$steps" --stage-steps "$steps" --restart-steps "$steps" \
    --bulk-iterations "$BULK_ITERATIONS" --fsbc "$FSBC" \
    --substep-steps "$RECORD_SPGTS" --json "$ev/g4_round21_forcing_admission.json"
  python3 -I "$CHECKER" --evidence "$ev/p6_a" --twin "$ev/p6_c" --steps "$steps" \
    --stage-steps "$steps" --bulk-iterations "$BULK_ITERATIONS" --fsbc "$FSBC" \
    --json "$ev/g4_round21_twin_c.json"
  python3 -I "$CHECKER" --evidence "$ev/smoke6_2step" --steps 2 --stage-steps 2 \
    --bulk-iterations "$BULK_ITERATIONS" --fsbc "$FSBC" --json "$ev/g4_round21_smoke.json"
  # round 19's passivity: every P5 word is reproduced by P6 (substep streams
  # on the consumed support; halo storage NEMO never sets is counted)
  python3 -I - "$G1_DIR" "$ev" "$steps" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import check_records as cr
import netCDF4
ev, steps = Path(sys.argv[2]), int(sys.argv[3])
halo, band = {}, {}
try:
    for twin in "abc":
        old, new = ev / f"p5_{twin}", ev / f"p6_{twin}"
        for step in range(1, steps + 1):
            _, before = cr.read_stream(cr.step_path(old, step).read_bytes(), f"p5 {twin} {step}")
            _, after = cr.read_stream(cr.step_path(new, step).read_bytes(), f"p6 {twin} {step}",
                                      allow_inner=True)
            for name, data in before.items():
                cr.require(name in after and data.tobytes() == after[name].tobytes(),
                           f"twin {twin} kt={step}: {name} moved against P5")
        for path in sorted(old.glob("ORCA2G1_*_restart.nc")):
            cr.restarts_identical(old, new, [int(path.name.split("_")[1])])
        for path in sorted(old.glob("ORCA2G1_300h_*.nc")):
            with netCDF4.Dataset(path) as a, netCDF4.Dataset(new / path.name) as b:
                for var in a.variables:
                    cr.require(var in b.variables and a[var][:].tobytes() == b[var][:].tobytes(),
                               f"twin {twin}: {path.name} {var} moved against P5")
        for path in sorted(old.glob("oracle_g1spgts_kt*.bin")):
            header, before = cr.read_substeps(path.read_bytes(), f"p5 {twin} {path.name}")
            header_new, after = cr.read_substeps((new / path.name).read_bytes(), f"p6 {twin} {path.name}")
            cr.require(header_new == header, f"twin {twin}: {path.name} header moved against P5")
            halo[f"{twin}/{path.name}"] = cr.twin_halo_words(
                header, before, after, f"twin {twin}: {path.name} against P5")
            js, je, i0, ie = header["Njs0"] - 1, header["Nje0"], header["Nis0"] - 1, header["Nie0"]
            band[f"{twin}/{path.name}"] = sum(
                int(((before[n].view("<u8") != after[n].view("<u8"))[js - 1:je + 1, i0 - 1:ie + 1]).sum())
                for n in before if before[n].shape == (header["jpj"], header["jpi"]))
    cr.restarts_identical(ev / "smoke_2step", ev / "smoke6_2step", [2])
    try:
        cr.twin_halo_words(header, before, after, "plant", plant=True)
        sys.exit("REFUSE: the passivity plant stayed green")
    except cr.Refuse:
        pass
except cr.Refuse as exc:
    sys.exit(f"REFUSE: {exc}")
(ev / "g4_round21_passivity.json").write_text(json.dumps(
    {"substep_halo_words_differing": halo, "of_which_adjacent_band": band,
     "plant": "fired"}, indent=1) + "\n")
print(f"G4R21_PASSIVE every P5 step frame kt=1..{steps}, restart and diagnostic of twins a, b, c "
      "and the smoke is byte-identical under P6; substep streams identical on the consumed "
      f"support ({sum(halo.values())} halo words differ, counted in g4_round21_passivity.json)")
PY
  ( cd "$ev" && sha256sum g4_round21_*.json p5_a/mesh_mask.nc p5_*/ORCA2G1_*_restart.nc \
      p6_*/ORCA2G1_*_restart.nc smoke_2step/ORCA2G1_00000002_restart.nc ref_100step/ORCA2G1_*_restart.nc \
      ref_100step/ocean.output p6_a/oracle_*.bin p6_a/ocean.output ) >"$ev/manifest/outputs.sha256"
  [[ ! -f "$ev/ref_100step/output.abort.nc" ]] || sha256sum "$ev/ref_100step/output.abort.nc" >>"$ev/manifest/outputs.sha256"
  printf 'G4_ACQUISITION_DONE %s\n' "$ev"
}

readonly RUN_DIRS=("$G3_EV"/{smoke_2step,ref_100step,p3_a,p3_b,p3_c}
                   "$G4_EV"/{smoke_2step,ref_100step,smoke6_2step,p5_a,p5_b,p5_c,p6_a,p6_b,p6_c})
if [[ "$mode" == --admit-existing ]]; then
  for target in "${RUN_DIRS[@]}"; do
    [[ -e "$target" ]] || { printf 'REFUSE: nothing to admit, %s is absent\n' "$target" >&2; exit 69; }
  done
  admit_g3
  admit_g4
  printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
  exit 0
fi
for target in "$G3_EV" "$G4_EV"; do
  [[ ! -e "$target" ]] || { printf 'REFUSE: %s already exists; use --admit-existing\n' "$target" >&2; exit 69; }
done
if [[ "$mode" != --run ]]; then
  printf 'WOULD: mpirun -np 1 ./nemo in %s\n' "${RUN_DIRS[*]}"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

for ev in "$G3_EV" "$G4_EV"; do
  mkdir -p "$ev/manifest"
  printf '%s\n' "$COMMIT" >"$ev/manifest/legoesm_commit"
  sha256sum "$G3_DECK" "$G4_DECK" "$CHECKER" "$STEP_PATCH" "$REF_BIN" "$P5_BIN" "$P6_BIN" \
    >"$ev/manifest/inputs.sha256"
done

# ---- smokes first (both rungs, both record writers) ------------------------
run_one "$REF_BIN" "$G3_EV/smoke_2step" g3 2 "$L_SMOKE" done
run_one "$REF_BIN" "$G4_EV/smoke_2step" g4 2 "$L_SMOKE" done
run_one "$P6_BIN" "$G4_EV/smoke6_2step" g4 2 "$L_SMOKE" done
python3 -I "$CHECKER" --evidence "$G4_EV/smoke6_2step" --steps 2 --stage-steps 2 \
  --bulk-iterations "$BULK_ITERATIONS" --fsbc "$FSBC" \
  || { printf 'REFUSE: the P6 smoke record is not admitted; nothing else runs\n' >&2; exit 71; }
printf 'R21_SMOKES_OK\n'

# ---- NEMO's boundary per rung, then the twins at its length ----------------
run_one "$REF_BIN" "$G3_EV/ref_100step" g3 100 "$L_REF" stop
g3_steps=$(record_steps "$G3_EV/ref_100step")
run_one "$REF_BIN" "$G4_EV/ref_100step" g4 100 "$L_REF" stop
g4_steps=$(record_steps "$G4_EV/ref_100step")
printf 'R21_BOUNDARIES G3 record kt 1..%s, G4 record kt 1..%s\n' "$g3_steps" "$g4_steps"
for twin in a b c; do
  run_one "$P5_BIN" "$G3_EV/p3_$twin" g3 "$g3_steps" "$(list_$twin "$g3_steps")" done
done
for twin in a b c; do
  run_one "$P5_BIN" "$G4_EV/p5_$twin" g4 "$g4_steps" "$(list_$twin "$g4_steps")" done
  run_one "$P6_BIN" "$G4_EV/p6_$twin" g4 "$g4_steps" "$(list_$twin "$g4_steps")" done
done

admit_g3
admit_g4
printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
