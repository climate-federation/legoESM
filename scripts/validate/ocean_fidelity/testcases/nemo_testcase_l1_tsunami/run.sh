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

# TSUNAMI lane round 1 ACQUISITION -- USER-EXECUTED ONLY.
#
#   preflight (default):  run.sh
#   acquire:              run.sh --run
#
# The agent that wrote this did not run NEMO.  Without --run nothing is built
# and nothing is written outside a temporary directory.
#
# WHAT THIS ACQUIRES.  NEMO 5.0.2 tests/TSUNAMI exactly as shipped: 201x201
# doubly periodic f-plane, ONE 100 m level, 100 steps of 1000 s.  Its cpp keys
# are key_qco key_xios key_vco_1d and NOT key_RK3, so NEMO runs the case's own
# MY_SRC/stpmlf.F90: the split-explicit external mode alone (dyn_spg, MLF
# branch of dynspg_ts).  This script REFUSES if the shipped keys change.
#
# TWO BUILDS (passivity, the VORTEX pattern):
#   TSUNAMI_OMIP_L1      shipped MY_SRC only           -> the reference run
#   TSUNAMI_OMIP_L1_P3   + read-only record writers    -> the records
# TSUNAMI's step never calls rst_write, so there is no restart to compare; the
# admission instead requires NEMO's OWN ssh / barotropic-velocity output
# (dia_wri, every nn_write = 5 steps) to be bit-identical between the two.
#
# DELIBERATE DEVIATIONS, all visible in the committed patches:
#   * key_xios is dropped (no XIOS in the conda-scalarmath toolchain); the
#     shipped stpmlf only guards iom_context_finalize with it.
#   * ln_meshmask = .true. (geometry receipt).
# Run length (nn_itend = 100) and every physics switch are the shipped values.
#
# RECORDS (P3 only):
#   oracle_tsustep_kt########.bin  every kt = 1..100: ssh/uu_b/vv_b at Naa
#                                  after dyn_spg; kt = 1..10 also the three
#                                  ssh/uu_b/vv_b/r3 time-level slots at entry,
#                                  r3(Naa) after dom_qco_r3c, un_adv/vn_adv
#                                  and the level-1 3-D velocity slots.
#   oracle_spgts_kt########.bin    kt = 1..10: every barotropic substep
#                                  (the VORTEX round-196 writer, gated).
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly TEST_CASE=TSUNAMI
readonly REF_NAME=TSUNAMI_OMIP_L1
readonly RUN_NAME=TSUNAMI_OMIP_L1_P3
readonly STEPS=100
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round1/oracle_tsunami_r1}
readonly SHIPPED_KEYS='key_qco key_xios key_vco_1d'

do_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) do_run=1 ;;
    *) printf 'Usage: %s [--run]\n' "$0" >&2; exit 64 ;;
  esac
  shift
done

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
vortex=$here/../nemo_testcase_l1_vortex
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run.sh is not inside a git worktree\n' >&2; exit 66; }
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: legoESM worktree is dirty; commit the exact acquisition tool first\n' >&2
  exit 66
fi
readonly COMMIT=$(git -C "$repo_root" rev-parse HEAD)

readonly SRC_CASE=$NEMO_ROOT/tests/$TEST_CASE
readonly SHIPPED_STPMLF=$SRC_CASE/MY_SRC/stpmlf.F90
readonly SHIPPED_SPGTS=$NEMO_ROOT/src/OCE/DYN/dynspg_ts.F90
readonly DECK=$here/namelist_cfg_tsunami_l1.patch
readonly STEP_PATCH=$here/stpmlf_step_record.patch
readonly STEP_MODULE=$here/tsunami_r1_step_record.F90
readonly SPGTS_PATCH=$here/dynspg_ts_substep_record_kt10.patch
readonly SPGTS_MODULE=$vortex/vortex_r12_spgts_terms.F90
readonly CHECKER=$here/check_records.py

for f in "$SRC_CASE/cpp_${TEST_CASE}.fcm" "$SRC_CASE/EXPREF/namelist_cfg" \
         "$SHIPPED_STPMLF" "$SHIPPED_SPGTS" "$DECK" "$STEP_PATCH" \
         "$STEP_MODULE" "$SPGTS_PATCH" "$SPGTS_MODULE" "$CHECKER" \
         "$NEMO_ROOT/makenemo" "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm"; do
  [[ -f "$f" ]] || { printf 'REFUSE: missing input %s\n' "$f" >&2; exit 67; }
done

# The step program the card transcribes is the one these keys select
# (nemogcm.F90:183-187).  A key_RK3 would silently switch to stprk3.
keys=$(sed -E 's/.*fppkeys *//' "$SRC_CASE/cpp_${TEST_CASE}.fcm" | xargs)
if [[ "$keys" != "$SHIPPED_KEYS" ]]; then
  printf 'REFUSE: cpp_%s.fcm keys are "%s", expected "%s"; re-survey the case\n' \
    "$TEST_CASE" "$keys" "$SHIPPED_KEYS" >&2
  exit 68
fi

# Additive patches only: every one must apply, unfuzzed, to a fresh copy.
dry=$(mktemp -d)
trap 'rm -rf "$dry"' EXIT
cp "$SRC_CASE/EXPREF/namelist_cfg" "$SHIPPED_STPMLF" "$SHIPPED_SPGTS" "$dry/"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$DECK"
patch -s --fuzz=0 "$dry/stpmlf.F90" <"$STEP_PATCH"
patch -s --fuzz=0 "$dry/dynspg_ts.F90" <"$SPGTS_PATCH"
# The writers only ADD lines; the deck patch replaces exactly one (meshmask).
removed=$(cat "$STEP_PATCH" "$SPGTS_PATCH" | grep -c '^-[^-]' || true)
[[ "$removed" -eq 0 ]] \
  || { printf 'REFUSE: an instrument patch removes %s line(s)\n' "$removed" >&2; exit 68; }
[[ "$(grep -c '^-[^-]' "$DECK")" -eq 1 ]] \
  || { printf 'REFUSE: the deck patch changes more than ln_meshmask\n' >&2; exit 68; }
grep -q 'ln_meshmask =  .true.' "$dry/namelist_cfg"
grep -q 'nn_itend    =     100' "$dry/namelist_cfg"
printf 'PREFLIGHT: patches apply; keys "%s"; commit %s\n' "$keys" "$COMMIT"

for name in "$REF_NAME" "$RUN_NAME"; do
  if [[ -e "$NEMO_ROOT/tests/$name" ]]; then
    printf 'REFUSE: %s already exists; moving it aside is the operator'"'"'s call\n' \
      "$NEMO_ROOT/tests/$name" >&2
    exit 69
  fi
done
if [[ -e "$EVIDENCE" ]]; then
  printf 'REFUSE: evidence directory %s already exists\n' "$EVIDENCE" >&2
  exit 69
fi

if [[ "$do_run" -ne 1 ]]; then
  printf 'WOULD: makenemo -a %s -n %s|%s -m conda-scalarmath del_key key_xios\n' \
    "$TEST_CASE" "$REF_NAME" "$RUN_NAME"
  printf 'WOULD: mpirun -np 1 ./nemo in %s/{ref,p3}; then %s\n' "$EVIDENCE" "$CHECKER"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

mkdir -p "$EVIDENCE/manifest"
(
  cd "$SRC_CASE"
  find EXPREF MY_SRC cpp_${TEST_CASE}.fcm -type f -print0 | sort -z | xargs -0 sha256sum
) >"$EVIDENCE/manifest/shipped_case.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$SHIPPED_SPGTS" \
  "$DECK" "$STEP_PATCH" "$STEP_MODULE" "$SPGTS_PATCH" "$SPGTS_MODULE" \
  "$CHECKER" >"$EVIDENCE/manifest/toolchain.sha256"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"

cd "$NEMO_ROOT"
build_one() {          # $1 = config name, $2 = 1 to apply the writers
  local name=$1 instrumented=$2 cfg=$NEMO_ROOT/tests/$1
  ./makenemo -a "$TEST_CASE" -n "$name" -m conda-scalarmath del_key 'key_xios'
  # cp -r, NOT cp -a: preserved mtimes let fcm skip a patched file.
  cp -r "$SRC_CASE/EXPREF/." "$cfg/EXP00/"
  cp -r "$SRC_CASE/MY_SRC/." "$cfg/MY_SRC/"
  patch "$cfg/EXP00/namelist_cfg" <"$DECK"
  if [[ "$instrumented" -eq 1 ]]; then
    cp -f "$SHIPPED_SPGTS" "$cfg/MY_SRC/dynspg_ts.F90"
    cp -f "$SPGTS_MODULE" "$STEP_MODULE" "$cfg/MY_SRC/"
    patch "$cfg/MY_SRC/stpmlf.F90" <"$STEP_PATCH"
    patch "$cfg/MY_SRC/dynspg_ts.F90" <"$SPGTS_PATCH"
  fi
  touch "$cfg/MY_SRC/"*.F90
  ./makenemo -n "$name" -m conda-scalarmath
  [[ -x "$cfg/BLD/bin/nemo.exe" ]]
  local pp=$cfg/BLD/ppsrc/nemo
  grep -q 'CALL stp_MLF' "$pp/nemogcm.f90" \
    || { printf 'REFUSE: %s does not call stp_MLF\n' "$name" >&2; exit 70; }
  if grep -q 'CALL stp_RK3' "$pp/nemogcm.f90"; then
    printf 'REFUSE: %s compiled the RK3 step program\n' "$name" >&2; exit 70
  fi
  grep -q 'TSUNAMI' "$pp/usrdef_hgr.f90" \
    || { printf 'REFUSE: %s did not compile the TSUNAMI usrdef_hgr\n' "$name" >&2; exit 70; }
  if grep -q 'tsu_open' "$pp/stpmlf.f90"; then
    [[ "$instrumented" -eq 1 ]] \
      || { printf 'REFUSE: the REFERENCE build carries the writer\n' >&2; exit 70; }
  else
    [[ "$instrumented" -eq 0 ]] \
      || { printf 'REFUSE: the instrumented build lost the writer\n' >&2; exit 70; }
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
    # No pipe: a failing mpirun must abort under set -e, not hide in a pipe.
    mpirun -np 1 --oversubscribe ./nemo >>run.user.log 2>&1
    tail -n 20 run.user.log
    printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.log
  )
  grep -qE 'nn_e *= *6 *$' "$dir/ocean.output" \
    || { printf 'REFUSE: %s did not resolve nn_e = 6 (card prediction)\n' "$1" >&2; exit 71; }
  if grep -q 'E R R O R' "$dir/ocean.output"; then
    printf 'REFUSE: %s ocean.output reports an error\n' "$1" >&2; exit 71
  fi
}

build_one "$REF_NAME" 0
build_one "$RUN_NAME" 1
sha256sum "$NEMO_ROOT/tests/$REF_NAME/BLD/bin/nemo.exe" \
  "$NEMO_ROOT/tests/$RUN_NAME/BLD/bin/nemo.exe" >"$EVIDENCE/manifest/binaries.sha256"
run_one "$REF_NAME" "$EVIDENCE/ref"
run_one "$RUN_NAME" "$EVIDENCE/p3"

python3 -I "$CHECKER" --evidence "$EVIDENCE/p3" --reference "$EVIDENCE/ref" \
  --steps "$STEPS" --json "$EVIDENCE/tsunami_round1_admission.json"
( cd "$EVIDENCE" && sha256sum tsunami_round1_admission.json p3/mesh_mask*.nc ) \
  >"$EVIDENCE/manifest/outputs.sha256"
printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
