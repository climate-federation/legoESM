#!/usr/bin/env bash
set -euo pipefail

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
# READ THIS BEFORE YOU RUN IT.  That last switch is also the reason the legoESM
# VORTEX card is NOT execution-ready: on flux-form momentum NEMO applies the
# energy-and-enstrophy scheme to the planetary vorticity alone, so on this case
# it IS the Coriolis operator, and legoESM has no such arm (it binds its own
# implementation to vector-invariant momentum).  The card declares that gap and
# its execution gate refuses it.  This acquisition is still worth running: the
# initial state does not depend on the momentum scheme, so the bit-exactness
# comparison is live today, and the record is what the round that builds the
# missing arm will measure against.
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
#   * TEOS-10 instead of the shipped S-EOS (campaign decision 64, note BF).
#     rn_a0 = 0.28 is LEFT IN PLACE because usrdef_istate.F90:19,88 reads it
#     from nameos to build the initial temperature, and eosbn2.F90:1890-1895
#     reads nameos whichever EOS is selected.
#   * nn_itend = nn_stock = 10 (the tanks' own kt1_10 cadence) instead of 3000.
#   * ln_meshmask = .true., so the geometry receipt is written.
#
# PATH.  FCM's extract step needs perl's Text::Balanced, which the system perl
# lacks; the conda build environment must come first, exactly as the tanks'
# round-33 acquisition documents.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round1}
readonly TEST_CASE=VORTEX
readonly REF_CFG=VORTEX_OMIP_L1
readonly RUN_CFG=VORTEX_OMIP_L1_P3
readonly RESTART=VORTEX_OMIP_L1_ZCO_00000010_restart.nc
readonly STEPS=10

do_run=0
case "${1:-}" in
  --run) do_run=1 ;;
  "") ;;
  *) printf 'Usage: %s [--run]\n' "$0" >&2 ; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly INSTRUMENT=$here/stprk3_step_record.patch
readonly DECK=$here/namelist_cfg_omip_l1.patch
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
grep -q 'ln_teos10   = .true.' "$dry/namelist_cfg" \
  || { printf 'REFUSE: the patched deck does not select TEOS-10 (decision 64)\n' >&2
       rm -rf "$dry"; exit 67; }
grep -q 'rn_a0       =  0.28' "$dry/namelist_cfg" \
  || { printf 'REFUSE: rn_a0 was dropped; usrdef_istate needs it for T\n' >&2
       rm -rf "$dry"; exit 67; }
# The card transcribes ln_zad_Aimp = .false., which this deck gets by LEAVING
# IT UNSET.  The tanks' own campaign decks set it .true.; if anyone copies that
# line in here the card and the oracle stop agreeing on the vertical momentum
# scheme, silently.  Refuse instead.
if grep -q 'ln_zad_Aimp' "$dry/namelist_cfg"; then
  printf 'REFUSE: the deck now sets ln_zad_Aimp; the card transcribes the unset default\n' >&2
  rm -rf "$dry"; exit 67
fi
rm -rf "$dry"
python "$CHECKER" --help >/dev/null \
  || { printf 'REFUSE: the record checker does not run\n' >&2; exit 67; }
printf 'PREFLIGHT_OK  instrument and deck patches apply to the shipped sources\n'
printf '  reference config : %s\n  instrumented cfg : %s\n  evidence         : %s\n' \
  "$ref_cfg" "$run_cfg" "$EVIDENCE"
printf '  deck deviations  :\n'
sed -n 's/^/    /p' "$DECK" | grep -E '^\s+[-+][^-+]' || true

if [[ "$do_run" -eq 0 ]]; then
  printf '\nDRY RUN.  Re-run with --run to build and acquire.\n'
  exit 0
fi

# ------------------------------------------------------------------ acquire
for target in "$ref_cfg" "$run_cfg" "$EVIDENCE"; do
  if [[ -e "$target" ]]; then
    printf 'REFUSE: target already exists: %s\n' "$target" >&2
    exit 64
  fi
done
for mount in /tmp "$(dirname "$EVIDENCE")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 2097152 ]]; then
    printf 'REFUSE: %s has %s kB free, under the 2 GB floor\n' "$mount" "$free_kb" >&2
    exit 68
  fi
done

manifest=$(mktemp -d /tmp/vortex-r1-provenance.XXXXXX)
printf 'provenance directory (retained): %s\n' "$manifest"
(
  cd "$SRC_CASE"
  find EXPREF MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/shipped_case.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$SRC_CASE/cpp_${TEST_CASE}.fcm" "$SHIPPED_STP" "$SHIPPED_CFG" \
  "$INSTRUMENT" "$DECK" "$CHECKER" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
build_one() {          # $1 = config name, $2 = 1 to apply the instrument
  local name=$1 instrumented=$2 cfg=$NEMO_ROOT/tests/$1
  ./makenemo -a "$TEST_CASE" -n "$name" -m conda-scalarmath \
    del_key 'key_xios key_agrif'
  # cp -r, NOT cp -a: preserved mtimes let fcm skip a patched file.
  cp -r "$SRC_CASE/EXPREF/." "$cfg/EXP00/"
  cp -r "$SRC_CASE/MY_SRC/." "$cfg/MY_SRC/"
  # The AGRIF child deck is meaningless without key_agrif; remove it so it
  # cannot be read by accident.
  rm -f "$cfg/EXP00/1_"* "$cfg/EXP00/AGRIF_FixedGrids.in"
  patch "$cfg/EXP00/namelist_cfg" <"$DECK"
  if [[ "$instrumented" -eq 1 ]]; then
    [[ ! -e "$cfg/MY_SRC/stprk3.F90" ]]
    cp "$SHIPPED_STP" "$cfg/MY_SRC/stprk3.F90"
    patch "$cfg/MY_SRC/stprk3.F90" <"$INSTRUMENT"
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
  if grep -q 'NEMO_L1_ENTRY_1' "$cfg/BLD/ppsrc/nemo/stprk3.f90"; then
    if [[ "$instrumented" -ne 1 ]]; then
      printf 'REFUSE: the REFERENCE build carries the writer\n' >&2; exit 69
    fi
  elif [[ "$instrumented" -eq 1 ]]; then
    printf 'REFUSE: the writer is absent from %s ppsrc (stale build)\n' \
      "$name" >&2; exit 69
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
}

build_one "$REF_CFG" 0
build_one "$RUN_CFG" 1
sha256sum "$ref_cfg/BLD/bin/nemo.exe" "$run_cfg/BLD/bin/nemo.exe" \
  >"$manifest/binaries.sha256"
# The two decks must be the SAME deck; only the compiled writer may differ.
cmp "$ref_cfg/EXP00/namelist_cfg" "$run_cfg/EXP00/namelist_cfg"

run_one "$REF_CFG" "$EVIDENCE/reference"
run_one "$RUN_CFG" "$EVIDENCE"
cp "$manifest"/*.sha256 "$EVIDENCE/"

# ADMISSION.  The checker parses every record's own header (note BD) and
# refuses unless the two restarts are byte-identical (note AS).  Its plant
# MUST turn it red, or it proves nothing.
python "$CHECKER" --run-dir "$EVIDENCE" --reference-dir "$EVIDENCE/reference" \
  --restart "$RESTART" --steps "$STEPS" \
  --output "$EVIDENCE/vortex_round1_admission.json"
if python "$CHECKER" --run-dir "$EVIDENCE" --reference-dir "$EVIDENCE/reference" \
     --restart "$RESTART" --steps "$STEPS" --plant \
     >"$EVIDENCE/vortex_round1_admission_plant.json" 2>&1; then
  printf 'REFUSE: the planted control did not turn the checker red\n' >&2
  exit 70
fi
(
  cd "$EVIDENCE"
  sha256sum oracle_*.bin vortex_round1_admission.json "$RESTART" mesh_mask.nc \
    >vortex_round1_outputs.sha256
)
printf 'VORTEX_ROUND1_KT1_10_ORACLE_READY %s\n' "$EVIDENCE"
