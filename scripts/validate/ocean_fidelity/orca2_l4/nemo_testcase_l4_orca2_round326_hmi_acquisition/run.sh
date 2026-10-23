#!/usr/bin/env bash
# ORCA2 round-326 acquisition: SI3's mean ice thickness hm_i as TKE reads it.
#
# Re-runs the rung-10 ten-step record (round 5, ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY)
# with one write-only hook in zdf_tke, just before tke_avn, whose nn_mxlice = 2
# branch reads hm_i (zdftke.f90:620-623).  One file per rank per step, kt 1..10.
# Admission: every restart, surface frame and step-entry frame byte-identical to
# round 5, then the gate (five plants must fire first).  The operator runs this
# file; the agent does not invoke makenemo or mpirun.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-326 hm_i acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R326HMI
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_tke_hmi_10step_np2
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly TKE_PATCH=$here/zdftke_round326.patch
readonly WRITER=$here/zdftke_round326_writer.F90
readonly GATE=$here/../nemo_testcase_l4_orca2_round326_hmi_gate.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round326_hmi_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l4_orca2_round326.md
readonly SOURCE_TKE=$NEMO_ROOT/src/OCE/ZDF/zdftke.F90

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 65; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 66;
  }
}

# --- preflight checks; each is a function so its plant can be run against it ---
# Whitespace-tolerant namelist-echo match (LOG-CHECK note): name\s*=\s*value.
echo_has() {  # echo_has <file> <name> <value>
  grep -E "(^|[[:space:]])$2[[:space:]]*=[[:space:]]*$3([[:space:]]|$)" "$1" >/dev/null
}
check_echo() {  # check_echo <ocean.output>
  echo_has "$1" nn_mxlice 2 && echo_has "$1" ln_mxl0 T && echo_has "$1" nn_fsbc 2 \
    && echo_has "$1" nn_mxl 3 && echo_has "$1" ln_tile F && echo_has "$1" nn_it000 1
}
nml_int() {  # nml_int <namelist> <name>: last uncommented assignment
  sed -E 's/!.*//' "$1" | grep -E "^[[:space:]]*$2[[:space:]]*=" | tail -n 1 \
    | sed -E "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*([-0-9]+).*/\1/"
}
check_namelist() {  # (3) nn_itend = 10 and nn_itend, nn_stock multiples of nn_fsbc
  local itend stock fsbc
  itend=$(nml_int "$1" nn_itend); stock=$(nml_int "$1" nn_stock); fsbc=$(nml_int "$1" nn_fsbc)
  [[ -n "$itend" && -n "$stock" && -n "$fsbc" ]] || return 1
  (( itend == 10 && itend % fsbc == 0 && stock % fsbc == 0 ))
}
check_ref_cfg() {  # (2) makenemo -r name is a reference config
  grep -E "^$1[[:space:]]" "$NEMO_ROOT/cfgs/ref_cfgs.txt" >/dev/null
}
check_no_continuation_insert() {  # (1) no inserted line follows a line ending in '&'
  awk '/^[ +]/ { line=substr($0,2); if (substr($0,1,1)=="+" && prev ~ /&[[:space:]]*$/) bad=1; prev=line } END { exit bad }' "$1"
}
free_ok() {  # (6) free_ok <path> [kB, default 8 GB] on the nearest existing ancestor
  local p=$1 need=${2:-8388608}
  while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done
  (( $(df -Pk "$p" | awk 'NR==2 {print $4}') >= need ))
}
check_absent() { [[ ! -e "$1" ]]; }  # (7)

plant_must_refuse() {  # plant_must_refuse <label> <cmd...>
  local label=$1; shift
  if "$@"; then printf 'REFUSE: preflight plant stayed green: %s\n' "$label" >&2; exit 72; fi
  printf 'PREFLIGHT_PLANT_FIRED %s\n' "$label"
}

for path in "$TKE_PATCH" "$WRITER" "$GATE" "$TEST" "$PREREG" "$SOURCE_TKE" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" \
  "$SOURCE_RUN/ocean.output" "$SOURCE_RUN/namelist_cfg"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing input %s\n' "$path" >&2; exit 64; }
done
pin 2e78c1a215e4a7ee51dab4b3b19ef446cceb1ef8e5ad9380dd7c482d37e10397 "$SOURCE_TKE" 'source zdftke'
pin 0d4360984a193ff9bce2cfbaa13a45dbd26fcd18b215e0159c95254075830630 "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source build binary'
pin 0d4360984a193ff9bce2cfbaa13a45dbd26fcd18b215e0159c95254075830630 "$SOURCE_RUN/nemo" 'source run binary'
pin 51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9 "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin 3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5 "$SOURCE_RUN/input_files.sha256" 'input manifest'
[[ ! -e "$SOURCE_ROOT/MY_SRC/zdftke.F90" ]] || { printf 'REFUSE: source config carries its own zdftke\n' >&2; exit 66; }

plants=$(mktemp -d /tmp/orca2-r326-plants.XXXXXX)
# (1) syntax/continuation: real patch passes, a planted insert after '&' refuses.
check_no_continuation_insert "$TKE_PATCH" || { printf 'REFUSE: patch inserts after a continuation\n' >&2; exit 66; }
printf ' x = a + &\n+ CALL planted()\n' >"$plants/cont.patch"
plant_must_refuse continuation check_no_continuation_insert "$plants/cont.patch"
# (2) reference config.
check_ref_cfg "$REFERENCE_CFG" || { printf 'REFUSE: %s not in ref_cfgs.txt\n' "$REFERENCE_CFG" >&2; exit 66; }
plant_must_refuse ref-cfg check_ref_cfg ORCA2_OMIP_L4
# (3) namelist rules on the deck the run copies unchanged.
check_namelist "$SOURCE_RUN/namelist_cfg" || { printf 'REFUSE: namelist breaks nn_itend/nn_stock/nn_fsbc rules\n' >&2; exit 65; }
sed -E 's/^([[:space:]]*nn_itend[[:space:]]*=[[:space:]]*)[0-9]+/\19/' "$SOURCE_RUN/namelist_cfg" >"$plants/namelist_cfg"
plant_must_refuse namelist check_namelist "$plants/namelist_cfg"
# Resolved selectors in the REAL ocean.output (LOG-CHECK note), and a plant.
check_echo "$SOURCE_RUN/ocean.output" || { printf 'REFUSE: source run echo lacks nn_mxlice=2/ln_mxl0=T/nn_fsbc=2/nn_mxl=3/ln_tile=F/nn_it000=1\n' >&2; exit 65; }
sed -E 's/(nn_mxlice[[:space:]]*=[[:space:]]*)2/\10/' "$SOURCE_RUN/ocean.output" >"$plants/ocean.output"
plant_must_refuse echo check_echo "$plants/ocean.output"
# (4)+(5) reader/writer layout and frame census: the committed test writes the
# writer's byte layout from its Fortran source and proves the gate's plants.
"$PY" -m pytest -q -p no:cacheprovider "$TEST"
# (6) disk, evaluated without creating the evidence directory.
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_ok "$mount" || { printf 'REFUSE: %s has under 8 GB free\n' "$mount" >&2; exit 67; }
done
free_ok "$EVIDENCE/not/created/yet" || { printf 'REFUSE: disk check fails on a missing parent\n' >&2; exit 67; }
plant_must_refuse disk free_ok /tmp 999999999999999
# (7) target config and run directory absent (not required in --admit-existing).
if [[ "$MODE" != --admit-existing ]]; then
  check_absent "$TARGET_ROOT" && check_absent "$TARGET_RUN" || {
    printf 'REFUSE: target config or run directory already exists\n' >&2; exit 64; }
fi
plant_must_refuse target-exists check_absent "$SOURCE_RUN"

# (1) gfortran syntax proof of the patched zdftke and the writer.
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$TKE_PATCH")
[[ "$removed" -eq 0 ]] || { printf 'REFUSE: additions-only patch removes %s lines\n' "$removed" >&2; exit 66; }
dry=$(mktemp -d /tmp/orca2-r326-tke.XXXXXX)
cp "$SOURCE_TKE" "$dry/zdftke.F90"
cp "$WRITER" "$dry/zdftke_round326_writer.F90"
patch -s --fuzz=0 -p0 -d "$dry" <"$TKE_PATCH"
[[ "$(grep -Fc 'CALL r326_write(' "$dry/zdftke.F90")" -eq 1 ]] || { printf 'REFUSE: writer call count changed\n' >&2; exit 66; }
[[ "$(grep -Fc 'CALL tke_avn(' "$dry/zdftke.F90")" -eq 1 ]] || { printf 'REFUSE: closure call count changed\n' >&2; exit 66; }
syntax=$(mktemp -d /tmp/orca2-r326-syntax.XXXXXX)
cp "$SOURCE_ROOT"/BLD/inc/*.mod "$syntax/"
for unit in zdftke_round326_writer zdftke; do
  cpp -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
    -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$dry/$unit.F90" -o "$syntax/$unit.f90"
  "$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" -J "$syntax" "$syntax/$unit.f90"
  printf 'SYNTAX_PROOF_PASS %s.f90\n' "$unit"
done
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND326_HMI_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant name
  [[ "$(cat "$TARGET_RUN/producer_commit.txt")" == "$COMMIT" ]] || { printf 'REFUSE: producer commit changed\n' >&2; exit 70; }
  # Write-only hook: the trajectory and every round-5 frame must be unchanged.
  for name in ORCA2_00000010_restart_0000.nc ORCA2_00000010_restart_0001.nc \
    ORCA2_00000010_restart_ice_0000.nc ORCA2_00000010_restart_ice_0001.nc; do
    cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name" || { printf 'REFUSE: trajectory changed: %s\n' "$name" >&2; exit 69; }
  done
  for kt in 1 2 3 4 5 6 7 8 9 10; do
    for name in $(printf 'oracle_ocean_surface_input_kt%08d.bin oracle_ocean_surface_input_rank0001_kt%08d.bin oracle_step_entry_kt%08d.bin oracle_step_entry_rank0001_kt%08d.bin' "$kt" "$kt" "$kt" "$kt"); do
      cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name" || { printf 'REFUSE: frame changed: %s\n' "$name" >&2; exit 69; }
    done
  done
  for plant in header field-order truncation stamp content; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" \
      --plant "$plant" >"$TARGET_RUN/round326_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep 'STATUS PLANT-FIRED' "$TARGET_RUN/round326_${plant}_plant.log" >/dev/null
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" --surface-root "$SOURCE_RUN" \
    --output "$TARGET_RUN/round326_hmi_admission.json"
  ( cd "$TARGET_RUN" && sha256sum oracle_tke_hmi_kt*.bin ORCA2_00000010_restart*.nc \
      round326_hmi_admission.json round326_*_plant.log >round326_outputs.sha256 )
  printf 'ORCA2_ROUND326_HMI_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$BINARY" && -x "$TARGET_RUN/nemo" ]] || { printf 'REFUSE: existing target lacks binary\n' >&2; exit 68; }
  cmp -s "$BINARY" "$TARGET_RUN/nemo" || { printf 'REFUSE: staged binary differs\n' >&2; exit 68; }
  admit
  exit 0
fi

manifest=$(mktemp -d /tmp/orca2-r326-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$TKE_PATCH" "$WRITER" "$GATE" "$TEST" "$PREREG" "$SOURCE_TKE" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/zdftke.f90" "$SOURCE_ROOT/BLD/ppsrc/nemo/icevar.f90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/icestp.f90" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$SOURCE_TKE" "$TARGET_ROOT/MY_SRC/zdftke.F90"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/zdftke_round326_writer.F90"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$TKE_PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 68; }
grep 'NEMO_L4_HMI_1' "$TARGET_ROOT/BLD/ppsrc/nemo/zdftke_round326_writer.f90" >/dev/null || {
  printf 'REFUSE: compiled writer is absent\n' >&2; exit 68;
}
nm -D "$BINARY" >"$manifest/nm_dynamic.txt"
if grep '_ZGV' "$manifest/nm_dynamic.txt" >/dev/null; then printf 'REFUSE: vector math symbol present\n' >&2; exit 68; fi

mkdir -p "$EVIDENCE"
mkdir "$TARGET_RUN"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/producer_commit.txt "$manifest"/toolchain.sha256 "$TARGET_RUN/"
( cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null )

(
  cd "$TARGET_RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  mpirun -np 2 --oversubscribe ./nemo >run.user.stdout.log 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 0 ]] || { printf 'REFUSE: mpirun exited %s\n' "$rc" >&2; exit 69; }
  printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
grep 'STOP 0' "$TARGET_RUN/run.user.stdout.log" >/dev/null || { printf 'REFUSE: NEMO did not report STOP 0\n' >&2; exit 69; }
[[ "$(grep -c 'ORCA2_R326_TKE_HMI' "$TARGET_RUN/ocean.output" || true)" -ge 10 ]] || {
  printf 'REFUSE: writer did not report ten steps on rank 0\n' >&2; exit 69; }
admit
