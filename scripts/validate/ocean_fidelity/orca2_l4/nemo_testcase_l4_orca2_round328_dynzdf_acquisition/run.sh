#!/usr/bin/env bash
# Operator-only passive acquisition. Default mode is a read-only preflight;
# this agent never invokes --run because the round contract reserves NEMO runs
# for the operator.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-328 dyn_zdf acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---preflight-only}
case "$MODE" in
  --preflight-only|--run|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R326HMI
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R328ZDF
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BASELINE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_dynzdf_kt1_np2
readonly SOURCE_DYNZDF=$NEMO_ROOT/src/OCE/DYN/dynzdf.F90
readonly SOURCE_STAGE=$SOURCE_ROOT/MY_SRC/stprk3_stg.F90
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly CONDA=/home/dbalwada/miniconda3/envs/nemo-build
readonly MEM_MIN_GB=150
readonly RUN_TIMEOUT_S=900
readonly FPP_KEYS='-Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3'
readonly SOURCE_DYNZDF_SHA=4489773bb4cafe46123cfd658b1561188215c9ce0db11f6859fb27cb019bb174
readonly SOURCE_STAGE_SHA=2b1636336acc5a5e3326291801c217f263f8ccc3ca875dbfec0d3b789d911e5b
readonly SOURCE_BINARY_SHA=20376f40a78cb1a5491838ff112bfbe738178e4da8da1e43d5bd5c3884c72934
readonly DECK_MANIFEST_SHA=51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9
readonly INPUT_MANIFEST_SHA=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
readonly OCEAN_OUTPUT_SHA=b1e6c021d79013f3303582034350574fae2f0b1e68c976caf3af4061ac51a80f

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/dynzdf_round328.patch
readonly STAGE_PATCH=$here/stprk3_stg_round328.patch
readonly WRITER=$here/dynzdf_round328_writer.F90
readonly FIXTURE_STUBS=$here/dynzdf_round328_fixture_stubs.F90
readonly FIXTURE=$here/dynzdf_round328_layout_fixture.F90
readonly EXCERPT=$here/round326_ocean.output.excerpt
readonly GATE=$here/../nemo_testcase_l4_orca2_round328_dynzdf_gate.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round328_dynzdf_gate.py
readonly R326_TEST=$REPO/tests/ocean/fidelity/test_orca2_round326_hmi_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l4_orca2_round328.md
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe

scratch=
created_target=0
created_run=0
run_completed=0
cleanup() {
  local rc=$? refused
  trap - EXIT
  set +e
  [[ -z "$scratch" || ! -e "$scratch" ]] || rm -rf -- "$scratch"
  if [[ $rc -ne 0 ]]; then
    if [[ $created_run -ne 0 && $run_completed -eq 1 && -e "$TARGET_RUN" ]]; then
      refused=$EVIDENCE/refused/$(basename "$TARGET_RUN")_admission_refused_$(date -u +%Y%m%dT%H%M%SZ)_$$
      if mkdir -p -- "$EVIDENCE/refused" && mv -- "$TARGET_RUN" "$refused"; then
        printf 'R328_ADMISSION_REFUSED_OUTPUTS_KEPT %s\n' "$refused" >&2
      else
        printf 'R328_ADMISSION_REFUSED_OUTPUTS_LEFT_IN_PLACE %s\n' "$TARGET_RUN" >&2
      fi
    else
      [[ $created_run -eq 0 || ! -e "$TARGET_RUN" ]] || rm -rf -- "$TARGET_RUN"
    fi
    [[ $created_target -eq 0 || ! -e "$TARGET_ROOT" ]] || rm -rf -- "$TARGET_ROOT"
    [[ $created_run -eq 0 && $created_target -eq 0 ]] || \
      printf 'R328_FAILED_INVOCATION_HANDLED target=%s run=%s completed=%s\n' \
        "$created_target" "$created_run" "$run_completed" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT

refuse() { printf 'REFUSE: %s\n' "$*" >&2; exit 65; }
plant_must_refuse() {
  local label=$1; shift
  if "$@"; then refuse "preflight plant stayed green: $label"; fi
  printf 'PREFLIGHT_PLANT_FIRED %s\n' "$label"
}
pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || refuse "missing $label: $path"
  [[ $(sha256sum "$path" | awk '{print $1}') == "$digest" ]] \
    || refuse "$label SHA-256 changed: $path"
}
echo_has() {
  grep -E "(^|[[:space:]])$2[[:space:]]*=[[:space:]]*$3([[:space:]]|$)" "$1" >/dev/null
}
check_echo() {
  echo_has "$1" nn_it000 1 && echo_has "$1" nn_itend 10 && echo_has "$1" ln_tile F \
    && echo_has "$1" ln_drgimp T && echo_has "$1" ln_drgice_imp T \
    && echo_has "$1" ln_dynadv_vec T && echo_has "$1" ln_dynspg_ts T \
    && echo_has "$1" ln_zad_Aimp F && echo_has "$1" nn_mxlice 2 \
    && echo_has "$1" jpl 1
}
nml_int() {
  sed -E 's/!.*//' "$1" | grep -E "^[[:space:]]*$2[[:space:]]*=" | tail -n 1 \
    | sed -E "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*([-0-9]+).*/\1/"
}
check_namelist() {
  local itend stock fsbc
  itend=$(nml_int "$1" nn_itend); stock=$(nml_int "$1" nn_stock); fsbc=$(nml_int "$1" nn_fsbc)
  [[ -n "$itend" && -n "$stock" && -n "$fsbc" ]] || return 1
  (( itend == 10 && itend % fsbc == 0 && stock % fsbc == 0 ))
}
check_ref_cfg() { grep -E "^$1[[:space:]]" "$NEMO_ROOT/cfgs/ref_cfgs.txt" >/dev/null; }
check_no_continuation_insert() {
  awk '/^[ +]/ { kind=substr($0,1,1); line=substr($0,2); if (kind=="+" && prev_kind==" " && prev ~ /&[[:space:]]*$/) bad=1; prev=line; prev_kind=kind } END { exit bad }' "$1"
}
check_no_removed_lines() { [[ $(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$1") -eq 0 ]]; }
free_ok() {
  local p=$1 need=${2:-8388608}
  while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done
  (( $(df -Pk "$p" | awk 'NR == 2 {print $4}') >= need ))
}
check_absent() { [[ ! -e "$1" ]]; }
mem_ok() { (( $(free -g | awk '$1 == "Mem:" {print $7}') >= $1 )); }
fpp() {
  ( cd "$(dirname "$1")" && cpp -Dkey_nosignedzero -P -traditional $FPP_KEYS \
      -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$(basename "$1")" -o "$2" )
}
same_program() {
  "$PY" "$GATE" --same-program "$1" --reference-bld "$SOURCE_ROOT/BLD" \
    --dynzdf-patch "$PATCH" --stage-patch "$STAGE_PATCH" \
    --new-cfg "$TARGET_CFG" --reference-cfg "$SOURCE_CFG" ${2:+--output "$2"}
}
compare_deterministic() {
  local reference=$1 candidate=$2 source name count=0
  while IFS= read -r -d '' source; do
    name=$(basename "$source")
    [[ -f "$candidate/$name" ]] || return 1
    cmp -s "$source" "$candidate/$name" || return 1
    count=$((count+1))
  done < <(find "$reference" -maxdepth 1 -type f \
    \( -name 'oracle_*.bin' -o -name 'ORCA2_00000010_restart*.nc' \) -print0)
  (( count > 0 )) || return 1
  printf 'DETERMINISTIC_PASSIVITY_PASS %s files=%s\n' "$candidate" "$count"
}

cd "$REPO"
[[ -z $(git status --porcelain --untracked-files=all) ]] \
  || refuse 'acquisition requires a clean committed tree'
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$PATCH" "$STAGE_PATCH" "$WRITER" "$FIXTURE_STUBS" "$FIXTURE" "$EXCERPT" \
  "$GATE" "$TEST" "$R326_TEST" "$PREREG" "$SOURCE_DYNZDF" "$SOURCE_STAGE" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$SOURCE_ROOT/BLD/bin/nemo.exe" \
  "$SOURCE_RUN/nemo" "$SOURCE_RUN/ocean.output" "$SOURCE_RUN/namelist_cfg" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" \
  "$SOURCE_ROOT/BLD/arch_nemo.fcm" "$SOURCE_ROOT/BLD/ppsrc/nemo/dynzdf.f90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90"; do
  [[ -f "$path" ]] || refuse "missing input $path"
done
pin "$SOURCE_DYNZDF_SHA" "$SOURCE_DYNZDF" 'source dynzdf'
pin "$SOURCE_STAGE_SHA" "$SOURCE_STAGE" 'round-326 record stprk3_stg'
pin "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source build binary'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'source run binary'
pin "$DECK_MANIFEST_SHA" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin "$INPUT_MANIFEST_SHA" "$SOURCE_RUN/input_files.sha256" 'input manifest'
pin "$OCEAN_OUTPUT_SHA" "$SOURCE_RUN/ocean.output" 'source ocean.output'

command -v timeout >/dev/null || refuse 'no timeout binary'
mkdir -p "$(dirname "$EVIDENCE")"
scratch=$(mktemp -d "$(dirname "$EVIDENCE")/.r328-scratch.XXXXXX")
plants=$scratch/plants; mkdir "$plants" "$scratch/mod"

# Checklist (1): additions-only and no insertion following a continued line.
for patch_file in "$PATCH" "$STAGE_PATCH"; do
  check_no_removed_lines "$patch_file" || refuse "observer patch removes a source line: $patch_file"
  check_no_continuation_insert "$patch_file" || refuse "patch inserts after a continuation: $patch_file"
done
printf ' x = a + &\n+ CALL planted()\n' >"$plants/continuation.patch"
plant_must_refuse continuation check_no_continuation_insert "$plants/continuation.patch"
# Checklist (2): makenemo -r must name a registered reference config.
check_ref_cfg "$REFERENCE_CFG" || refuse "$REFERENCE_CFG not in ref_cfgs.txt"
plant_must_refuse ref-cfg check_ref_cfg ORCA2_OMIP_L4
# Checklist (3): the copied deck must satisfy NEMO's timestep/restart rules.
check_namelist "$SOURCE_RUN/namelist_cfg" || refuse 'source namelist breaks itend/stock/fsbc rules'
sed -E 's/^([[:space:]]*nn_itend[[:space:]]*=[[:space:]]*)[0-9]+/\1 9/' \
  "$SOURCE_RUN/namelist_cfg" >"$plants/namelist_cfg"
plant_must_refuse namelist check_namelist "$plants/namelist_cfg"
# LOG-CHECK: a committed excerpt is pinned to the real round-326 output.
[[ $(sed -nE 's/^# Complete source SHA-256: ([0-9a-f]{64})$/\1/p' "$EXCERPT") == "$OCEAN_OUTPUT_SHA" ]] \
  || refuse 'committed ocean.output excerpt has the wrong source SHA'
check_echo "$EXCERPT" || refuse 'real ocean.output excerpt lacks frozen selectors'
sed -E 's/(ln_zad_Aimp[[:space:]]*=[[:space:]]*)F/\1T/' "$EXCERPT" >"$plants/ocean.output"
plant_must_refuse echo check_echo "$plants/ocean.output"
sed -E 's/(nn_mxlice[[:space:]]*=[[:space:]]*)2/\1 0/' "$EXCERPT" >"$plants/ocean.output.mxlice"
plant_must_refuse echo-mxlice check_echo "$plants/ocean.output.mxlice"
sed -E 's/(jpl[[:space:]]*=[[:space:]]*)1/\1 5/' "$EXCERPT" >"$plants/ocean.output.jpl"
plant_must_refuse echo-jpl check_echo "$plants/ocean.output.jpl"
# Checklist (4)+(5): compiled exact writer layout, every expected frame, A4,
# and one-ULP A5 plants. The imported round-326 test proves ELF root-only normalisation.
"$PY" -m pytest -q -p no:cacheprovider "$TEST" \
  "$R326_TEST::test_objects_equal_ignores_only_the_embedded_config_root"
# Checklist (6): disk checks work before the evidence directory exists.
for mount in "$EVIDENCE" "$NEMO_ROOT"; do
  free_ok "$mount" || refuse "$mount has under 8 GB free"
done
free_ok "$EVIDENCE/not/created/yet" || refuse 'disk check fails on a missing parent'
plant_must_refuse disk free_ok "$NEMO_ROOT" 999999999999999
mem_ok "$MEM_MIN_GB" || refuse "under $MEM_MIN_GB GB available memory"
plant_must_refuse memory mem_ok 999999
# Checklist (7): both build and run targets must be absent.
if [[ "$MODE" != --admit-existing ]]; then
  check_absent "$TARGET_ROOT" && check_absent "$TARGET_RUN" \
    || refuse 'target config or run directory already exists'
fi
plant_must_refuse target-exists check_absent "$SOURCE_RUN"

# Prove all inherited deterministic frames and terminal restarts already pass
# between round 5 and its admitted round-326 record. Diagnostic grid NetCDFs
# are deliberately excluded because their TimeStamp attributes are wall-clock.
compare_deterministic "$BASELINE_RUN" "$SOURCE_RUN" \
  || refuse 'round-326 deterministic outputs differ from round 5'

# Syntax proof uses the record program's stprk3_stg, not pristine NEMO.
dry=$scratch/dry; mkdir "$dry"
cp "$SOURCE_DYNZDF" "$dry/dynzdf.F90"
cp "$SOURCE_STAGE" "$dry/stprk3_stg.F90"
cp "$WRITER" "$dry/dynzdf_round328_writer.F90"
patch -s --fuzz=0 -p0 -d "$dry" <"$PATCH"
patch -s --fuzz=0 -p0 -d "$dry" <"$STAGE_PATCH"
[[ $(grep -Fc 'CALL r328_' "$dry/dynzdf.F90") -eq 17 ]] || refuse 'dynzdf writer call census changed'
[[ $(grep -Fc 'CALL r328_finish' "$dry/stprk3_stg.F90") -eq 1 ]] || refuse 'stage writer call census changed'
cp "$SOURCE_ROOT"/BLD/inc/*.mod "$scratch/mod/"
for unit in dynzdf_round328_writer dynzdf stprk3_stg; do
  fpp "$dry/$unit.F90" "$scratch/$unit.f90"
  "$FC" -fsyntax-only -ffree-line-length-none -I "$scratch/mod" -J "$scratch/mod" "$scratch/$unit.f90"
  printf 'SYNTAX_PROOF_PASS %s.f90\n' "$unit"
done

# Same-program preflight: round 326 plus both observer insertions, including
# stprk3_stg; a non-observer source plant must refuse.
dry_bld=$scratch/dry_bld; mkdir "$dry_bld"
cp -as "$SOURCE_ROOT/BLD/ppsrc" "$dry_bld/ppsrc"
cp -as "$SOURCE_ROOT/BLD/obj" "$dry_bld/obj"
for name in arch_nemo.fcm cpp.fcm bld.cfg; do
  sed "s|$SOURCE_CFG|$TARGET_CFG|g" "$SOURCE_ROOT/BLD/$name" >"$dry_bld/$name"
done
rm -f "$dry_bld/ppsrc/nemo/dynzdf.f90" "$dry_bld/ppsrc/nemo/stprk3_stg.f90"
cp "$scratch/dynzdf.f90" "$dry_bld/ppsrc/nemo/dynzdf.f90"
cp "$scratch/stprk3_stg.f90" "$dry_bld/ppsrc/nemo/stprk3_stg.f90"
cp "$scratch/dynzdf_round328_writer.f90" "$dry_bld/ppsrc/nemo/dynzdf_round328_writer.f90"
: >"$dry_bld/obj/dynzdf_round328_writer.o"
same_program "$dry_bld"
rm -f "$dry_bld/ppsrc/nemo/icevar.f90"
{ sed '1s/$/ ! planted/' "$SOURCE_ROOT/BLD/ppsrc/nemo/icevar.f90"; } >"$dry_bld/ppsrc/nemo/icevar.f90"
plant_must_refuse same-program same_program "$dry_bld"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND328_DYNZDF_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant
  [[ $(cat "$TARGET_RUN/producer_commit.txt") == "$COMMIT" ]] || refuse 'producer commit changed'
  compare_deterministic "$SOURCE_RUN" "$TARGET_RUN" \
    || refuse 'write-only observer changed an inherited deterministic output'
  for plant in header field-order truncation stamp avm bottom top solve endpoint; do
    if "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$COMMIT" --plant "$plant" \
      >"$TARGET_RUN/round328_${plant}_plant.log" 2>&1; then
      refuse "$plant plant stayed green"
    fi
    grep 'REFUSE:' "$TARGET_RUN/round328_${plant}_plant.log" >/dev/null \
      || refuse "$plant plant did not reach its refusal"
  done
  "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$COMMIT" \
    --output "$TARGET_RUN/round328_dynzdf_admission.json"
  ( cd "$TARGET_RUN" && sha256sum oracle_dynzdf_*.bin round328_dynzdf_admission.json \
      round328_*_plant.log >round328_outputs.sha256 )
  printf 'ORCA2_ROUND328_DYNZDF_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$BINARY" && -x "$TARGET_RUN/nemo" ]] || refuse 'existing target lacks its binary'
  cmp -s "$BINARY" "$TARGET_RUN/nemo" || refuse 'staged binary differs'
  admit
  exit 0
fi

manifest=$scratch/manifest; mkdir "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$PATCH" "$STAGE_PATCH" "$WRITER" "$FIXTURE_STUBS" "$FIXTURE" \
  "$EXCERPT" "$GATE" "$TEST" "$PREREG" "$SOURCE_DYNZDF" "$SOURCE_STAGE" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" >"$manifest/toolchain.sha256"

# makenemo rewrites shared work_cfgs.txt; only the target's line may change.
cp "$NEMO_ROOT/cfgs/work_cfgs.txt" "$scratch/work_cfgs.before"
cd "$NEMO_ROOT"
created_target=1
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$SOURCE_DYNZDF" "$TARGET_ROOT/MY_SRC/dynzdf.F90"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/dynzdf_round328_writer.F90"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$PATCH"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$STAGE_PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || refuse 'build produced no executable'
grep 'NEMO_L4_R328ZDF' "$TARGET_ROOT/BLD/ppsrc/nemo/dynzdf_round328_writer.f90" >/dev/null \
  || refuse 'compiled writer is absent'
diff <(grep -v "^$TARGET_CFG " "$scratch/work_cfgs.before") \
     <(grep -v "^$TARGET_CFG " "$NEMO_ROOT/cfgs/work_cfgs.txt") >/dev/null \
  || refuse 'work_cfgs.txt changed beyond the target line'
same_program "$TARGET_ROOT/BLD" "$manifest/same_program.json"

mkdir -p "$EVIDENCE"
created_run=1
mkdir "$TARGET_RUN"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done \
  <"$SOURCE_RUN/deck_files.sha256"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done \
  <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
( cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null )

(
  cd "$TARGET_RUN"
  mem_ok "$MEM_MIN_GB" || refuse "under $MEM_MIN_GB GB before run"
  free -g >run.user.memory.log
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  timeout --kill-after=60 "$RUN_TIMEOUT_S" mpirun -np 2 --oversubscribe ./nemo >run.user.stdout.log 2>&1
  rc=$?
  set -e
  [[ $rc -eq 0 ]] || refuse "mpirun exited $rc"
  printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
grep 'STOP 0' "$TARGET_RUN/run.user.stdout.log" >/dev/null || refuse 'NEMO did not report STOP 0'
run_completed=1
check_echo "$TARGET_RUN/ocean.output" || refuse 'run selectors changed'
admit
