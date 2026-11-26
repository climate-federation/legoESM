#!/usr/bin/env bash
# Operator-only Tool B. Default is a read-only preflight; Codex never uses --run.
set -Eeuo pipefail

readonly MODE=${1:---preflight-only}
case "$MODE" in
  --preflight-only|--run|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac
refuse() { printf 'REFUSE: %s\n' "$*" >&2; exit 65; }
plant_must_refuse() {
  local label=$1; shift
  if "$@"; then refuse "preflight plant stayed green: $label"; fi
  printf 'PREFLIGHT_PLANT_FIRED %s\n' "$label"
}

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly here
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY
readonly BASELINE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R339BASE
readonly TARGET_CFG=ORCA2_STANDALONE_IIF_OMIP_L4_R339K0
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly BASELINE_ROOT=$NEMO_ROOT/cfgs/$BASELINE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2
readonly TOOL_A_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/ice_cover_producer/orca2_full_si3_year1_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/k0_smoke
readonly TARGET_RUN=$EVIDENCE/orca2_k0_10step_np2
readonly EXISTING_RUN=${R339_K0_EXISTING_RUN:-$TARGET_RUN}
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly PATCH=$here/k0_namelist.patch
readonly GATE=$here/../nemo_testcase_l4_orca2_round339_k0_smoke_gate.py
readonly COMMON=$here/../nemo_testcase_l4_orca2_round339_common.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round339_acquisition.py
readonly NAMELIST_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ref
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly MEM_MIN_GB=150
readonly RUN_TIMEOUT_S=900
readonly ADMISSION_IDENTITY_ONLY=${R339_K0_ADMISSION_IDENTITY_ONLY:-0}

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

[[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] \
  || refuse 'acquisition requires a clean committed tree'
readonly COMMIT=$(git -C "$REPO" rev-parse HEAD)
for path in "$PATCH" "$GATE" "$COMMON" "$TEST" "$NAMELIST_REF" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/namelist_cfg" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256"; do
  [[ -f "$path" ]] || refuse "missing pinned input $path"
done
if [[ "$ADMISSION_IDENTITY_ONLY" == 1 ]]; then
  [[ "$MODE" == --admit-existing && -n ${R339_K0_ADMISSION_TEST_ROOT:-} ]] \
    || refuse 'identity-only admission requires --admit-existing and R339_K0_ADMISSION_TEST_ROOT'
  test_root=$(realpath -e -- "$R339_K0_ADMISSION_TEST_ROOT")
  retained=$(realpath -e -- "$EXISTING_RUN")
  [[ "$retained" == "$test_root/refused/"* ]] \
    || refuse 'identity-only retained run is outside the test refused directory'
  "$PY" - "$retained/producer_manifest.json" "$retained/nemo" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
p=c.verify_manifest(Path(sys.argv[1])); c.require(p['binary_sha256']==c.sha256(Path(sys.argv[2])),'retained binary digest moved')
print('R339_K0_SYNTHETIC_RETAINED_ADMISSION_PASS',p['producer_commit'])
PY
  exit 0
fi
[[ "$ADMISSION_IDENTITY_ONLY" == 0 ]] || refuse 'R339_K0_ADMISSION_IDENTITY_ONLY must be 0 or 1'

mem_ok() { (( $(free -g | awk '$1 == "Mem:" {print $7}') >= $1 )); }
check_k0_deck() {
  local file=$1
  grep -E '^[[:space:]]*nn_ice[[:space:]]*=[[:space:]]*1([[:space:]!]|$)' "$file" >/dev/null \
    && grep -E '^[[:space:]]*ln_drgice_imp[[:space:]]*=[[:space:]]*\.false\.([[:space:]!]|$)' "$file" >/dev/null \
    && grep -E '^[[:space:]]*ln_mxl0[[:space:]]*=[[:space:]]*\.true\.([[:space:]!]|$)' "$file" >/dev/null \
    && grep -E '^[[:space:]]*nn_mxlice[[:space:]]*=[[:space:]]*0([[:space:]!]|$)' "$file" >/dev/null \
    && grep -E '^[[:space:]]*ln_icebergs[[:space:]]*=[[:space:]]*\.false\.([[:space:]!]|$)' "$file" >/dev/null \
    && grep -F "sn_ice      ='ice_cover_clim.nc'" "$file" >/dev/null
}
same_program() {
  "$PY" - "$1" "$BASELINE_ROOT/BLD" "$TARGET_CFG" "$BASELINE_CFG" <<'PY'
import json,sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
print(json.dumps(c.same_program_no_observer(Path(sys.argv[1]),Path(sys.argv[2]),new_cfg=sys.argv[3],reference_cfg=sys.argv[4]),sort_keys=True))
PY
}
check_k0_my_src() {
  local root=$1 source count=0
  while IFS= read -r -d '' source; do
    count=$((count+1))
    grep -EHn '(^|[^A-Za-z0-9_])(rCdU_ice|hm_i|divu_i|shear_i|delta_i|a_i|v_i|v_s|sv_i|oa_i|fr_i)([^A-Za-z0-9_]|$)' "$source" >/dev/null \
      && return 1
  done < <(find "$root" -maxdepth 1 \( -type f -o -type l \) -print0)
  [[ $count -eq 0 ]]
}
snapshot_build_sources() {
  local build_root=$1 destination=$2 source
  mkdir -p "$destination/MY_SRC" "$destination/BLD"
  cp -a "$build_root/BLD/ppsrc" "$destination/"
  for source in "$build_root"/MY_SRC/*; do
    [[ -e "$source" || -L "$source" ]] || continue
    cp -a "$source" "$destination/MY_SRC/$(basename "$source")"
  done
  for source in arch_nemo.fcm cpp.fcm bld.cfg full_key_list.txt; do
    [[ -f "$build_root/BLD/$source" ]] && cp -a "$build_root/BLD/$source" "$destination/BLD/$source"
  done
  cp -a "$build_root/cpp_$TARGET_CFG.fcm" "$destination/cpp_$TARGET_CFG.fcm"
}
write_producer_manifest() {
  local root=$1
  "$PY" - "$root" "$COMMIT" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
root=Path(sys.argv[1]); files=[p for p in root.rglob('*') if p.is_file() and p.name!='producer_manifest.json']
c.write_manifest(root/'producer_manifest.json',producer_commit=sys.argv[2],binary=root/'nemo',files=files,root=root,build_source_root=root/'build_sources')
PY
}
verify_tool_a() {
  "$PY" - "$TOOL_A_ROOT/producer_manifest.json" "$TOOL_A_ROOT/ice_cover_clim.nc" \
    "$TOOL_A_ROOT/round339_ice_cover_admission.json" <<'PY'
import json
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
p=c.verify_manifest(Path(sys.argv[1])); report=json.loads(Path(sys.argv[3]).read_text())
c.require(report['producer_commit']==p['producer_commit'],'Tool A admission producer moved')
c.require(c.sha256(Path(sys.argv[2]))==report['climatology']['sha256'],'Tool A climatology digest moved')
print('R339_TOOL_A_PIN_PASS',p['producer_commit'],report['climatology']['sha256'])
PY
}

scratch=$(mktemp -d /data/abyssal/dbalwada/nemo-testcases-l2/phase3/.r339-k0-preflight.XXXXXX)
cleanup() { local rc=$?; trap - EXIT; [[ ! -d "$scratch" ]] || rm -rf -- "$scratch"; exit "$rc"; }
trap cleanup EXIT
cp "$SOURCE_RUN/namelist_cfg" "$scratch/namelist_cfg"
patch --batch -s --fuzz=0 -p0 -d "$scratch" <"$PATCH"
check_k0_deck "$scratch/namelist_cfg" || refuse 'K0 patch does not resolve the selected deck'
"$PY" - "$NAMELIST_REF" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_k0_smoke_gate as g
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
c.validate_group_membership(Path(sys.argv[1]),g.GROUPS)
for variable,wrong in [('sn_ice','namsbc'),('ln_drgice_imp','namdrg_top'),('nn_mxlice','namzdf')]:
    planted=dict(g.GROUPS); planted[variable]=wrong
    try: c.validate_group_membership(Path(sys.argv[1]),planted)
    except c.GateError: print('PREFLIGHT_PLANT_FIRED group-'+variable)
    else: raise AssertionError('misplaced namelist variable stayed green: '+variable)
planted=dict(g.GROUPS); planted['nn_mxlicee']='namzdf_tke'
try: c.validate_group_membership(Path(sys.argv[1]),planted)
except c.GateError: print('PREFLIGHT_PLANT_FIRED misspelled-nn_mxlice')
else: raise AssertionError('misspelled variable stayed green')
PY
cp "$scratch/namelist_cfg" "$scratch/misplaced.nml"
sed -E -i 's/^([[:space:]]*nn_mxlice[[:space:]]*=)/   nn_mxlicee =/' "$scratch/misplaced.nml"
plant_must_refuse staged-name check_k0_deck "$scratch/misplaced.nml"
mkdir "$scratch/k0-my-src"
check_k0_my_src "$scratch/k0-my-src" || refuse 'empty K0 MY_SRC hygiene control failed'
cp "$NEMO_ROOT/src/OCE/stprk3.F90" "$scratch/k0-my-src/stprk3.F90"
plant_must_refuse k0-my-src-not-empty check_k0_my_src "$scratch/k0-my-src"
rm "$scratch/k0-my-src/stprk3.F90"
"$PY" -m pytest -q -p no:cacheprovider "$TEST"
synthetic_root=$scratch/synthetic-admission
synthetic_run=$synthetic_root/run
mkdir -p "$synthetic_run/build_sources/ppsrc/nemo" "$synthetic_run/build_sources/MY_SRC" \
  "$synthetic_root/refused"
printf 'synthetic binary\n' >"$synthetic_run/nemo"
printf 'program synthetic\n' >"$synthetic_run/build_sources/ppsrc/nemo/synthetic.f90"
"$PY" - "$synthetic_run" "$COMMIT" <<'PY'
import sys
from pathlib import Path
from scripts.validate.ocean_fidelity.orca2_l4 import nemo_testcase_l4_orca2_round339_common as c
root=Path(sys.argv[1]); files=[p for p in root.rglob('*') if p.is_file()]
c.write_manifest(root/'producer_manifest.json',producer_commit=sys.argv[2],binary=root/'nemo',files=files,root=root,build_source_root=root/'build_sources')
PY
synthetic_retained=$synthetic_root/refused/run_failed_synthetic
mv -- "$synthetic_run" "$synthetic_retained"
R339_K0_ADMISSION_IDENTITY_ONLY=1 R339_K0_ADMISSION_TEST_ROOT=$synthetic_root \
  R339_K0_EXISTING_RUN=$synthetic_retained "$0" --admit-existing \
  >"$scratch/synthetic-retained-admission.log"
grep -F 'R339_K0_SYNTHETIC_RETAINED_ADMISSION_PASS' "$scratch/synthetic-retained-admission.log" >/dev/null \
  || refuse 'Tool B --admit-existing did not accept its retained manifest'
printf 'PREFLIGHT_RETAINED_ADMISSION_PASS tool=B route=refused\n'
mem_ok "$MEM_MIN_GB" || refuse "under $MEM_MIN_GB GB memory available"
plant_must_refuse memory mem_ok 999999

admit_existing() {
  local root=$1
  [[ "$(realpath -m "$root")" == "$(realpath -m "$EVIDENCE")"/* ]] \
    || refuse 'existing run is outside round-339 K0 evidence'
  local producer
  producer=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["producer_commit"])' "$root/producer_manifest.json")
  for plant in group error-banner selector nan truncation manifest; do
    case "$plant" in
      group) expected="sn_ice belongs to ['namsbc'], expected membership in namsbc_iif" ;;
      error-banner) expected='ocean.output contains E R R O R' ;;
      selector) expected='resolved nn_ice is not 1' ;;
      nan) expected='non-finite restart variable' ;;
      truncation) expected='truncated restart plant' ;;
      manifest) expected='manifest producer commit moved' ;;
    esac
    plant_log="$scratch/admit-$plant.log"
    if "$PY" "$GATE" --root "$root" --reference-namelist "$NAMELIST_REF" \
      --expect-commit "$producer" --plant "$plant" >"$plant_log" 2>&1; then
      refuse "preflight plant stayed green: $plant"
    fi
    grep -F "$expected" "$plant_log" >/dev/null \
      || refuse "$plant plant missed its specific check: $expected"
    printf 'PREFLIGHT_PLANT_FIRED %s check=%s\n' "$plant" "$expected"
  done
  "$PY" "$GATE" --root "$root" --reference-namelist "$NAMELIST_REF" \
    --expect-commit "$producer" --output "$root/round339_k0_smoke_admission.json"
  printf 'ORCA2_ROUND339_K0_SMOKE_PASS scores_physics=false root=%s\n' "$root"
}

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND339_K0_PREFLIGHT_READY prerequisite=%s\n' "$TOOL_A_ROOT/ice_cover_clim.nc"
  exit 0
fi
if [[ "$MODE" == --admit-existing ]]; then
  admit_existing "$EXISTING_RUN"
  exit 0
fi

verify_tool_a
[[ -x "$BASELINE_ROOT/BLD/bin/nemo.exe" ]] \
  || refuse 'Tool A observer-free baseline build is missing'
[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] \
  || refuse 'K0 target build or run exists; retain/move it aside before retrying'
mkdir -p "$EVIDENCE"
run_created=0
retain_on_failure() {
  local rc=$? retained
  trap - EXIT
  set +e
  if [[ $rc -ne 0 && $run_created -eq 1 && -d "$TARGET_RUN" ]]; then
    retained=$EVIDENCE/refused/$(basename "$TARGET_RUN")_$(date -u +%Y%m%dT%H%M%SZ)_$$
    mkdir -p "$EVIDENCE/refused"
    mv -- "$TARGET_RUN" "$retained" \
      && printf 'R339_K0_FAILED_OUTPUTS_KEPT path=%s build=%s\n' "$retained" "$TARGET_ROOT" >&2
  fi
  [[ ! -d "$scratch" ]] || rm -rf -- "$scratch"
  exit "$rc"
}
trap retain_on_failure EXIT

cp "$NEMO_ROOT/cfgs/work_cfgs.txt" "$scratch/work_cfgs.before"
cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
check_k0_my_src "$TARGET_ROOT/MY_SRC" \
  || refuse 'K0 MY_SRC is not empty or touches a SI3 array'
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || refuse 'K0 build produced no executable'
same_program "$TARGET_ROOT/BLD" >"$scratch/same_program.json"
diff <(grep -v "^$TARGET_CFG " "$scratch/work_cfgs.before") \
     <(grep -v "^$TARGET_CFG " "$NEMO_ROOT/cfgs/work_cfgs.txt") >/dev/null \
  || refuse 'work_cfgs.txt changed beyond the K0 target line'

mkdir "$TARGET_RUN"; run_created=1
while read -r _digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r _digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
patch --batch -s --fuzz=0 -p0 -d "$TARGET_RUN" <"$PATCH"
check_k0_deck "$TARGET_RUN/namelist_cfg" || refuse 'staged K0 namelist moved'
cp "$TOOL_A_ROOT/ice_cover_clim.nc" "$TARGET_RUN/ice_cover_clim.nc"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$scratch/same_program.json" "$TARGET_RUN/same_program.json"
cp "$TARGET_RUN/namelist_cfg" "$TARGET_RUN/staged.namelist_cfg"
printf '%s\n' key_si3 key_qco key_vco_1d3d key_RK3 >"$TARGET_RUN/compiled_keys.txt"
printf '%s\n' "$COMMIT" >"$TARGET_RUN/producer_commit.txt"
snapshot_build_sources "$TARGET_ROOT" "$TARGET_RUN/build_sources"
write_producer_manifest "$TARGET_RUN"

(
  cd "$TARGET_RUN"
  mem_ok "$MEM_MIN_GB" || refuse "under $MEM_MIN_GB GB memory before K0 mpirun"
  free -g >run.user.memory.log
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  timeout --kill-after=60 "$RUN_TIMEOUT_S" mpirun -np 2 --oversubscribe ./nemo >run.user.stdout.log 2>&1
  printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
grep 'STOP 0' "$TARGET_RUN/run.user.stdout.log" >/dev/null || refuse 'K0 did not report STOP 0'
! grep -F 'E R R O R' "$TARGET_RUN/ocean.output" "$TARGET_RUN/run.user.stdout.log" >/dev/null \
  || refuse 'K0 reported E R R O R'
[[ -f "$TARGET_RUN/output.namelist.dyn" ]] || refuse 'K0 resolved namelist is missing'
admit_existing "$TARGET_RUN"
