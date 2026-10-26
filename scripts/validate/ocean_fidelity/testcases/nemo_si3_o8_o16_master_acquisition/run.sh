#!/usr/bin/env bash
# SI3 O8--O16 MASTER ACQUISITION -- OPERATOR EXECUTION ONLY.
# Default is read-only preflight. Only explicit --run invokes makenemo/NEMO.
set -Eeuo pipefail

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=ORCA2_SAS_ICE
readonly BASE_BLD=${BASE_BLD:-$NEMO_ROOT/cfgs/ORCA2_SAS_ICE_O7_REF/BLD}
readonly REF_CFG=${REF_CFG:-ORCA2_SAS_ICE_O8O16_REF}
readonly REC_CFG=${REC_CFG:-ORCA2_SAS_ICE_O8O16_REC}
readonly EVIDENCE_ROOT=${EVIDENCE_ROOT:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18}
EVIDENCE=${EVIDENCE:-$EVIDENCE_ROOT/oracle_o8_o16_master}
readonly REFUSED_DIR=${REFUSED_DIR:-$EVIDENCE_ROOT/refused}
readonly RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-14400}
readonly MIN_FREE_RAM_GB=${MIN_FREE_RAM_GB:-150}
readonly MIN_FREE_DISK_KB=${MIN_FREE_DISK_KB:-150000000}
readonly MAX_EVIDENCE_DISK_KB=${MAX_EVIDENCE_DISK_KB:-150000000}
readonly ORCA2_BUNDLE=${ORCA2_BUNDLE:-/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0.tar.gz}
readonly ORCA2_BUNDLE_SHA256=${ORCA2_BUNDLE_SHA256:-5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4}
readonly SAS_BUNDLE=${SAS_BUNDLE:-/data/abyssal/dbalwada/nemo-testcases-l4/inputs/SAS_v5.0.0_download/SAS_v5.0.0.tar.gz}
readonly SAS_BUNDLE_SHA256=${SAS_BUNDLE_SHA256:-ee00646832ea89f468d0f01bf537ed231d21f23bba77ab1a9aab26147ecec6cf}
readonly PY=${PYTHON_BIN:-/home/dbalwada/legoESM/.venv/bin/python3}

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
export PYTHONPATH="$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/ice:$REPO/packages/atmosphere:$REPO/packages/land:$REPO/packages/coupler:$REPO/packages/driver"
readonly GATE=$here/../nemo_si3_o8_o16_master_gate.py
readonly PASSIVITY=$here/../nemo_si3_o8_o16_master_passivity.py
readonly WRITER=$here/o8_o16_master_writer.F90
readonly STUB_NEMO=$here/stub_nemo.sh
readonly PROD=$NEMO_ROOT/cfgs/$SOURCE_CFG/EXPREF
readonly SHARED=$NEMO_ROOT/cfgs/SHARED
readonly REF_ROOT=$NEMO_ROOT/cfgs/$REF_CFG
readonly REC_ROOT=$NEMO_ROOT/cfgs/$REC_CFG
readonly UNITS=(icestp icethd icethd_do icethd_da icethd_sal icethd_pnd icethd_zdf_bl99)
readonly MASTER_BYTES=$($PY "$GATE" --size | $PY -c 'import json,sys; print(json.load(sys.stdin)["master_bytes_worst_case"])')
export O8O16_GATE=$GATE O8O16_PY=$PY

mode=${1:---preflight-only}; arg=${2:-}
case "$mode" in
  --preflight-only) PREFLIGHT_JSON=$arg ;;
  --run|--admit-existing) [[ -z "$arg" ]] || EVIDENCE=$arg ;;
  --move-aside-builds) [[ -z "$arg" ]] || { printf 'REFUSE: --move-aside-builds takes no path\n' >&2; exit 64; } ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing|--move-aside-builds] [path]\n' "$0" >&2; exit 64 ;;
esac

ACTIVE_RUN=""; ACTIVE_SCRATCH=""; CREATED_REF=0; CREATED_REC=0; RUN_STARTED_EPOCH=0
cleanup() {
  local rc=$?
  [[ -z "$ACTIVE_SCRATCH" || ! -d "$ACTIVE_SCRATCH" ]] || printf 'PREFLIGHT_SCRATCH_RETAINED: %s\n' "$ACTIVE_SCRATCH" >&2
  if [[ "$rc" -ne 0 ]]; then
    mkdir -p "$REFUSED_DIR"
    if [[ -n "$ACTIVE_RUN" && -d "$ACTIVE_RUN" ]]; then
      local dst="$REFUSED_DIR/$(basename "$ACTIVE_RUN")_$(date -u +%Y%m%dT%H%M%SZ)_rc${rc}_$$"
      mv "$ACTIVE_RUN" "$dst"; printf 'REFUSE_RETAINED: %s\n' "$dst" >&2
    fi
    [[ "$CREATED_REF" -eq 0 || ! -d "$REF_ROOT" ]] || printf 'REFUSE_BUILD_RETAINED: %s\n' "$REF_ROOT" >&2
    [[ "$CREATED_REC" -eq 0 || ! -d "$REC_ROOT" ]] || printf 'REFUSE_BUILD_RETAINED: %s\n' "$REC_ROOT" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'printf "REFUSE: command failed at line %s; evidence is retained\n" "$LINENO" >&2' ERR

refuse() { printf 'REFUSE: %s\n' "$*" >&2; return 1; }
ram_gate() { local n; n=$(free -g | awk '/^Mem:/ {print $7}'); [[ -n "$n" && "$n" -ge "$1" ]] || refuse "available RAM ${n:-unknown} GB below $1 GB"; }
disk_gate() { local p=$1 n; while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done; n=$(df -Pk "$p" | awk 'NR==2 {print $4}'); [[ -n "$n" && "$n" -ge "$2" ]] || refuse "free disk ${n:-unknown} KB below $2 KB"; }

render_deck() {
  local dir=$1
  mkdir -p "$dir"
  cp "$PROD/namelist_cfg" "$dir/namelist_cfg"
  cp "$PROD/namelist_ice_cfg" "$dir/namelist_ice_cfg"
  cp -L "$SHARED/namelist_ref" "$dir/namelist_ref"
  cp -L "$SHARED/namelist_ice_ref" "$dir/namelist_ice_ref"
  "$PY" - "$dir/namelist_cfg" <<'PY'
import pathlib,re,sys
p=pathlib.Path(sys.argv[1]); s=p.read_text()
s,n=re.subn(r"(^\s*nn_itend\s*=\s*)[^!,\n]+",r"\g<1>100 ",s,count=1,flags=re.I|re.M)
if n != 1: raise SystemExit("STATUS PLANT-FIRED: cannot set nn_itend=100")
m=re.search(r"(^\s*&namrun\b[\s\S]*?)(^\s*/\s*$)",s,re.I|re.M)
if not m: raise SystemExit("STATUS PLANT-FIRED: cannot find &namrun")
s=s[:m.start(2)]+"   nn_stock = 100\n   nn_write = -1\n"+s[m.start(2):]
p.write_text(s)
PY
  cmp -s "$PROD/namelist_ice_cfg" "$dir/namelist_ice_cfg" || refuse 'production ice namelist was modified'
  ! grep -Eiq 'ln_zdf_chkcvg\s*=|O8|O9|O16' "$dir/namelist_ice_cfg" || refuse 'ice namelist contains an acquisition overlay'
}

verify_namelists() {
  local dir=$1 plant=${2:-none}
  "$PY" - "$dir" "$plant" <<'PY'
import re,sys
from pathlib import Path
d=Path(sys.argv[1]); plant=sys.argv[2]
def groups(text):
    out={}; group=None
    for raw in text.splitlines():
        code=raw.split('!',1)[0].strip()
        m=re.match(r'&([A-Za-z0-9_]+)',code)
        if m: group=m.group(1).lower(); out.setdefault(group,set()); continue
        if code.startswith('/'): group=None; continue
        if group and '=' in code: out[group].add(re.sub(r'\(.*\)$','',code.split('=',1)[0].strip()).lower())
    return out
for stem in ('namelist','namelist_ice'):
    cfg=(d/f'{stem}_cfg').read_text(); ref=(d/f'{stem}_ref').read_text()
    if plant == 'misspelled' and stem == 'namelist': cfg=re.sub(r'(&namrun)',r'\1\n nn_writ=-1',cfg,count=1,flags=re.I)
    if plant == 'misplaced' and stem == 'namelist': cfg=re.sub(r'(&namdom)',r'\1\n nn_write=-1',cfg,count=1,flags=re.I)
    got,want=groups(cfg),groups(ref)
    owners={n:{g for g,names in want.items() if n in names} for names in want.values() for n in names}
    for g,names in got.items():
        if g not in want: raise SystemExit(f'STATUS PLANT-FIRED: unknown group {g}')
        for n in names:
            if n not in want[g]: raise SystemExit(f'STATUS PLANT-FIRED: {n} staged in &{g}; owners={sorted(owners.get(n,set()))}')
cfg_text=(d/'namelist_cfg').read_text(); ref_text=(d/'namelist_ref').read_text()
def last_int(name):
    cfg=re.findall(rf'^\s*{name}\s*=\s*(-?\d+)',cfg_text,re.I|re.M)
    ref=re.findall(rf'^\s*{name}\s*=\s*(-?\d+)',ref_text,re.I|re.M)
    return int((cfg or ref)[-1])
for name,want in {'nn_it000':1,'nn_itend':100,'nn_stock':100,'nn_write':-1,'nn_fsbc':1}.items():
    if last_int(name)!=want: raise SystemExit(f'STATUS PLANT-FIRED: {name} != {want}')
print('PASS_UNMODIFIED_100_STEP_DECK')
PY
}

patch_real_sources() {
  local out=$1 unit
  mkdir -p "$out"
  for unit in "${UNITS[@]}"; do
    cp "$BASE_BLD/ppsrc/nemo/$unit.f90" "$out/$unit.F90"
    patch --batch --fuzz=0 "$out/$unit.F90" "$here/${unit}_o8_o16.patch" >/dev/null
  done
}

placement_gate() {
  local dir=$1 plant=${2:-none}
  "$PY" - "$dir" "$plant" <<'PY'
import sys
from pathlib import Path
d=Path(sys.argv[1]); plant=sys.argv[2]
checks={
'icethd_do.F90':[('CALL o8o16_frazil(1','fraz_frac(:,:) = 0._wp'),('fraz_frac(:,:) = 0._wp','CALL o8o16_frazil(2')],
'icethd.F90':[('CALL o8o16_category(3','CALL ice_thd_zdf'),('CALL ice_thd_zdf','CALL o8o16_category(4'),("CALL ice_cons2D  (1, 'icethd'",'CALL o8o16_global(10)'),('CALL o8o16_global(10)','CALL ice_thd_pnd'),('CALL ice_thd_pnd','CALL o8o16_global(11)'),('CALL ice_itd_rem','CALL o8o16_global(12)'),('CALL ice_thd_do','CALL o8o16_global(13)'),('CALL ice_cor( kt , 2 )','CALL o8o16_global(14)')],
'icethd_zdf_bl99.F90':[('END DO  ! End of the do while iterative procedure','CALL o8o16_picard')],
'icethd_sal.F90':[('IF( nn_icesal /= 4 )','CALL o8o16_salinity_config')],
'icethd_pnd.F90':[('IF( ioptio /= 1 )','CALL o8o16_pond_config')],
'icestp.F90':[('CALL ice_thd','CALL o8o16_update(18)'),('CALL o8o16_update(18)','CALL ice_var_glo2eqv'),('CALL ice_var_agg','CALL o8o16_update(15)'),('CALL o8o16_update(15)','CALL ice_update_flx'),('CALL ice_update_flx','CALL o8o16_update(16)')],
}
for file,pairs in checks.items():
    text=(d/file).read_text()
    for i,(a,b) in enumerate(pairs):
        ai=text.find(a); bi=text.find(b,ai+1)
        if plant == 'misplace' and file=='icethd_do.F90' and i==0: ai=bi+1
        if ai<0 or bi<0 or ai>=bi: raise SystemExit(f'STATUS PLANT-FIRED: placement {file}: {a} !< {b}')
print('PASS_REAL_PPSRC_PLACEMENT')
PY
}

expect_refusal() { if "$@" >/dev/null 2>&1; then refuse "plant did not refuse: $*"; fi; }

production_flags() {
  awk -F'= ' '/^export FFLAGS = / {print $2; exit}' "$BASE_BLD/Makefile"
}

compile_real_bounds_fixture() {
  local scratch=$1 run=$scratch/bounds_run unit obj
  local flags_text; flags_text=$(production_flags)
  read -r -a flags <<< "$flags_text"
  mkdir -p "$scratch/bounds_mod" "$scratch/bounds_obj" "$run/o8_o16_master_records"
  mpif90 -c -cpp "${flags[@]}" -fcheck=bounds -I"$BASE_BLD/inc" -J"$scratch/bounds_mod" \
    "$WRITER" -o "$scratch/bounds_obj/o8_o16_master_writer.o"
  for unit in "${UNITS[@]}"; do
    mpif90 -c -cpp "${flags[@]}" -fcheck=bounds -I"$scratch/bounds_mod" -I"$BASE_BLD/inc" \
      -J"$scratch/bounds_mod" "$scratch/patched/$unit.F90" -o "$scratch/bounds_obj/$unit.o"
  done
  mpif90 -c -cpp "${flags[@]}" -fcheck=bounds -I"$scratch/bounds_mod" -I"$BASE_BLD/inc" \
    -J"$scratch/bounds_mod" "$here/real_bounds_driver.F90" -o "$scratch/bounds_obj/real_bounds_driver.o"
  local base_objects=()
  for obj in "$BASE_BLD"/obj/*.o; do
    case $(basename "$obj") in
      nemo.o|icestp.o|icethd.o|icethd_do.o|icethd_da.o|icethd_sal.o|icethd_pnd.o|icethd_zdf_bl99.o) ;;
      *) base_objects+=("$obj") ;;
    esac
  done
  local link_flags; link_flags=$(awk -F'= ' '/^export LDFLAGS = / {print $2; exit}' "$BASE_BLD/Makefile")
  # shellcheck disable=SC2086
  mpif90 -o "$scratch/real_bounds_fixture" "$scratch/bounds_obj/real_bounds_driver.o" \
    "$scratch/bounds_obj/o8_o16_master_writer.o" \
    "$scratch/bounds_obj/icestp.o" "$scratch/bounds_obj/icethd.o" \
    "$scratch/bounds_obj/icethd_do.o" "$scratch/bounds_obj/icethd_da.o" \
    "$scratch/bounds_obj/icethd_sal.o" "$scratch/bounds_obj/icethd_pnd.o" \
    "$scratch/bounds_obj/icethd_zdf_bl99.o" "${base_objects[@]}" $link_flags
  (cd "$run" && "$scratch/real_bounds_fixture")
  "$PY" - "$run/o8_o16_master_records" <<'PY'
import importlib.util,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); gate=Path(__import__('os').environ['O8O16_GATE'])
spec=importlib.util.spec_from_file_location('bounds_gate',gate); mod=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
r1=mod.read_master_record(root/mod.record_name(1,1)); r2=mod.read_master_record(root/mod.record_name(2,1))
assert np.signbit(r1['fields']['qlead'][0,0])
assert np.unique(r1['fields']['qlead']).size > 2
assert np.all(r2['fields']['ht_i_new']==0) and np.all(r2['fields']['fraz_frac']==0)
print('PASS_REAL_PATCHED_WRITER_LAYOUT')
PY
}

same_program_fixture() {
  local scratch=$1 new=$scratch/same_new expected=$scratch/expected_ppsrc unit name
  local flags_text; flags_text=$(production_flags)
  read -r -a flags <<< "$flags_text"
  mkdir -p "$new"
  cp -a "$BASE_BLD/ppsrc" "$BASE_BLD/obj" "$new/"
  cp "$BASE_BLD/arch_nemo.fcm" "$BASE_BLD/cpp.fcm" "$BASE_BLD/bld.cfg" "$new/"
  for name in arch_nemo.fcm cpp.fcm bld.cfg; do
    "$PY" - "$new/$name" "$REF_CFG" PREFLIGHT_O8O16_REC <<'PY'
import sys
from pathlib import Path
p=Path(sys.argv[1]); text=p.read_text().replace(sys.argv[2],sys.argv[3]); p.unlink(); p.write_text(text)
PY
  done
  patch_real_sources "$expected"
  mkdir -p "$scratch/same_mod"
  cp "$WRITER" "$new/ppsrc/nemo/o8_o16_master_writer.f90"
  mpif90 -c -cpp "${flags[@]}" -I"$BASE_BLD/inc" -J"$scratch/same_mod" \
    "$WRITER" -o "$new/obj/o8_o16_master_writer.o"
  for unit in "${UNITS[@]}"; do
    rm -f -- "$new/ppsrc/nemo/$unit.f90" "$new/obj/$unit.o"
    cp "$expected/$unit.F90" "$new/ppsrc/nemo/$unit.f90"
    mpif90 -c -cpp "${flags[@]}" -I"$scratch/same_mod" -I"$BASE_BLD/inc" \
      -J"$scratch/same_mod" "$new/ppsrc/nemo/$unit.f90" -o "$new/obj/$unit.o"
  done
  "$PY" "$GATE" --same-program "$new" --reference-bld "$BASE_BLD" --patch-dir "$here" \
    --expected-ppsrc-dir "$expected" --new-cfg PREFLIGHT_O8O16_REC --reference-cfg "$REF_CFG" \
    > "$scratch/same_program.json"
  expect_refusal "$PY" "$GATE" --same-program "$new" --reference-bld "$BASE_BLD" \
    --patch-dir "$here" --expected-ppsrc-dir "$expected" --new-cfg PREFLIGHT_O8O16_REC \
    --reference-cfg "$REF_CFG" --plant same-program
  printf 'PASS_OBSERVER_FREE_SAME_PROGRAM\n'
}

stub_driver_fixture() {
  local scratch=$1 root=$scratch/stub_nemo_root bundles=$scratch/stub_bundles
  local evidence=$scratch/stub_evidence refused=$scratch/stub_refused fail_evidence=$scratch/stub_failure
  mkdir -p "$root/cfgs/$SOURCE_CFG/EXPREF" "$root/cfgs/SHARED" "$bundles/orca" "$bundles/sas"
  cp "$PROD/namelist_cfg" "$root/cfgs/$SOURCE_CFG/EXPREF/namelist_cfg"
  cp "$PROD/namelist_ice_cfg" "$root/cfgs/$SOURCE_CFG/EXPREF/namelist_ice_cfg"
  cp -L "$SHARED/namelist_ref" "$root/cfgs/SHARED/namelist_ref"
  cp -L "$SHARED/namelist_ice_ref" "$root/cfgs/SHARED/namelist_ice_ref"
  printf 'stub input\n' > "$bundles/orca/dummy.nc"; printf 'stub input\n' > "$bundles/sas/dummy_sas.nc"
  tar -czf "$bundles/orca.tar.gz" -C "$bundles/orca" .
  tar -czf "$bundles/sas.tar.gz" -C "$bundles/sas" .
  local orca_sha sas_sha; orca_sha=$(sha256sum "$bundles/orca.tar.gz"); orca_sha=${orca_sha%% *}
  sas_sha=$(sha256sum "$bundles/sas.tar.gz"); sas_sha=${sas_sha%% *}
  local envs=(env O8O16_STUB_MODE=1 O8O16_SKIP_DRIVER_TEST=1 NEMO_ROOT="$root" BASE_BLD="$BASE_BLD" \
    REF_CFG=STUB_O8O16_REF REC_CFG=STUB_O8O16_REC REFUSED_DIR="$refused" MIN_FREE_RAM_GB=1 \
    MIN_FREE_DISK_KB=1 ORCA2_BUNDLE="$bundles/orca.tar.gz" ORCA2_BUNDLE_SHA256="$orca_sha" \
    SAS_BUNDLE="$bundles/sas.tar.gz" SAS_BUNDLE_SHA256="$sas_sha")
  bash -n "$0"; bash -n "$STUB_NEMO"
  cp "$STUB_NEMO" "$scratch/syntax_plant.sh"; printf '\nif [[ 1 == 1 ]]\n' >> "$scratch/syntax_plant.sh"
  expect_refusal bash -n "$scratch/syntax_plant.sh"
  "${envs[@]}" EVIDENCE="$evidence" "$0" --run
  "${envs[@]}" EVIDENCE="$evidence" "$0" --admit-existing
  expect_refusal "${envs[@]}" EVIDENCE="$evidence" "$0" --run
  "${envs[@]}" "$0" --move-aside-builds
  mv "$evidence" "$refused/success_evidence"
  expect_refusal "${envs[@]}" O8O16_STUB_FAIL=1 EVIDENCE="$fail_evidence" "$0" --run
  find "$refused" -maxdepth 1 -type d -name 'stub_failure_*_rc*' | grep -q . \
    || refuse 'stub failure evidence was not retained'
  [[ -d "$root/cfgs/STUB_O8O16_REF" && -d "$root/cfgs/STUB_O8O16_REC" ]] \
    || refuse 'stub failure builds were not retained'
  "${envs[@]}" "$0" --move-aside-builds
  printf 'PASS_100_STEP_STUB_DRIVER_RETENTION\n'
}

run_preflight() {
  local scratch rung plant
  scratch=$(mktemp -d); ACTIVE_SCRATCH=$scratch
  if [[ ${O8O16_SKIP_DRIVER_TEST:-0} == 1 ]]; then
    bash -n "$0"; bash -n "$STUB_NEMO"
    "$PY" "$GATE" --self-test > "$scratch/self.json"
    printf 'PREFLIGHT_OK: stub-recursion control path\n'
    rm -rf -- "$scratch"; ACTIVE_SCRATCH=""; return
  fi
  printf 'PREFLIGHT 1/10 schema, sizes, signed zero, checksum and truncation plants\n'
  "$PY" "$GATE" --schema > "$scratch/schema.json"
  "$PY" "$GATE" --size > "$scratch/size.json"
  "$PY" "$GATE" --self-test > "$scratch/self.json"
  for plant in header truncation swapped-field one-bit; do expect_refusal "$PY" "$GATE" --self-test --plant "$plant"; done
  printf 'PREFLIGHT 2/10 every rung projection and missing/foreign/category plants\n'
  for rung in O8 O9 O10 O11 O12 O13 O14 O15 O16; do
    "$PY" "$GATE" --projection-self-test "$rung" >/dev/null
    expect_refusal "$PY" "$GATE" --projection-self-test "$rung" --plant missing-step
    expect_refusal "$PY" "$GATE" --projection-self-test "$rung" --plant foreign-kind
  done
  expect_refusal "$PY" "$GATE" --projection-self-test O9 --plant wrong-category
  printf 'PREFLIGHT 3/10 exact unmodified production namelist and group plants\n'
  render_deck "$scratch/deck"; verify_namelists "$scratch/deck"
  expect_refusal verify_namelists "$scratch/deck" misspelled
  expect_refusal verify_namelists "$scratch/deck" misplaced
  printf 'PREFLIGHT 4/10 real ppsrc patch placement and misplacement plant\n'
  patch_real_sources "$scratch/patched"; placement_gate "$scratch/patched"
  expect_refusal placement_gate "$scratch/patched" misplace
  printf 'PREFLIGHT 5/10 production-flag writer/patched-source compile and FP census plant\n'
  "$PY" "$PASSIVITY" > "$scratch/passivity.json"
  printf 'PREFLIGHT 6/10 real writer and all patched units under production flags plus -fcheck=bounds\n'
  compile_real_bounds_fixture "$scratch"
  printf 'PREFLIGHT 7/10 observer-free same-program comparison and plant\n'
  same_program_fixture "$scratch"
  printf 'PREFLIGHT 8/10 source-derived SAS time level and A2D(0) extent evidence\n'
  grep -q '"sas_time_level": 2' "$scratch/self.json"
  grep -q 'ALLOCATE( ht_i_new (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))' "$BASE_BLD/ppsrc/nemo/ice.f90"
  grep -q "SIZE(a,1)==ni" "$WRITER"
  printf 'PREFLIGHT 9/10 memory/disk and global wall-clock gate logic\n'
  ram_gate 1; disk_gate "$EVIDENCE" 1024
  grep -q 'remaining_timeout' "$0"
  grep -q 'REFUSE_RETAINED' "$0"
  printf 'PREFLIGHT 10/10 generated driver: syntax/refusal/admission/retention/move-aside with 100-step stub NEMO\n'
  stub_driver_fixture "$scratch"
  if [[ -n "${PREFLIGHT_JSON:-}" ]]; then printf '{"status":"PREFLIGHT_OK","nemo_run":false,"plants":39}\n' > "$PREFLIGHT_JSON"; fi
  printf 'PREFLIGHT_OK: no real NEMO integration was run; worst-case master %s bytes (122.37 GiB)\n' "$MASTER_BYTES"
  rm -rf -- "$scratch"; ACTIVE_SCRATCH=""
}

move_builds_aside() {
  local dst="$REFUSED_DIR/builds_$(date -u +%Y%m%dT%H%M%SZ)_$$" moved=0
  mkdir -p "$dst"
  if [[ -d "$REF_ROOT" ]]; then mv "$REF_ROOT" "$dst/$REF_CFG"; moved=$((moved+1)); fi
  if [[ -d "$REC_ROOT" ]]; then mv "$REC_ROOT" "$dst/$REC_CFG"; moved=$((moved+1)); fi
  [[ "$moved" -gt 0 ]] || refuse 'no retained O8-O16 build exists to move aside'
  printf 'BUILDS_MOVED_ASIDE: %s\n' "$dst"
}

admit() {
  [[ -f "$EVIDENCE/manifest/run_manifest.json" ]] || refuse 'missing pre-run manifest'
  "$PY" "$GATE" --admit-existing "$EVIDENCE"
}

if [[ "$mode" == --move-aside-builds ]]; then move_builds_aside; exit 0; fi
if [[ "$mode" == --preflight-only ]]; then run_preflight; exit 0; fi
if [[ "$mode" == --admit-existing ]]; then run_preflight; admit; printf 'ACQUISITION_ADMITTED: %s\n' "$EVIDENCE"; exit 0; fi

# Operator-only --run begins here.
[[ ${O8O16_STUB_MODE:-0} == 1 || -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] \
  || refuse '--run requires a clean committed tree'
readonly COMMIT=$(git -C "$REPO" rev-parse HEAD)
RUN_STARTED_EPOCH=$(date +%s)
run_preflight
[[ ! -e "$REF_ROOT" && ! -e "$REC_ROOT" && ! -e "$EVIDENCE" ]] || refuse 'target config/evidence already exists; use --move-aside-builds after a refusal'
ram_gate "$MIN_FREE_RAM_GB"; disk_gate "$EVIDENCE" "$MIN_FREE_DISK_KB"
mkdir -p "$EVIDENCE/input" "$EVIDENCE/manifest/production_deck" "$EVIDENCE/manifest/binaries" "$EVIDENCE/manifest/per_rung"
ACTIVE_RUN=$EVIDENCE
for pair in "$ORCA2_BUNDLE:$ORCA2_BUNDLE_SHA256" "$SAS_BUNDLE:$SAS_BUNDLE_SHA256"; do
  file=${pair%%:*}; want=${pair##*:}; got=$(sha256sum "$file"); [[ ${got%% *} == "$want" ]] || refuse "bundle checksum $file"
done
tar -xzf "$ORCA2_BUNDLE" --strip-components=1 -C "$EVIDENCE/input"
tar -xzf "$SAS_BUNDLE" --strip-components=1 -C "$EVIDENCE/input"
cp "$PROD/namelist_cfg" "$EVIDENCE/manifest/production_deck/namelist_cfg"
cp "$PROD/namelist_ice_cfg" "$EVIDENCE/manifest/production_deck/namelist_ice_cfg"
cp -L "$SHARED/namelist_ref" "$EVIDENCE/manifest/production_deck/namelist_ref"
cp -L "$SHARED/namelist_ice_ref" "$EVIDENCE/manifest/production_deck/namelist_ice_ref"

build_target() {
  local cfg=$1 instrument=$2 root=$NEMO_ROOT/cfgs/$1 unit
  [[ "$cfg" == "$REF_CFG" ]] && CREATED_REF=1 || CREATED_REC=1
  if [[ ${O8O16_STUB_MODE:-0} == 1 ]]; then
    mkdir -p "$root/BLD/bin"
    cp "$STUB_NEMO" "$root/BLD/bin/nemo.exe"; chmod +x "$root/BLD/bin/nemo.exe"
    return
  fi
  (cd "$NEMO_ROOT" && ./makenemo -r "$SOURCE_CFG" -n "$cfg" -m conda-scalarmath del_key 'key_xios')
  if [[ "$instrument" -eq 1 ]]; then
    for unit in "${UNITS[@]}"; do
      cp "$BASE_BLD/ppsrc/nemo/$unit.f90" "$root/MY_SRC/$unit.F90"
      patch --batch --fuzz=0 "$root/MY_SRC/$unit.F90" "$here/${unit}_o8_o16.patch" >/dev/null
    done
    cp "$WRITER" "$root/MY_SRC/o8_o16_master_writer.F90"
    (cd "$NEMO_ROOT" && ./makenemo -n "$cfg" -m conda-scalarmath del_key 'key_xios')
  fi
  [[ -x "$root/BLD/bin/nemo.exe" ]] || refuse "$cfg lacks nemo.exe"
}

build_target "$REF_CFG" 0; build_target "$REC_CFG" 1
mkdir -p "$EVIDENCE/manifest/expected_ppsrc"
if [[ ${O8O16_STUB_MODE:-0} == 1 ]]; then
  printf '{"status":"STUB_ONLY_REAL_SAME_PROGRAM_PROVED_IN_OUTER_PREFLIGHT"}\n' \
    > "$EVIDENCE/manifest/same_program.json"
else
  patch_real_sources "$EVIDENCE/manifest/expected_ppsrc"
  "$PY" "$GATE" --same-program "$REC_ROOT/BLD" --reference-bld "$REF_ROOT/BLD" \
    --patch-dir "$here" --expected-ppsrc-dir "$EVIDENCE/manifest/expected_ppsrc" \
    --new-cfg "$REC_CFG" --reference-cfg "$REF_CFG" \
    > "$EVIDENCE/manifest/same_program.json"
fi
cp "$REF_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.ref"
cp "$REC_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.rec"

# Checklist 9: manifest and every source digest exist before either integration.
"$PY" - "$EVIDENCE" "$REPO" "$here" "$COMMIT" "$MASTER_BYTES" <<'PY'
import hashlib,json,sys
from pathlib import Path
ev,repo,here,commit=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4]
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources=[here/'run.sh',here/'stub_nemo.sh',here.parent/'nemo_si3_o8_o16_master_gate.py',here.parent/'nemo_si3_o8_o16_master_passivity.py']+sorted(here.glob('*.F90'))+sorted(here.glob('*.patch'))
deck=ev/'manifest/production_deck'
nemo=Path('/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo')
manifest={
 'producer_commit':commit,'binary_sha256':{'reference':sha(ev/'manifest/binaries/nemo.ref'),'record':sha(ev/'manifest/binaries/nemo.rec')},
 'source_sha256':{str(p.relative_to(repo)):sha(p) for p in sources},
 'nemo_source_sha256':{unit:sha(nemo/f'{unit}.f90') for unit in ('icestp','icethd','icethd_do','icethd_da','icethd_sal','icethd_pnd','icethd_zdf_bl99')},
 'deck_sha256':{p.name:sha(p) for p in sorted(deck.iterdir())},
 'forcing_disposition':'O7_EXACT_PRODUCTS_TO_O8_O16',
 'dynamics_disposition':'PRODUCTION_DECK_ON_EXCLUDED_FROM_PROJECTIONS',
 'record_schema_sha256':sha(here.parent/'nemo_si3_o8_o16_master_gate.py'),
 'estimated_master_bytes':int(sys.argv[5]),'max_evidence_disk_kib':150000000,'wall_clock_budget_seconds':14400,
 'retention':'raw master and manifests immutable until O8-O16 all admit; per-rung manifests reference relative paths without copying',
 'steps':100,'ranks':1,'ice_namelist':'UNMODIFIED_PRODUCTION','ln_zdf_chkcvg_overlay':False,
}
(ev/'manifest/run_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY

remaining_timeout() {
  local elapsed remaining
  elapsed=$(( $(date +%s) - RUN_STARTED_EPOCH )); remaining=$(( RUN_TIMEOUT_SEC - elapsed ))
  [[ "$remaining" -gt 0 ]] || refuse "global wall-clock budget ${RUN_TIMEOUT_SEC}s exhausted"
  printf '%s\n' "$remaining"
}

stage_run() {
  local cfg=$1 dir=$2 record=$3 root=$NEMO_ROOT/cfgs/$1 f
  mkdir -p "$dir"; [[ "$record" -eq 0 ]] || mkdir -p "$dir/o8_o16_master_records"
  render_deck "$dir"; verify_namelists "$dir"
  for f in "$EVIDENCE/input"/*.nc; do ln -s "$f" "$dir/$(basename "$f")"; done
  cp "$root/BLD/bin/nemo.exe" "$dir/nemo"
  local remaining; remaining=$(remaining_timeout)
  if [[ ${O8O16_STUB_MODE:-0} == 1 ]]; then
    (cd "$dir" && printf 'RUN_STARTED\n' > run.user.time.log && timeout "$remaining" ./nemo > run.user.stdout.log 2>&1 && printf 'STUB_RC=0\nRUN DONE\n' >> run.user.time.log)
  else
    (cd "$dir" && printf 'RUN_STARTED\n' > run.user.time.log && timeout "$remaining" mpirun -np 1 --oversubscribe ./nemo > run.user.stdout.log 2>&1 && printf 'MPIRUN_RC=0\nRUN DONE\n' >> run.user.time.log)
  fi
  ! grep -Eiq 'E[[:space:]]+R[[:space:]]+R[[:space:]]+O[[:space:]]+R|ABORT' "$dir/ocean.output" "$dir/run.user.stdout.log" || refuse "fatal token in $dir"
  touch "$dir/RUN_COMPLETE"
}

stage_run "$REF_CFG" "$EVIDENCE/reference_100step" 0
stage_run "$REC_CFG" "$EVIDENCE/record_100step" 1
(cd "$EVIDENCE" && find record_100step/o8_o16_master_records -type f -print0 | sort -z | xargs -0 sha256sum > manifest/records.sha256)
for rung in O8 O9 O10 O11 O12 O13 O14 O15 O16; do
  "$PY" "$GATE" --read-projection "$rung" --records "$EVIDENCE/record_100step/o8_o16_master_records" > "$EVIDENCE/manifest/per_rung/$rung.json"
done
touch "$EVIDENCE/RUN_COMPLETE"
"$PY" - "$EVIDENCE" "$RUN_STARTED_EPOCH" "$MASTER_BYTES" "$MAX_EVIDENCE_DISK_KB" "$RUN_TIMEOUT_SEC" <<'PY'
import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]); started=int(sys.argv[2]); schema=int(sys.argv[3]); max_disk=int(sys.argv[4]); max_wall=int(sys.argv[5])
record_bytes=sum(p.stat().st_size for p in (root/'record_100step/o8_o16_master_records').glob('*.bin'))
disk_kib=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())//1024+1
wall=time.time()-started
if not 0 < record_bytes <= schema: raise SystemExit('REFUSE: measured records exceed schema budget')
if not 0 < disk_kib <= max_disk: raise SystemExit('REFUSE: measured evidence exceeds disk budget')
if not 0 <= wall <= max_wall: raise SystemExit('REFUSE: acquisition exceeds wall-clock budget')
usage={'record_bytes':record_bytes,'evidence_disk_kib':disk_kib,'wall_clock_seconds':wall,
       'schema_record_bytes_gate':schema,'evidence_disk_kib_gate':max_disk,'wall_clock_seconds_gate':max_wall}
(root/'manifest/resource_usage.json').write_text(json.dumps(usage,indent=2,sort_keys=True)+'\n')
PY
admit
printf 'ACQUISITION_DONE: %s\n' "$EVIDENCE"
