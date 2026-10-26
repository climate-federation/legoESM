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
# 119 GB record plus two run directories and retained inputs/build margin.
readonly MIN_FREE_DISK_KB=${MIN_FREE_DISK_KB:-150000000}
readonly ORCA2_BUNDLE=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0.tar.gz
readonly ORCA2_BUNDLE_SHA256=5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4
readonly SAS_BUNDLE=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/SAS_v5.0.0_download/SAS_v5.0.0.tar.gz
readonly SAS_BUNDLE_SHA256=ee00646832ea89f468d0f01bf537ed231d21f23bba77ab1a9aab26147ecec6cf
readonly PY=${PYTHON_BIN:-/home/dbalwada/legoESM/.venv/bin/python3}

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
export PYTHONPATH="$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/ice:$REPO/packages/atmosphere:$REPO/packages/land:$REPO/packages/coupler:$REPO/packages/driver"
readonly GATE=$here/../nemo_si3_o8_o16_master_gate.py
readonly PASSIVITY=$here/../nemo_si3_o8_o16_master_passivity.py
readonly WRITER=$here/o8_o16_master_writer.F90
readonly PROD=$NEMO_ROOT/cfgs/$SOURCE_CFG/EXPREF
readonly SHARED=$NEMO_ROOT/cfgs/SHARED
readonly REF_ROOT=$NEMO_ROOT/cfgs/$REF_CFG
readonly REC_ROOT=$NEMO_ROOT/cfgs/$REC_CFG
readonly UNITS=(icestp icethd icethd_do icethd_da icethd_zdf_bl99)

mode=${1:---preflight-only}; arg=${2:-}
case "$mode" in
  --preflight-only) PREFLIGHT_JSON=$arg ;;
  --run|--admit-existing) [[ -z "$arg" ]] || EVIDENCE=$arg ;;
  --move-aside-builds) [[ -z "$arg" ]] || { printf 'REFUSE: --move-aside-builds takes no path\n' >&2; exit 64; } ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing|--move-aside-builds] [path]\n' "$0" >&2; exit 64 ;;
esac

ACTIVE_RUN=""; ACTIVE_SCRATCH=""; CREATED_REF=0; CREATED_REC=0
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
'icestp.F90':[('CALL o8o16_update(15)','CALL ice_update_flx'),('CALL ice_update_flx','CALL o8o16_update(16)')],
}
for file,pairs in checks.items():
    text=(d/file).read_text()
    for i,(a,b) in enumerate(pairs):
        ai=text.find(a); bi=text.find(b)
        if plant == 'misplace' and file=='icethd_do.F90' and i==0: ai=bi+1
        if ai<0 or bi<0 or ai>=bi: raise SystemExit(f'STATUS PLANT-FIRED: placement {file}: {a} !< {b}')
print('PASS_REAL_PPSRC_PLACEMENT')
PY
}

expect_refusal() { if "$@" >/dev/null 2>&1; then refuse "plant did not refuse: $*"; fi; }

run_preflight() {
  local scratch rung plant
  scratch=$(mktemp -d); ACTIVE_SCRATCH=$scratch
  printf 'PREFLIGHT 1/9 schema, sizes, signed zero, checksum and truncation plants\n'
  "$PY" "$GATE" --schema > "$scratch/schema.json"
  "$PY" "$GATE" --size > "$scratch/size.json"
  "$PY" "$GATE" --self-test > "$scratch/self.json"
  for plant in header truncation swapped-field one-bit; do expect_refusal "$PY" "$GATE" --self-test --plant "$plant"; done
  printf 'PREFLIGHT 2/9 every rung projection and missing/foreign/category plants\n'
  for rung in O8 O9 O10 O11 O12 O13 O14 O15 O16; do
    "$PY" "$GATE" --projection-self-test "$rung" >/dev/null
    expect_refusal "$PY" "$GATE" --projection-self-test "$rung" --plant missing-step
    expect_refusal "$PY" "$GATE" --projection-self-test "$rung" --plant foreign-kind
  done
  expect_refusal "$PY" "$GATE" --projection-self-test O9 --plant wrong-category
  printf 'PREFLIGHT 3/9 exact unmodified production namelist and group plants\n'
  render_deck "$scratch/deck"; verify_namelists "$scratch/deck"
  expect_refusal verify_namelists "$scratch/deck" misspelled
  expect_refusal verify_namelists "$scratch/deck" misplaced
  printf 'PREFLIGHT 4/9 real ppsrc patch placement and misplacement plant\n'
  patch_real_sources "$scratch/patched"; placement_gate "$scratch/patched"
  expect_refusal placement_gate "$scratch/patched" misplace
  printf 'PREFLIGHT 5/9 production-flag writer/patched-source compile and FP census plant\n'
  "$PY" "$PASSIVITY" > "$scratch/passivity.json"
  printf 'PREFLIGHT 6/9 -fcheck=bounds execution of patched ice_thd_da with distinct/-0 fixture\n'
  gfortran -fdefault-real-8 -O0 -fcheck=bounds -ffree-line-length-none -J"$scratch" -I"$scratch" \
    "$here/o11_bounds_stubs.F90" "$scratch/patched/icethd_da.F90" "$here/o11_bounds_driver.F90" \
    -o "$scratch/o11_bounds_fixture"
  "$scratch/o11_bounds_fixture"
  printf 'PREFLIGHT 7/9 source-derived SAS time level and A2D(0) extent evidence\n'
  grep -q '"sas_time_level": 2' "$scratch/self.json"
  grep -q 'ALLOCATE( ht_i_new (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))' "$BASE_BLD/ppsrc/nemo/ice.f90"
  grep -q "SIZE(a,1)==ni" "$WRITER"
  printf 'PREFLIGHT 8/9 memory/disk gate logic, retention and explicit-run boundary\n'
  ram_gate 1; disk_gate "$EVIDENCE" 1024
  grep -q 'timeout "$RUN_TIMEOUT_SEC" mpirun -np 1' "$0"
  grep -q 'REFUSE_RETAINED' "$0"
  printf 'PREFLIGHT 9/9 no NEMO command executed\n'
  if [[ -n "${PREFLIGHT_JSON:-}" ]]; then printf '{"status":"PREFLIGHT_OK","nemo_run":false,"plants":36}\n' > "$PREFLIGHT_JSON"; fi
  printf 'PREFLIGHT_OK: no NEMO build or integration was run; worst-case master 114202493300 bytes (106.36 GiB)\n'
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
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] || refuse '--run requires a clean committed tree'
readonly COMMIT=$(git -C "$REPO" rev-parse HEAD)
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
"$PY" "$GATE" --same-program "$REC_ROOT/BLD" --reference-bld "$REF_ROOT/BLD" \
  --patch-dir "$here" --new-cfg "$REC_CFG" --reference-cfg "$REF_CFG" \
  > "$EVIDENCE/manifest/same_program.json"
cp "$REF_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.ref"
cp "$REC_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.rec"

# Checklist 9: manifest and every source digest exist before either integration.
"$PY" - "$EVIDENCE" "$REPO" "$here" "$COMMIT" <<'PY'
import hashlib,json,sys
from pathlib import Path
ev,repo,here,commit=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4]
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources=[here/'run.sh',here.parent/'nemo_si3_o8_o16_master_gate.py',here.parent/'nemo_si3_o8_o16_master_passivity.py']+sorted(here.glob('*.F90'))+sorted(here.glob('*.patch'))
deck=ev/'manifest/production_deck'
manifest={
 'producer_commit':commit,'binary_sha256':{'reference':sha(ev/'manifest/binaries/nemo.ref'),'record':sha(ev/'manifest/binaries/nemo.rec')},
 'source_sha256':{str(p.relative_to(repo)):sha(p) for p in sources},
 'nemo_source_sha256':{unit:sha(Path('/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo')/f'{unit}.f90') for unit in ('icestp','icethd','icethd_do','icethd_da','icethd_zdf_bl99')},
 'deck_sha256':{p.name:sha(p) for p in sorted(deck.iterdir())},
 'forcing_disposition':'O7_EXACT_PRODUCTS_TO_O8_O16',
 'dynamics_disposition':'PRODUCTION_DECK_ON_EXCLUDED_FROM_PROJECTIONS',
 'record_schema_sha256':sha(here.parent/'nemo_si3_o8_o16_master_gate.py'),
 'estimated_master_bytes':114202493300,'wall_clock_budget_seconds':14400,
 'retention':'raw master and manifests immutable until O8-O16 all admit; per-rung manifests reference relative paths without copying',
 'steps':100,'ranks':1,'ice_namelist':'UNMODIFIED_PRODUCTION','ln_zdf_chkcvg_overlay':False,
}
(ev/'manifest/run_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY

stage_run() {
  local cfg=$1 dir=$2 record=$3 root=$NEMO_ROOT/cfgs/$1 f
  mkdir -p "$dir"; [[ "$record" -eq 0 ]] || mkdir -p "$dir/o8_o16_master_records"
  render_deck "$dir"; verify_namelists "$dir"
  for f in "$EVIDENCE/input"/*.nc; do ln -s "$f" "$dir/$(basename "$f")"; done
  cp "$root/BLD/bin/nemo.exe" "$dir/nemo"
  (cd "$dir" && printf 'RUN_STARTED\n' > run.user.time.log && timeout "$RUN_TIMEOUT_SEC" mpirun -np 1 --oversubscribe ./nemo > run.user.stdout.log 2>&1 && printf 'MPIRUN_RC=0\nRUN DONE\n' >> run.user.time.log)
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
admit
printf 'ACQUISITION_DONE: %s\n' "$EVIDENCE"
