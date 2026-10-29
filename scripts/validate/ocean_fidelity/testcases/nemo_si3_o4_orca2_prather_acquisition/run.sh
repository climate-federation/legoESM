#!/usr/bin/env bash
# SI3 O4 PRATHER ACQUISITION -- OPERATOR EXECUTION ONLY.
# Default is read-only preflight. Only explicit --run invokes makenemo/NEMO.
set -Eeuo pipefail

export PATH=${O4_PATH_PREFIX:+$O4_PATH_PREFIX:}/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly NEMO_SOURCE_ROOT=${NEMO_SOURCE_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=ORCA2_SAS_ICE
readonly BASE_BLD=${BASE_BLD:-$NEMO_ROOT/cfgs/ORCA2_SAS_ICE_O7_REF/BLD}
readonly REF_CFG=${REF_CFG:-ORCA2_SAS_ICE_O4_REF}
readonly REC_CFG=${REC_CFG:-ORCA2_SAS_ICE_O4_REC}
readonly EVIDENCE_ROOT=${EVIDENCE_ROOT:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round25}
EVIDENCE=${EVIDENCE:-$EVIDENCE_ROOT/oracle_o4_prather}
readonly REFUSED_DIR=${REFUSED_DIR:-$EVIDENCE_ROOT/refused}
readonly RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-3600}
readonly MIN_FREE_RAM_GB=${MIN_FREE_RAM_GB:-150}
readonly MIN_FREE_DISK_KB=${MIN_FREE_DISK_KB:-30000000}
readonly MAX_EVIDENCE_KB=${MAX_EVIDENCE_KB:-25000000}
readonly ORCA2_BUNDLE=${ORCA2_BUNDLE:-/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0.tar.gz}
readonly ORCA2_BUNDLE_SHA256=${ORCA2_BUNDLE_SHA256:-5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4}
readonly SAS_BUNDLE=${SAS_BUNDLE:-/data/abyssal/dbalwada/nemo-testcases-l4/inputs/SAS_v5.0.0_download/SAS_v5.0.0.tar.gz}
readonly SAS_BUNDLE_SHA256=${SAS_BUNDLE_SHA256:-ee00646832ea89f468d0f01bf537ed231d21f23bba77ab1a9aab26147ecec6cf}
readonly REAL_SAS_FIXTURE=${O4_REAL_SAS_FIXTURE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round13/refused/oracle_o7_orca2_sas_ice_20261025T172428Z_rc2_3061284}
readonly PY=${PYTHON_BIN:-/home/dbalwada/legoESM/.venv/bin/python3}
readonly FC=${FC:-/home/dbalwada/miniconda3/envs/nemo-build/bin/mpif90}

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
export PYTHONPATH="$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/ice:$REPO/packages/atmosphere:$REPO/packages/land:$REPO/packages/coupler:$REPO/packages/driver"
readonly GATE=$here/../nemo_si3_o4_orca2_prather_gate.py
readonly READSET=$here/../nemo_si3_o4_orca2_prather_readset.py
readonly PASSIVITY=$here/../nemo_si3_o4_orca2_prather_passivity.py
readonly EXTENTS=$here/../../orca2_l4/nemo_acquisition_extents.py
readonly WRITER=$here/o4_prather_writer.F90
readonly PATCH=$here/icedyn_adv_pra_o4.patch
readonly STUB_WRITER=$here/write_stub_records.py
readonly STUB_NEMO=$here/stub_nemo.sh
readonly STUB_MPIRUN=$here/stub_mpirun.sh
readonly BOUNDS_STUBS=$here/o4_bounds_stubs.F90
readonly BOUNDS_DRIVER=$here/o4_bounds_driver.F90
readonly REAL_EXCERPT=$here/../nemo_si3_o7_orca2_sas_acquisition/ocean.output.excerpt
readonly PROD=$NEMO_ROOT/cfgs/$SOURCE_CFG/EXPREF
readonly SHARED=$NEMO_ROOT/cfgs/SHARED
readonly REF_ROOT=$NEMO_ROOT/cfgs/$REF_CFG
readonly REC_ROOT=$NEMO_ROOT/cfgs/$REC_CFG

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
  if [[ "$rc" -ne 0 ]]; then
    mkdir -p "$REFUSED_DIR"
    if [[ -n "$ACTIVE_RUN" && -d "$ACTIVE_RUN" ]]; then
      local dst="$REFUSED_DIR/$(basename "$ACTIVE_RUN")_$(date -u +%Y%m%dT%H%M%SZ)_rc${rc}_$$"
      mv "$ACTIVE_RUN" "$dst"; printf 'REFUSE_RETAINED: %s\n' "$dst" >&2
    fi
    [[ "$CREATED_REF" -eq 0 || ! -d "$REF_ROOT" ]] || printf 'REFUSE_BUILD_RETAINED: %s\n' "$REF_ROOT" >&2
    [[ "$CREATED_REC" -eq 0 || ! -d "$REC_ROOT" ]] || printf 'REFUSE_BUILD_RETAINED: %s\n' "$REC_ROOT" >&2
    [[ -z "$ACTIVE_SCRATCH" || ! -d "$ACTIVE_SCRATCH" ]] || printf 'PREFLIGHT_SCRATCH_RETAINED: %s\n' "$ACTIVE_SCRATCH" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'printf "REFUSE: command failed at line %s; evidence/builds are retained\n" "$LINENO" >&2' ERR

refuse() { printf 'REFUSE: %s\n' "$*" >&2; return 1; }
expect_refusal() { if "$@" >/dev/null 2>&1; then refuse "plant did not refuse: $*"; fi; }
ram_gate() { local n; n=$(free -g | awk '/^Mem:/ {print $7}'); [[ -n "$n" && "$n" -ge "$1" ]] || refuse "available RAM ${n:-unknown} GB below $1 GB"; }
disk_gate() { local p=$1 n; while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done; n=$(df -Pk "$p" | awk 'NR==2 {print $4}'); [[ -n "$n" && "$n" -ge "$2" ]] || refuse "free disk ${n:-unknown} KB below $2 KB"; }

render_deck() {
  local dir=$1
  mkdir -p "$dir"
  cp "$PROD/namelist_cfg" "$dir/namelist_cfg"
  cp "$PROD/namelist_ice_cfg" "$dir/namelist_ice_cfg"
  cp -L "$SHARED/namelist_ref" "$dir/namelist_ref"
  cp -L "$SHARED/namelist_ice_ref" "$dir/namelist_ice_ref"
  "$PY" - "$dir" <<'PY'
import re,sys
from pathlib import Path
d=Path(sys.argv[1])
def replace(text,name,value):
    text,n=re.subn(rf'(^\s*{name}\s*=\s*)[^!,\n]+',rf'\g<1>{value} ',text,count=1,flags=re.I|re.M)
    if n != 1: raise SystemExit(f'STATUS PLANT-FIRED: cannot set {name}')
    return text
def add_to_group(text,group,lines):
    m=re.search(rf'(^\s*&{group}\b[^\n]*\n)',text,re.I|re.M)
    if not m: raise SystemExit(f'STATUS PLANT-FIRED: missing &{group}')
    return text[:m.end()]+''.join(f'   {line}\n' for line in lines)+text[m.end():]
p=d/'namelist_cfg'; s=replace(p.read_text(),'nn_itend','2')
m=re.search(r'(^\s*&namrun\b[\s\S]*?)(^\s*/\s*$)',s,re.I|re.M)
if not m: raise SystemExit('STATUS PLANT-FIRED: missing &namrun')
s=s[:m.start(2)]+'   nn_stock = 2\n   nn_write = -1\n'+s[m.start(2):]; p.write_text(s)
p=d/'namelist_ice_cfg'; s=p.read_text()
s=add_to_group(s,'nampar',['ln_icedyn = .true.','ln_icethd = .false.'])
s=add_to_group(s,'namdyn',['ln_dynALL = .false.','ln_dynRHGADV = .false.',
                           'ln_dynADV1D = .false.','ln_dynADV2D = .true.',
                           'rn_uice = 0.5','rn_vice = 0.5'])
s=add_to_group(s,'namdyn_adv',['ln_adv_Pra = .true.','ln_adv_UMx = .false.'])
p.write_text(s)
PY
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
        code=raw.split('!',1)[0].strip(); m=re.match(r'&([A-Za-z0-9_]+)',code)
        if m: group=m.group(1).lower(); out.setdefault(group,{}); continue
        if code.startswith('/'): group=None; continue
        if group and '=' in code:
            name=re.sub(r'\(.*\)$','',code.split('=',1)[0].strip()).lower()
            out[group][name]=code.split('=',1)[1].strip().rstrip(',')
    return out
for stem in ('namelist','namelist_ice'):
    cfg=(d/f'{stem}_cfg').read_text(); ref=(d/f'{stem}_ref').read_text()
    if plant == 'misspelled' and stem == 'namelist': cfg=re.sub(r'(&namrun)',r'\1\n nn_writ=-1',cfg,count=1,flags=re.I)
    if plant == 'misplaced' and stem == 'namelist': cfg=re.sub(r'(&namdom)',r'\1\n nn_write=-1',cfg,count=1,flags=re.I)
    if plant == 'o4-misplaced' and stem == 'namelist_ice': cfg=re.sub(r'(&namdyn_adv[^\n]*\n)',r'\1 ln_dynADV2D=.true.\n',cfg,count=1,flags=re.I)
    got,want=groups(cfg),groups(ref)
    owners={n:{g for g,names in want.items() if n in names} for names in want.values() for n in names}
    for g,names in got.items():
        if g not in want: raise SystemExit(f'STATUS PLANT-FIRED: unknown group &{g}')
        for n in names:
            if n not in want[g]:
                raise SystemExit(f'STATUS PLANT-FIRED: {n} staged in &{g}; declared owners={sorted(owners.get(n,set()))}')
ice=groups((d/'namelist_ice_cfg').read_text())
def logical(group,name): return ice[group][name].lower() in {'.true.','true','t'}
want={('nampar','ln_icedyn'):True,('nampar','ln_icethd'):False,
      ('namdyn','ln_dynall'):False,('namdyn','ln_dynrhgadv'):False,
      ('namdyn','ln_dynadv1d'):False,('namdyn','ln_dynadv2d'):True,
      ('namdyn_adv','ln_adv_pra'):True,('namdyn_adv','ln_adv_umx'):False}
for (g,n),v in want.items():
    if logical(g,n) != v: raise SystemExit(f'STATUS PLANT-FIRED: O4 selector {g}/{n}')
for n in ('rn_uice','rn_vice'):
    if float(ice['namdyn'][n].replace('_wp','')) != 0.5: raise SystemExit(f'STATUS PLANT-FIRED: {n} != 0.5')
oce=groups((d/'namelist_cfg').read_text()); ref=groups((d/'namelist_ref').read_text())
def value(group,name): return oce.get(group,{}).get(name,ref[group][name])
for n,v in {'nn_it000':1,'nn_itend':2,'nn_stock':2,'nn_write':-1}.items():
    if int(value('namrun',n)) != v: raise SystemExit(f'STATUS PLANT-FIRED: {n} != {v}')
if int(value('namsbc','nn_fsbc')) != 1: raise SystemExit('STATUS PLANT-FIRED: nn_fsbc != 1')
print('PASS_O4_NAMELIST_GROUPS_AND_EXACT_SELECTORS')
PY
}

preprocess() {
  local source=$1 output=$2 keys=() key cpp_text fpp_text
  local -a cpp_cmd fpp_flags
  while read -r key; do keys+=("-D$key"); done < <(awk '/bld::tool::fppkeys/ {for(i=2;i<=NF;i++) print $i}' "$BASE_BLD/cpp.fcm")
  cpp_text=$(awk '/^%CPP[[:space:]]/ {$1=""; sub(/^ +/,""); print}' "$BASE_BLD/arch_nemo.fcm")
  fpp_text=$(awk '/^%FPPFLAGS[[:space:]]/ {$1=""; sub(/^ +/,""); print}' "$BASE_BLD/arch_nemo.fcm")
  read -r -a cpp_cmd <<< "$cpp_text"; read -r -a fpp_flags <<< "$fpp_text"
  [[ ${#cpp_cmd[@]} -gt 0 && ${#fpp_flags[@]} -gt 0 ]] || refuse 'cannot resolve record-build CPP invocation'
  "${cpp_cmd[@]}" "${fpp_flags[@]}" "${keys[@]}" -I "$BASE_BLD/../WORK" "$source" > "$output"
}

prepare_patched_ppsrc() {
  local scratch=$1
  mkdir -p "$scratch"
  cp "$NEMO_SOURCE_ROOT/src/ICE/icedyn_adv_pra.F90" "$scratch/icedyn_adv_pra.F90"
  patch --batch --fuzz=0 "$scratch/icedyn_adv_pra.F90" "$PATCH" >/dev/null
  preprocess "$scratch/icedyn_adv_pra.F90" "$scratch/icedyn_adv_pra.f90"
}

same_program_gate() {
  local observed=$1 reference=$2 plant=${3:-none}
  "$PY" - "$observed" "$reference" "$plant" <<'PY'
import re,sys
from pathlib import Path
observed=Path(sys.argv[1]).read_text().splitlines(); expected_observers=list(observed)
reference=Path(sys.argv[2]).read_text().splitlines(); plant=sys.argv[3]
if plant == 'unmarked': observed.insert(len(observed)//2,'pato_i(1,1)=pato_i(1,1)+1._wp')
if plant == 'marked-content': observed=[line.replace('CALL o4_cfl(','CALL o4_cfl_changed(',1) for line in observed]
def split(lines):
    out=[]; inside=False; blocks=0
    observer=[]
    for line in lines:
        if '! O4_OBSERVER_BEGIN' in line:
            if inside: raise SystemExit('STATUS PLANT-FIRED: nested observer marker')
            inside=True; blocks+=1; observer.append(line.strip()); continue
        if '! O4_OBSERVER_END' in line:
            if not inside: raise SystemExit('STATUS PLANT-FIRED: unmatched observer marker')
            inside=False; observer.append(line.strip()); continue
        if inside: observer.append(re.sub(r'\s+',' ',line.strip()))
        else: out.append(line.replace('ice_dyn_adv_pra_physics','ice_dyn_adv_pra'))
    if inside or blocks != 6: raise SystemExit(f'STATUS PLANT-FIRED: observer block count {blocks}')
    return [re.sub(r'\s+',' ',x.strip()) for x in out if x.strip()],observer
body,observer=split(observed); _expected_body,expected=split(expected_observers)
if observer != expected: raise SystemExit('STATUS PLANT-FIRED: marked observer insertion differs from expected real ppsrc')
if body != [re.sub(r'\s+',' ',x.strip()) for x in reference if x.strip()]:
    raise SystemExit('STATUS PLANT-FIRED: observer-stripped ppsrc differs from real reference ppsrc')
print('PASS_REAL_PPSRC_SAME_PROGRAM')
PY
}

placement_gate() {
  local ppsrc=$1 plant=${2:-none}
  "$PY" - "$ppsrc" "$plant" <<'PY'
import sys
from pathlib import Path
text=Path(sys.argv[1]).read_text(); plant=sys.argv[2]
pairs=[('entry','CALL o4_state( 2','CALL ice_dyn_adv_pra_physics('),
 ('current-cfl','zcfl = MAX( zcfl','zcfl_current = zcfl'),
 ('current-before-delay','zcfl_current = zcfl',"CALL mpp_max( 'icedyn_adv_pra'"),
 ('delayed-cfl',"IF    ( zcfl > 1.5 )",'o4_zcfl_current = zcfl_current'),
 ('delayed-after-timestep','z1_dt = 1._wp / zdt','o4_icycle = icycle'),
 ('delayed-before-loop','o4_icycle = icycle','zudy(:,:) = pu_ice(:,:)'),
 ('exit','CALL ice_dyn_adv_pra_physics(','CALL o4_cfl('),
 ('exit-state','CALL o4_cfl(','CALL o4_state( 5')]
for name,a,b in pairs:
    ai=text.find(a); bi=text.find(b,ai+1)
    if plant == name: ai=bi+1
    if ai < 0 or bi < 0 or ai >= bi: raise SystemExit(f'STATUS PLANT-FIRED: placement {name}: {a} !< {b}')
print('PASS_O4_ENTRY_INBODY_EXIT_PLACEMENT')
PY
}

production_flags() { awk -F'= ' '/^export FFLAGS = / {print $2; exit}' "$BASE_BLD/Makefile"; }

compile_bounds_fixture() {
  local scratch=$1 flags_text
  flags_text=$(production_flags); read -r -a flags <<< "$flags_text"
  mkdir -p "$scratch/mod" "$scratch/records/o4_prather_records"
  "$FC" -c -cpp "${flags[@]}" -fcheck=bounds -J"$scratch/mod" "$BOUNDS_STUBS" -o "$scratch/stubs.o"
  "$FC" -c -cpp "${flags[@]}" -fcheck=bounds -I"$scratch/mod" -J"$scratch/mod" "$WRITER" -o "$scratch/writer.o"
  "$FC" -c "${flags[@]}" -fcheck=bounds -I"$scratch/mod" -J"$scratch/mod" "$scratch/../patched/icedyn_adv_pra.f90" -o "$scratch/icedyn_adv_pra.o"
  "$FC" -c -cpp "${flags[@]}" -fcheck=bounds -I"$scratch/mod" -J"$scratch/mod" "$BOUNDS_DRIVER" -o "$scratch/driver.o"
  "$FC" -fcheck=bounds "$scratch/stubs.o" "$scratch/writer.o" "$scratch/icedyn_adv_pra.o" "$scratch/driver.o" -o "$scratch/o4_bounds"
  (cd "$scratch/records" && ../o4_bounds)
  [[ $(find "$scratch/records/o4_prather_records" -type f | wc -l) -eq 12 ]] || refuse 'patched O4 bounds fixture did not emit twelve observer frames'
  "$PY" - "$scratch/records/o4_prather_records" <<'PY'
import sys
from pathlib import Path
import numpy as np
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_gate import read_record
root=Path(sys.argv[1])
a=read_record(root/'o4_kind02_step0001_rank0000.bin')
b=read_record(root/'o4_kind02_step0002_rank0001.bin')
assert a['meta']['rank'] == 0 and b['meta']['rank'] == 1
assert np.all(a['fields']['ato_i'] == 1.0) and np.all(b['fields']['ato_i'] == 2.0)
assert a['sha256'] != b['sha256']
assert np.all(a['fields']['wfx_res'] == 0.0) and np.all(np.signbit(a['fields']['wfx_res']))
print('PASS_O4_BOUNDS_MULTIRANK_DISTINCT_SIGNED_ZERO')
PY
  "$PY" "$GATE" "$scratch/records/o4_prather_records" --plant step >/dev/null 2>&1 && refuse 'bounds fixture reader plant survived'
  printf 'PASS_PATCHED_O4_SOURCE_FCHECK_BOUNDS\n'
}

stub_score_and_plants() {
  local root=$1 plant
  mkdir -p "$root/o4_prather_records" "$root/nontrivial_records"
  "$PY" "$STUB_WRITER" "$root/o4_prather_records" --source-real "$REAL_EXCERPT"
  "$PY" "$GATE" "$root/o4_prather_records" > "$root/score.json"
  "$PY" "$STUB_WRITER" "$root/nontrivial_records" --source-real "$REAL_EXCERPT" --nontrivial-score-fixture
  "$PY" - "$root/o4_prather_records" "$root/nontrivial_records" <<'PY'
import sys
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_gate import score_deck
for records,plants in ((sys.argv[1],('header','truncation','step','one-bit')),
                       (sys.argv[2],('invert-sweep','wrong-fold-sign','omit-subcycle-lbc'))):
    for plant in plants:
        try: score_deck(records,plant)
        except ValueError as exc:
            if 'STATUS PLANT-FIRED' in str(exc): continue
            raise
        raise SystemExit(f'REFUSE: scorer plant survived: {plant}')
print('PASS_O4_READER_AND_SCORE_PLANTS')
PY
}

move_builds_aside() {
  local dst="$REFUSED_DIR/builds_$(date -u +%Y%m%dT%H%M%SZ)_$$" moved=0
  mkdir -p "$dst"
  if [[ -d "$REF_ROOT" ]]; then mv "$REF_ROOT" "$dst/$REF_CFG"; moved=$((moved+1)); fi
  if [[ -d "$REC_ROOT" ]]; then mv "$REC_ROOT" "$dst/$REC_CFG"; moved=$((moved+1)); fi
  [[ "$moved" -gt 0 ]] || refuse 'no retained O4 build exists to move aside'
  printf 'BUILDS_MOVED_ASIDE: %s\n' "$dst"
}

write_manifest() {
  local commit=$1 fixture=$2
  "$PY" - "$EVIDENCE" "$REPO" "$here" "$commit" "$fixture" <<'PY'
import hashlib,json,sys
from pathlib import Path
ev,repo,here,commit,fixture=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],bool(int(sys.argv[5]))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=[here/'run.sh',here/'stub_nemo.sh',here/'stub_mpirun.sh',here/'write_stub_records.py',here/'o4_prather_writer.F90',here/'icedyn_adv_pra_o4.patch',here/'o4_bounds_stubs.F90',here/'o4_bounds_driver.F90',here.parent/'nemo_si3_o4_orca2_prather_gate.py',here.parent/'nemo_si3_o4_orca2_prather_passivity.py',here.parent/'nemo_si3_o4_orca2_prather_readset.py',here.parents[1]/'orca2_l4/nemo_acquisition_extents.py']
deck=ev/'manifest/production_deck'; inputs=ev/'input'
manifest={'schema':'si3-o4-prather-v2','producer_commit':commit,'fixture':fixture,
 'binary_paths':{'reference':'manifest/binaries/nemo.ref','recording':'manifest/binaries/nemo.rec'},
 'binary_sha256':{'reference':sha(ev/'manifest/binaries/nemo.ref'),'recording':sha(ev/'manifest/binaries/nemo.rec')},
 'source_sha256':{str(p.relative_to(repo)):sha(p) for p in sources},
 'deck_sha256':{str(p.relative_to(ev)):sha(p) for p in sorted(deck.iterdir())},
 'input_sha256':{str(p.relative_to(ev)):sha(p) for p in sorted(inputs.iterdir()) if p.is_file()},
 'planned_records':[f'o4_kind{k:02d}_step{s:04d}_rank0000.bin' for s in (1,2) for k in range(1,7)],
 'steps':[1,2],'ranks':1,'nn_stock':2,
 'dynamics':{'ln_dynALL':False,'ln_dynRHGADV':False,'ln_dynADV1D':False,'ln_dynADV2D':True,'ln_adv_Pra':True,'ln_adv_UMx':False,'rn_uice':0.5,'rn_vice':0.5},
 'retention':'failed evidence and both builds retained; rerun uses --move-aside-builds'}
(ev/'manifest/run_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY
}

admit() {
  local commit=$1 plant
  "$PY" "$GATE" --admit-evidence "$EVIDENCE" --expect-commit "$commit" > "$EVIDENCE/o4_admission.json"
  for plant in producer binary checksum missing-complete error-log restart missing-frame wet-1ulp; do expect_refusal "$PY" "$GATE" --admit-evidence "$EVIDENCE" --expect-commit "$commit" --plant "$plant"; done
}

stub_driver_fixture() {
  # O4_DRIVER_TRUNCATION_POINT: cutting run.sh here leaves this function open.
  local scratch=$1 root=$scratch/stub_nemo_root bundles=$scratch/stub_bundles
  local evidence=$scratch/stub_evidence refused=$scratch/stub_refused failed=$scratch/stub_failure
  mkdir -p "$root/cfgs/$SOURCE_CFG/EXPREF" "$root/cfgs/SHARED" "$bundles/orca" "$bundles/sas" "$scratch/stub_bin"
  ln -s "$STUB_MPIRUN" "$scratch/stub_bin/mpirun"
  cp "$PROD/namelist_cfg" "$root/cfgs/$SOURCE_CFG/EXPREF/namelist_cfg"; cp "$PROD/namelist_ice_cfg" "$root/cfgs/$SOURCE_CFG/EXPREF/namelist_ice_cfg"
  cp -L "$SHARED/namelist_ref" "$root/cfgs/SHARED/namelist_ref"; cp -L "$SHARED/namelist_ice_ref" "$root/cfgs/SHARED/namelist_ice_ref"
  cp "$NEMO_ROOT/cfgs/ref_cfgs.txt" "$root/cfgs/ref_cfgs.txt"
  printf 'stub ocean input\n' > "$bundles/orca/dummy.nc"; printf 'stub SAS input\n' > "$bundles/sas/dummy_sas.nc"
  tar -czf "$bundles/orca.tar.gz" -C "$bundles/orca" .; tar -czf "$bundles/sas.tar.gz" -C "$bundles/sas" .
  local orca_sha sas_sha; orca_sha=$(sha256sum "$bundles/orca.tar.gz"); orca_sha=${orca_sha%% *}; sas_sha=$(sha256sum "$bundles/sas.tar.gz"); sas_sha=${sas_sha%% *}
  local envs=(env O4_PATH_PREFIX="$scratch/stub_bin" O4_DRIVER_FIXTURE_ACTIVE=1 NEMO_EXE_OVERRIDE="$STUB_NEMO" NEMO_ROOT="$root" NEMO_SOURCE_ROOT="$NEMO_SOURCE_ROOT" BASE_BLD="$BASE_BLD" REF_CFG=STUB_O4_REF REC_CFG=STUB_O4_REC REFUSED_DIR="$refused" MIN_FREE_RAM_GB=1 MIN_FREE_DISK_KB=1 MAX_EVIDENCE_KB=100000 ORCA2_BUNDLE="$bundles/orca.tar.gz" ORCA2_BUNDLE_SHA256="$orca_sha" SAS_BUNDLE="$bundles/sas.tar.gz" SAS_BUNDLE_SHA256="$sas_sha" O4_REAL_SAS_FIXTURE="$REAL_SAS_FIXTURE")
  bash -n "$0"; bash -n "$STUB_NEMO"; bash -n "$STUB_MPIRUN"
  awk '{print} /O4_DRIVER_TRUNCATION_POINT/ {exit}' "$0" > "$scratch/run_truncated.sh"; expect_refusal bash -n "$scratch/run_truncated.sh"
  "${envs[@]}" EVIDENCE="$evidence" "$0" --run
  "${envs[@]}" EVIDENCE="$evidence" "$0" --admit-existing
  expect_refusal "${envs[@]}" EVIDENCE="$evidence" "$0" --run
  "${envs[@]}" "$0" --move-aside-builds; mv "$evidence" "$refused/success_evidence"
  expect_refusal "${envs[@]}" O4_STUB_FAIL=1 EVIDENCE="$failed" "$0" --run
  find "$refused" -maxdepth 1 -type d -name 'stub_failure_*_rc*' | grep . >/dev/null || refuse 'stub failure evidence not retained'
  [[ -d "$root/cfgs/STUB_O4_REF" && -d "$root/cfgs/STUB_O4_REC" ]] || refuse 'stub failure builds not retained'
  "${envs[@]}" "$0" --move-aside-builds
  printf 'PASS_STUB_RUN_REFUSAL_RETENTION_ADMIT_RERUN\n'
}

run_preflight() {
  local scratch plant
  scratch=$(mktemp -d); ACTIVE_SCRATCH=$scratch
  printf 'PREFLIGHT 1/10 exact O4 deck, group membership, and misplaced-variable plants\n'
  awk -v cfg="$SOURCE_CFG" '$1==cfg {found=1} END {exit !found}' "$NEMO_ROOT/cfgs/ref_cfgs.txt"
  render_deck "$scratch/deck"; verify_namelists "$scratch/deck"
  for plant in misspelled misplaced o4-misplaced; do expect_refusal verify_namelists "$scratch/deck" "$plant"; done
  if [[ "$mode" != --admit-existing ]]; then [[ ! -e "$REF_ROOT" && ! -e "$REC_ROOT" && ! -e "$EVIDENCE" ]] || refuse 'target config/evidence exists; after refusal run --move-aside-builds then --run'; fi
  if [[ ${O4_DRIVER_FIXTURE_ACTIVE:-0} == 1 ]]; then
    bash -n "$0"; bash -n "$STUB_NEMO"; bash -n "$STUB_MPIRUN"
    printf 'PREFLIGHT_OK: nested stub driver uses outer full preflight\n'; ACTIVE_SCRATCH=""; return
  fi
  printf 'PREFLIGHT 2/10 fuzz-zero patch and real recorded CPP output\n'; prepare_patched_ppsrc "$scratch/patched"
  printf 'PREFLIGHT 3/10 like-with-like same-program proof and plants\n'
  same_program_gate "$scratch/patched/icedyn_adv_pra.f90" "$BASE_BLD/ppsrc/nemo/icedyn_adv_pra.f90"
  expect_refusal same_program_gate "$scratch/patched/icedyn_adv_pra.f90" "$BASE_BLD/ppsrc/nemo/icedyn_adv_pra.f90" unmarked
  expect_refusal same_program_gate "$scratch/patched/icedyn_adv_pra.f90" "$BASE_BLD/ppsrc/nemo/icedyn_adv_pra.f90" marked-content
  printf 'PREFLIGHT 4/10 entry/in-body/exit placement and one plant per capture\n'; placement_gate "$scratch/patched/icedyn_adv_pra.f90"
  for plant in entry current-cfl current-before-delay delayed-cfl delayed-after-timestep delayed-before-loop exit exit-state; do expect_refusal placement_gate "$scratch/patched/icedyn_adv_pra.f90" "$plant"; done
  printf 'PREFLIGHT 5/10 source-derived extents/read-set and plants\n'
  "$PY" "$EXTENTS" --ppsrc "$BASE_BLD/ppsrc/nemo" --writer "$WRITER" --deck-log "$REAL_EXCERPT" > "$scratch/extents.json"
  for plant in missing-allocation writer-mode dummy-actual; do expect_refusal "$PY" "$EXTENTS" --ppsrc "$BASE_BLD/ppsrc/nemo" --writer "$WRITER" --deck-log "$REAL_EXCERPT" --plant "$plant"; done
  expect_refusal "$PY" "$EXTENTS" --ppsrc "$BASE_BLD/ppsrc/nemo" --writer "$WRITER" --deck-log "$REAL_EXCERPT" --fixture-include "$scratch/planted.inc" --plant fixture-allocation
  "$PY" "$READSET" --ppsrc "$BASE_BLD/ppsrc/nemo" > "$scratch/readset.json"; expect_refusal "$PY" "$READSET" --ppsrc "$BASE_BLD/ppsrc/nemo" --plant drop-sxice
  printf 'PREFLIGHT 6/10 real patched source compiled and run with -fcheck=bounds\n'; compile_bounds_fixture "$scratch/bounds"
  printf 'PREFLIGHT 7/10 real -O3 per-function FP census and injected-op plant\n'; "$PY" "$PASSIVITY" > "$scratch/passivity.json"
  printf 'PREFLIGHT 8/10 independent stub records, reader and production-score plants\n'; stub_score_and_plants "$scratch/scorer"
  printf 'PREFLIGHT 9/10 generated shell syntax, resources, and full stub driver control flow\n'; ram_gate 1; disk_gate "$EVIDENCE" 1024; grep -q 'run.user.stderr.log' "$0"
  if [[ ${O4_DRIVER_FIXTURE_ACTIVE:-0} == 0 ]]; then stub_driver_fixture "$scratch"; fi
  printf 'PREFLIGHT 10/10 acquisition remains operator-only\n'
  if [[ -n "${PREFLIGHT_JSON:-}" ]]; then printf '{"status":"PREFLIGHT_OK","nemo_run":false,"plants":45}\n' > "$PREFLIGHT_JSON"; fi
  printf 'PREFLIGHT_OK: no real NEMO build or integration was run\n'; ACTIVE_SCRATCH=""
}

if [[ "$mode" == --move-aside-builds ]]; then
  # Refused builds are never deleted. Rerun exactly:
  #   run.sh --move-aside-builds
  #   run.sh --run
  move_builds_aside; exit 0
fi
if [[ "$mode" == --preflight-only ]]; then run_preflight; exit 0; fi
if [[ "$mode" == --admit-existing ]]; then
  run_preflight
  [[ -f "$EVIDENCE/manifest/run_manifest.json" ]] || refuse 'missing run manifest'
  commit=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["producer_commit"])' "$EVIDENCE/manifest/run_manifest.json")
  admit "$commit"; printf 'ACQUISITION_ADMITTED: %s\n' "$EVIDENCE"; exit 0
fi

# ---- Operator-only --run mode begins here. ----
if [[ ${O4_DRIVER_FIXTURE_ACTIVE:-0} == 0 ]]; then [[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] || refuse '--run requires a clean committed tree'; fi
readonly COMMIT=$(git -C "$REPO" rev-parse HEAD)
RUN_STARTED_EPOCH=$(date +%s)
run_preflight
[[ ! -e "$REF_ROOT" && ! -e "$REC_ROOT" && ! -e "$EVIDENCE" ]] || refuse 'target config/evidence already exists'
ram_gate "$MIN_FREE_RAM_GB"; disk_gate "$EVIDENCE" "$MIN_FREE_DISK_KB"
mkdir -p "$EVIDENCE/input" "$EVIDENCE/manifest/production_deck" "$EVIDENCE/manifest/binaries"; ACTIVE_RUN=$EVIDENCE
for pair in "$ORCA2_BUNDLE:$ORCA2_BUNDLE_SHA256" "$SAS_BUNDLE:$SAS_BUNDLE_SHA256"; do file=${pair%%:*}; want=${pair##*:}; got=$(sha256sum "$file"); [[ ${got%% *} == "$want" ]] || refuse "bundle checksum $file"; done
tar -xzf "$ORCA2_BUNDLE" --strip-components=1 -C "$EVIDENCE/input"; tar -xzf "$SAS_BUNDLE" --strip-components=1 -C "$EVIDENCE/input"
render_deck "$EVIDENCE/manifest/production_deck"; verify_namelists "$EVIDENCE/manifest/production_deck"

build_target() {
  local cfg=$1 instrument=$2 root=$NEMO_ROOT/cfgs/$1
  [[ "$cfg" == "$REF_CFG" ]] && CREATED_REF=1 || CREATED_REC=1
  if [[ -n "${NEMO_EXE_OVERRIDE:-}" ]]; then mkdir -p "$root/BLD/bin"; cp "$NEMO_EXE_OVERRIDE" "$root/BLD/bin/nemo.exe"; chmod +x "$root/BLD/bin/nemo.exe"; return; fi
  cd "$NEMO_ROOT"; ./makenemo -r "$SOURCE_CFG" -n "$cfg" -m conda-scalarmath del_key 'key_xios'
  if [[ "$instrument" -eq 1 ]]; then
    cp "$NEMO_SOURCE_ROOT/src/ICE/icedyn_adv_pra.F90" "$root/MY_SRC/icedyn_adv_pra.F90"; patch --batch --fuzz=0 "$root/MY_SRC/icedyn_adv_pra.F90" "$PATCH" >/dev/null
    cp "$WRITER" "$root/MY_SRC/o4_prather_writer.F90"; ./makenemo -n "$cfg" -m conda-scalarmath del_key 'key_xios'
  fi
  [[ -x "$root/BLD/bin/nemo.exe" ]] || refuse "$cfg lacks nemo.exe"
}

build_target "$REF_CFG" 0; build_target "$REC_CFG" 1
cp "$REF_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.ref"; cp "$REC_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.rec"
if [[ ${O4_DRIVER_FIXTURE_ACTIVE:-0} == 0 ]]; then
  mkdir -p "$EVIDENCE/manifest/expected_ppsrc"; prepare_patched_ppsrc "$EVIDENCE/manifest/expected_ppsrc"
  cmp -s "$EVIDENCE/manifest/expected_ppsrc/icedyn_adv_pra.f90" "$REC_ROOT/BLD/ppsrc/nemo/icedyn_adv_pra.f90" || refuse 'real record build ppsrc differs from expected patch'
  same_program_gate "$REC_ROOT/BLD/ppsrc/nemo/icedyn_adv_pra.f90" "$REF_ROOT/BLD/ppsrc/nemo/icedyn_adv_pra.f90" > "$EVIDENCE/manifest/same_program.txt"
  "$PY" "$PASSIVITY" --reference-object "$REF_ROOT/BLD/obj/icedyn_adv_pra.o" \
    --record-object "$REC_ROOT/BLD/obj/icedyn_adv_pra.o" --arch "$REC_ROOT/BLD/arch_nemo.fcm" \
    --json "$EVIDENCE/manifest/fp_census.json" >/dev/null
else
  printf 'STUB_SAME_PROGRAM_EXERCISED_BY_PREFLIGHT\n' > "$EVIDENCE/manifest/same_program.txt"; printf '{"fixture":true,"pass":true}\n' > "$EVIDENCE/manifest/fp_census.json"
fi
write_manifest "$COMMIT" "${O4_DRIVER_FIXTURE_ACTIVE:-0}"

remaining_timeout() { local elapsed remaining; elapsed=$(( $(date +%s) - RUN_STARTED_EPOCH )); remaining=$(( RUN_TIMEOUT_SEC - elapsed )); [[ "$remaining" -gt 0 ]] || refuse "global wall-clock budget ${RUN_TIMEOUT_SEC}s exhausted"; printf '%s\n' "$remaining"; }
stage_run() {
  local cfg=$1 dir=$2 record=$3 root=$NEMO_ROOT/cfgs/$1 f remaining rc=0
  mkdir -p "$dir"; [[ "$record" -eq 0 ]] || mkdir -p "$dir/o4_prather_records"
  render_deck "$dir"; verify_namelists "$dir"
  for f in "$EVIDENCE/input"/*.nc; do ln -s "$f" "$dir/$(basename "$f")"; done
  cp "$root/BLD/bin/nemo.exe" "$dir/nemo"; remaining=$(remaining_timeout)
  (cd "$dir" && printf 'RUN_STARTED\n' > run.user.time.log && timeout "$remaining" mpirun -np 1 --oversubscribe ./nemo > run.user.stdout.log 2> run.user.stderr.log) || rc=$?
  if [[ -f "$dir/ocean.output" ]] && grep -E 'E[[:space:]]+R[[:space:]]+R[[:space:]]+O[[:space:]]+R|ABORT' "$dir/ocean.output" "$dir/run.user.stdout.log" "$dir/run.user.stderr.log" >/dev/null; then
    grep -E 'E[[:space:]]+R[[:space:]]+R[[:space:]]+O[[:space:]]+R|ABORT' "$dir/ocean.output" "$dir/run.user.stdout.log" "$dir/run.user.stderr.log" >&2 || true; [[ "$rc" -ne 0 ]] || rc=1
  fi
  [[ "$rc" -eq 0 ]] || return "$rc"
  printf 'MPIRUN_RC=0\nRUN DONE\n' >> "$dir/run.user.time.log"; touch "$dir/RUN_COMPLETE"
}

export O4_REAL_SAS_FIXTURE=$REAL_SAS_FIXTURE O4_STUB_WRITER=$STUB_WRITER O4_PY=$PY
stage_run "$REF_CFG" "$EVIDENCE/reference_2step" 0; stage_run "$REC_CFG" "$EVIDENCE/record_2step" 1
touch "$EVIDENCE/RUN_COMPLETE"
(cd "$EVIDENCE" && find reference_2step record_2step manifest input -type f ! -name artifacts.sha256 ! -name o4_admission.json -print0 | sort -z | xargs -0 sha256sum > manifest/artifacts.sha256)
size=$(du -sk "$EVIDENCE" | awk '{print $1}'); [[ "$size" -le "$MAX_EVIDENCE_KB" ]] || refuse "evidence ${size} KiB exceeds $MAX_EVIDENCE_KB"
admit "$COMMIT"; ACTIVE_RUN=""
printf 'ACQUISITION_DONE: %s\n' "$EVIDENCE"
