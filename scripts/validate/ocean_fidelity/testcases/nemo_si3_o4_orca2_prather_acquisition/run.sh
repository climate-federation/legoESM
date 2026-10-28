#!/usr/bin/env bash
# SI3 O4 PRATHER ACQUISITION -- operator execution only; default is read-only preflight.
set -Eeuo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly BASE_BLD=${BASE_BLD:-$NEMO_ROOT/cfgs/ORCA2_SAS_ICE_O7_REF/BLD}
readonly TOOLCHAIN=${TOOLCHAIN:-/home/dbalwada/miniconda3/envs/nemo-build}
readonly EVIDENCE_ROOT=${EVIDENCE_ROOT:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round25}
readonly EVIDENCE=${EVIDENCE:-$EVIDENCE_ROOT/oracle_o4_prather}
readonly REFUSED=${REFUSED:-$EVIDENCE_ROOT/refused}
readonly MIN_FREE_RAM_GB=${MIN_FREE_RAM_GB:-150}
readonly MIN_FREE_DISK_KB=${MIN_FREE_DISK_KB:-30000000}
readonly MAX_EVIDENCE_KB=${MAX_EVIDENCE_KB:-25000000}
readonly RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-3600}
readonly PY=${PYTHON_BIN:-python3}
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
export PATH="$TOOLCHAIN/bin:$PATH"
export PYTHONPATH="$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/ice:$REPO/packages/atmosphere:$REPO/packages/land:$REPO/packages/coupler:$REPO/packages/driver"
readonly GATE=$here/../nemo_si3_o4_orca2_prather_gate.py
readonly READSET=$here/../nemo_si3_o4_orca2_prather_readset.py
readonly PASSIVITY=$here/../nemo_si3_o4_orca2_prather_passivity.py
readonly EXTENTS=$here/../../orca2_l4/nemo_acquisition_extents.py
readonly WRITER=$here/o4_prather_writer.F90
readonly PATCH=$here/icedyn_adv_pra_o4.patch
readonly REAL_SEED=$here/../nemo_si3_o7_orca2_sas_acquisition/ocean.output.excerpt
readonly PPSRC=$BASE_BLD/ppsrc/nemo

mode=${1:---preflight-only}; [[ $# -le 2 ]] || { echo 'REFUSE: too many arguments' >&2; exit 64; }
case "$mode" in --preflight-only) out=${2:-};; --run|--admit-existing) out=${2:-$EVIDENCE};; *) echo "REFUSE: usage: $0 [--preflight-only|--run|--admit-existing] [path]" >&2; exit 64;; esac

active=""
cleanup() { local rc=$?; if [[ $rc -ne 0 && -n $active && -d $active ]]; then mkdir -p "$REFUSED"; local dst="$REFUSED/$(basename "$active")_$(date -u +%Y%m%dT%H%M%SZ)_rc${rc}_$$"; mv "$active" "$dst"; echo "REFUSE_RETAINED: $dst" >&2; fi; exit "$rc"; }
trap cleanup EXIT
trap 'echo "REFUSE: command failed at line $LINENO" >&2' ERR
refuse(){ echo "REFUSE: $*" >&2; return 1; }

resource_gates(){
  local ram disk; ram=$(free -g | awk '/^Mem:/{print $7}'); [[ ${ram:-0} -ge $MIN_FREE_RAM_GB ]] || refuse "available RAM ${ram:-unknown} GB below $MIN_FREE_RAM_GB"
  mkdir -p "$EVIDENCE_ROOT"; disk=$(df -Pk "$EVIDENCE_ROOT" | awk 'NR==2{print $4}'); [[ ${disk:-0} -ge $MIN_FREE_DISK_KB ]] || refuse "free disk below $MIN_FREE_DISK_KB KB"
}

render_deck(){
  local d=$1; mkdir -p "$d"
  cp "$NEMO_ROOT/cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg" "$d/namelist_cfg"
  cp "$NEMO_ROOT/cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg" "$d/namelist_ice_cfg"
  cp -L "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$d/namelist_ref"
  cp -L "$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref" "$d/namelist_ice_ref"
  "$PY" - "$d" <<'PY'
import re,sys
from pathlib import Path
d=Path(sys.argv[1])
def replace(path,name,value):
 s=path.read_text(); s,n=re.subn(rf'(^\s*{name}\s*=\s*)[^!,\n]+',rf'\g<1>{value} ',s,count=1,flags=re.I|re.M)
 if n!=1: raise SystemExit(f'STATUS PLANT-FIRED: absent {name}')
 path.write_text(s)
for n,v in [('nn_it000','1'),('nn_itend','2'),('nn_fsbc','1')]: replace(d/'namelist_cfg',n,v)
p=d/'namelist_cfg'; s=p.read_text(); m=re.search(r'(^\s*&namrun\b[\s\S]*?)(^\s*/\s*$)',s,re.I|re.M)
if not m: raise SystemExit('STATUS PLANT-FIRED: absent &namrun')
s=s[:m.start(2)]+'   nn_stock = 1\n   nn_write = -1\n'+s[m.start(2):]; p.write_text(s)
p=d/'namelist_ice_cfg'; s=p.read_text()
for group,lines in [('nampar',['ln_icedyn = .true.','ln_icethd = .false.']),('namdyn_adv',['ln_dynADV2D = .true.'])]:
 m=re.search(rf'(^\s*&{group}\b[^\n]*\n)',s,re.I|re.M)
 if not m: raise SystemExit(f'STATUS PLANT-FIRED: absent &{group}')
 s=s[:m.end()]+''.join(f'   {line}\n' for line in lines)+s[m.end():]
p.write_text(s)
for n,v in [('rn_uice','0.5'),('rn_vice','0.5')]: replace(d/'namelist_ice_ref',n,v)
PY
  grep -Eiq '^\s*ln_icedyn\s*=\s*\.true\.' "$d/namelist_ice_cfg"
  grep -Eiq '^\s*ln_icethd\s*=\s*\.false\.' "$d/namelist_ice_cfg"
  grep -Eiq '^\s*ln_dynADV2D\s*=\s*\.true\.' "$d/namelist_ice_cfg"
  [[ $(grep -Eic '^\s*rn_(uice|vice)\s*=\s*0\.5' "$d/namelist_ice_ref") -eq 2 ]]
}

static_gates(){
  patch --dry-run --batch --fuzz=0 "$NEMO_ROOT/src/ICE/icedyn_adv_pra.F90" "$PATCH" >/dev/null
  "$PY" "$EXTENTS" --ppsrc "$PPSRC" --writer "$WRITER" --deck-log "$REAL_SEED" >/dev/null
  "$PY" "$READSET" --ppsrc "$PPSRC" >/dev/null
  if "$PY" "$READSET" --ppsrc "$PPSRC" --plant drop-sxice >/dev/null 2>&1; then refuse 'read-set plant survived'; fi
  for plant in missing-allocation writer-mode dummy-actual; do
    if "$PY" "$EXTENTS" --ppsrc "$PPSRC" --writer "$WRITER" --deck-log "$REAL_SEED" --plant "$plant" >/dev/null 2>&1; then refuse "extent plant $plant survived"; fi
  done
  local fixture_plant; fixture_plant=$(mktemp /tmp/o4_fixture_plant.XXXXXX)
  if "$PY" "$EXTENTS" --ppsrc "$PPSRC" --writer "$WRITER" --deck-log "$REAL_SEED" --fixture-include "$fixture_plant" --plant fixture-allocation >/dev/null 2>&1; then refuse 'extent plant fixture-allocation survived'; fi
}

stub_flow(){
  local d=$1; mkdir -p "$d/o4_prather_records"
  "$PY" "$here/write_stub_records.py" "$d/o4_prather_records" --source-real "$REAL_SEED"
  "$PY" "$GATE" "$d/o4_prather_records" > "$d/score.json"
  "$PY" - "$d/o4_prather_records" <<'PY'
import sys
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_gate import score_deck
for plant in ('header','truncation','step','one-bit','invert-sweep','wrong-fold-sign','omit-subcycle-lbc'):
    try: score_deck(sys.argv[1],plant)
    except ValueError as exc:
        if 'STATUS PLANT-FIRED' in str(exc): continue
        raise
    raise SystemExit(f'REFUSE: scorer plant {plant} survived')
print('PASS_O4_SCORER_PLANTS')
PY
}

manifest_before_run(){
  local d=$1 refexe=$2 recexe=$3
  "$PY" - "$d/manifest.json" "$refexe" "$recexe" "$REPO" "$PATCH" "$WRITER" <<'PY'
import hashlib,json,subprocess,sys,time
from pathlib import Path
out,ref,rec,repo,patch,writer=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
doc={'schema':'si3-o4-prather-v1','created_before_nemo':True,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'producer_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
 'binaries':{'reference':sha(ref),'recording':sha(rec)},'observer':{'patch':sha(patch),'writer':sha(writer)},
 'planned_records':[f'o4_kind{k:02d}_step{s:04d}_rank0000.bin' for s in (1,2) for k in range(1,7)],
 'claim':'every production Prather exit, cyclic seam, and north fold cell is 0 ULP'}
out.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
PY
}

admit(){
  local d=$1; [[ -s $d/manifest.json && -s $d/checksums.sha256 && -s $d/reference/ocean.output && -s $d/recording/ocean.output ]] || refuse 'incomplete evidence'
  (cd "$d" && sha256sum -c checksums.sha256 >/dev/null)
  ! grep -Eiq 'E R R O R|ABORT|STOP [1-9]' "$d/reference/ocean.output" "$d/recording/ocean.output" || refuse 'NEMO log failure marker'
  cmp -s "$d/reference/restart_ice.nc" "$d/recording/restart_ice.nc" || refuse 'observer changed ice restart'
  "$PY" "$GATE" "$d/recording/o4_prather_records" > "$d/score.json"
  local size; size=$(du -sk "$d"|awk '{print $1}'); [[ $size -le $MAX_EVIDENCE_KB ]] || refuse "evidence ${size} KB exceeds cap"
  echo "ADMITTED: $d"
}

if [[ $mode == --preflight-only ]]; then
  bash -n "$0"; static_gates
  scratch=$(mktemp -d /tmp/si3_o4_preflight.XXXXXX); render_deck "$scratch/deck"; stub_flow "$scratch/stub"
  "$PY" "$PASSIVITY" > "$scratch/passivity.json"
  [[ -z $out ]] || cp "$scratch/score.json" "$out"
  echo "PREFLIGHT_PASS: $scratch"; exit 0
fi
if [[ $mode == --admit-existing ]]; then admit "$out"; exit 0; fi

resource_gates
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=no) ]] || refuse 'tracked tree is dirty'
active=$out; [[ ! -e $active ]] || refuse "evidence path exists: $active"; mkdir -p "$active/reference" "$active/recording/o4_prather_records"
render_deck "$active/deck"
# Operator may supply prebuilt observer-free/recording executables. Building is deliberately explicit and outside NEMO execution.
readonly REF_EXE=${O4_REFERENCE_EXE:-$NEMO_ROOT/cfgs/ORCA2_SAS_ICE_O4_REF/BLD/bin/nemo.exe}
readonly REC_EXE=${O4_RECORDING_EXE:-$NEMO_ROOT/cfgs/ORCA2_SAS_ICE_O4_REC/BLD/bin/nemo.exe}
[[ -x $REF_EXE && -x $REC_EXE ]] || refuse 'build O4 reference/recording executables first (preflight never builds or runs NEMO)'
manifest_before_run "$active" "$REF_EXE" "$REC_EXE"
for lane in reference recording; do
  exe=$REF_EXE; [[ $lane == recording ]] && exe=$REC_EXE
  cp "$active/deck"/* "$active/$lane/"
  (cd "$active/$lane" && /usr/bin/time -v -o resource.txt timeout "$RUN_TIMEOUT_SEC" mpirun -np 1 "$exe" > ocean.output 2>&1)
done
find "$active" -type f \( -name '*.nc' -o -name '*.bin' -o -name 'ocean.output' -o -name 'manifest.json' \) -print0 | sort -z | xargs -0 sha256sum | sed "s#  $active/#  #" > "$active/checksums.sha256"
admit "$active"; active=""
