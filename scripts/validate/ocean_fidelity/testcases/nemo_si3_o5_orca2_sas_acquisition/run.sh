#!/usr/bin/env bash
# SI3 O5 NEMO ACQUISITION -- OPERATOR EXECUTION ONLY.
# Default is read-only preflight.  Only --run invokes makenemo/mpirun.
set -Eeuo pipefail

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=ORCA2_SAS_ICE
readonly REF_CFG=${REF_CFG:-ORCA2_SAS_ICE_O5_REF}
readonly REC_CFG=${REC_CFG:-ORCA2_SAS_ICE_O5_REC}
readonly EVIDENCE_ROOT=${EVIDENCE_ROOT:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round12}
EVIDENCE=${EVIDENCE:-$EVIDENCE_ROOT/oracle_o5_orca2_sas_ice}
readonly REFUSED_DIR=${REFUSED_DIR:-$EVIDENCE_ROOT/refused}
readonly RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-1800}
readonly MIN_FREE_RAM_GB=${MIN_FREE_RAM_GB:-150}
readonly MIN_FREE_DISK_KB=${MIN_FREE_DISK_KB:-8388608}
readonly ORCA2_BUNDLE=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0.tar.gz
readonly ORCA2_BUNDLE_SHA256=5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4
readonly SAS_BUNDLE=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/SAS_v5.0.0_download/SAS_v5.0.0.tar.gz
readonly SAS_BUNDLE_SHA256=ee00646832ea89f468d0f01bf537ed231d21f23bba77ab1a9aab26147ecec6cf
readonly PY=${PYTHON_BIN:-/home/dbalwada/legoESM/.venv/bin/python3}
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
export PYTHONPATH="$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/ice:$REPO/packages/atmosphere:$REPO/packages/land:$REPO/packages/coupler:$REPO/packages/driver"
readonly GATE=$here/../nemo_si3_o5_orca2_sas_gate.py
readonly WRITER=$here/o5_si3_writer.F90
readonly STUBS=$here/o5_si3_fixture_stubs.F90
readonly FIXTURE=$here/o5_si3_layout_fixture.F90
readonly RHG_PATCH=$here/icedyn_rhg_evp_o5.patch
readonly RDG_PATCH=$here/icedyn_rdgrft_o5.patch
readonly DYN_PATCH=$here/icedyn_o5.patch
readonly NML=$here/namelist_cfg
readonly ICE_NML=$here/namelist_ice_cfg
readonly REAL_EXCERPT=$here/ocean.output.excerpt
readonly REF_ROOT=$NEMO_ROOT/cfgs/$REF_CFG
readonly REC_ROOT=$NEMO_ROOT/cfgs/$REC_CFG

mode=${1:---preflight-only}
arg=${2:-}
case "$mode" in
  --preflight-only) PREFLIGHT_JSON=$arg ;;
  --run|--admit-existing) [[ -z "$arg" ]] || EVIDENCE=$arg ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing] [path]\n' "$0" >&2; exit 64 ;;
esac

ACTIVE_RUN=""; ACTIVE_SCRATCH=""; CREATED_REF=0; CREATED_REC=0; RUN_COMPLETE=0
cleanup() {
  local rc=$?
  [[ -z "$ACTIVE_SCRATCH" || ! -d "$ACTIVE_SCRATCH" ]] || rm -rf -- "$ACTIVE_SCRATCH"
  if [[ "$rc" -ne 0 ]]; then
    mkdir -p "$REFUSED_DIR"
    if [[ -n "$ACTIVE_RUN" && -d "$ACTIVE_RUN" ]]; then
      local dst="$REFUSED_DIR/$(basename "$ACTIVE_RUN")_$(date -u +%Y%m%dT%H%M%SZ)_rc${rc}_$$"
      mv "$ACTIVE_RUN" "$dst"
      printf 'REFUSE_RETained: %s\n' "$dst" >&2
    fi
    [[ "$CREATED_REF" -eq 0 || "$RUN_COMPLETE" -eq 1 || ! -d "$REF_ROOT" ]] || rm -rf -- "$REF_ROOT"
    [[ "$CREATED_REC" -eq 0 || "$RUN_COMPLETE" -eq 1 || ! -d "$REC_ROOT" ]] || rm -rf -- "$REC_ROOT"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'printf "REFUSE: command failed at line %s; EXIT trap retains logs\n" "$LINENO" >&2' ERR

refuse() { printf 'REFUSE: %s\n' "$*" >&2; return 1; }
ram_gate() {
  local avail; avail=$(free -g | awk '/^Mem:/ {print $7}')
  [[ -n "$avail" && "$avail" -ge "$1" ]] || refuse "available RAM ${avail:-unknown} GB below $1 GB"
}
disk_gate() {
  local probe=$1 avail
  while [[ ! -e "$probe" ]]; do probe=$(dirname "$probe"); done
  avail=$(df -Pk "$probe" | awk 'NR==2 {print $4}')
  [[ -n "$avail" && "$avail" -ge "$2" ]] || refuse "free disk ${avail:-unknown} KB below $2 KB"
}

verify_deck() {
  "$PY" - "$NML" "$ICE_NML" "$NEMO_ROOT/cfgs/$SOURCE_CFG/EXPREF/namelist_cfg" \
    "$NEMO_ROOT/cfgs/$SOURCE_CFG/EXPREF/namelist_ice_cfg" "$NEMO_ROOT/cfgs/SHARED/namelist_ref" \
    "$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref" <<'PY'
import pathlib, re, sys
oce, ice, base_oce, base_ice, ref_oce, ref_ice = [pathlib.Path(p).read_text() for p in sys.argv[1:]]
def vals(text):
    return {m.group(1).lower(): m.group(2).strip().lower()
            for m in re.finditer(r"^\s*([a-z][a-z0-9_]*)\s*=\s*([^!,\n]+)", text, re.I|re.M)}
o, i, bo, bi = map(vals, (oce, ice, base_oce, base_ice))
eo, ei = vals(ref_oce+"\n"+base_oce+"\n"+oce), vals(ref_ice+"\n"+base_ice+"\n"+ice)
want_o = {"nn_it000":"1", "nn_itend":"3", "nn_stock":"3", "nn_fsbc":"1", "ln_ice_embd":".false."}
want_i = {"ln_icedyn":".true.", "ln_icethd":".false.", "ln_dynall":".true.",
          "ln_dynrhgadv":".false.", "ln_ridging":".false.", "ln_rafting":".false.",
          "ln_str_h79":".true.", "ln_str_smooth":".true.", "ln_aevp":".true.", "nn_nevp":"100"}
for key, value in {**want_o, **want_i}.items():
    got = eo.get(key, ei.get(key))
    if got != value: raise SystemExit(f"STATUS PLANT-FIRED: {key}={got}, expected {value}")
allowed_o = {"nn_itend", "nn_stock", "nn_write", "ln_ice_embd"}
allowed_i = {"ln_icedyn", "ln_icethd", "ln_dynall", "ln_dynrhgadv", "ln_ridging", "ln_rafting"}
if {k for k in o|bo if o.get(k)!=bo.get(k)} != allowed_o:
    raise SystemExit("STATUS PLANT-FIRED: ocean deck has unregistered deviations")
if {k for k in i|bi if i.get(k)!=bi.get(k)} != allowed_i:
    raise SystemExit("STATUS PLANT-FIRED: ice deck has unregistered deviations")
PY
}

check_patches_and_landings() {
  local scratch=$1
  cp "$NEMO_ROOT/src/ICE/icedyn_rhg_evp.F90" "$scratch/icedyn_rhg_evp.F90"
  cp "$NEMO_ROOT/src/ICE/icedyn_rdgrft.F90" "$scratch/icedyn_rdgrft.F90"
  cp "$NEMO_ROOT/src/ICE/icedyn.F90" "$scratch/icedyn.F90"
  patch -s --fuzz=0 "$scratch/icedyn_rhg_evp.F90" < "$RHG_PATCH"
  patch -s --fuzz=0 "$scratch/icedyn_rdgrft.F90" < "$RDG_PATCH"
  patch -s --fuzz=0 "$scratch/icedyn.F90" < "$DYN_PATCH"
  "$PY" - "$scratch" <<'PY'
import pathlib, re, sys
root=pathlib.Path(sys.argv[1])
text="\n".join(p.read_text() for p in (root/'icedyn_rhg_evp.F90',root/'icedyn_rdgrft.F90',root/'icedyn.F90'))
checks={
"o5_context":r"CALL o5_context\([\s\S]{0,220}?\)\n\s*IF\( kt == nit000",
"o5_fimask":r"CALL lbc_lnk\( 'icedyn_rhg_evp', fimask[^\n]+\)\n\s*CALL o5_fimask",
"o5_strength_entry":r"CALL o5_strength_entry\([\s\S]{0,220}?\)\n\s*CALL ice_strength",
"o5_strength_smooth":r"CALL ice_strength\n\s*CALL o5_strength_smooth",
"o5_aevp_entry_state":r"zsshdyn\(:,:\) = ice_var_sshdyn[^\n]+\n\s*CALL o5_aevp_entry_state",
"o5_frame_delta":r"CALL lbc_lnk\( 'icedyn_rhg_evp', zdelta[^\n]+\n[\s\S]{0,180}?CALL o5_frame_delta",
"o5_frame_stress":r"CALL o5_frame_stress[\s\S]{0,100}?\n\s*! --- Ice internal stresses",
"o5_frame_velocity":r"ENDIF\n[\s\S]{0,180}?CALL o5_frame_velocity[\s\S]{0,80}?#if defined key_agrif",
"o5_aevp_exit":r"pstress12_i\(:,:\) = zs12\(:,:\)\n\s*CALL o5_aevp_exit",
"o5_strength_raw":r"CALL o5_strength_raw[^\n]+\n\s*CASE \( np_strcst \)",
"o5_rdgrft_entry":r"CALL o5_rdgrft_entry[^\n]+\n\s*CALL ice_dyn_rdgrft",
"o5_rdgrft_exit":r"CALL ice_dyn_rdgrft[^\n]+\n\s*CALL o5_rdgrft_exit",
}
def verify(value):
    missing=[name for name,pat in checks.items() if re.search(pat,value,re.M) is None]
    if missing: raise ValueError(','.join(missing))
verify(text)
for name in checks:
    planted=text.replace(f"CALL {name}", f"CALL misplaced_{name}", 1)+f"\nCALL {name}\n"
    try: verify(planted)
    except ValueError: continue
    raise SystemExit(f"STATUS PLANT-FIRED: misplacement plant inert for {name}")
print("PASS_LANDINGS_AND_12_MISPLACEMENT_PLANTS")
PY
}

compile_fixture() {
  local scratch=$1
  mkdir -p "$scratch/o5_si3_records"
  (cd "$scratch" && "$FC" -O2 -fcheck=bounds -fbacktrace -Wall -Wextra -J. -I. \
    "$STUBS" "$WRITER" "$FIXTURE" -o o5_fixture && ./o5_fixture)
  [[ $(find "$scratch/o5_si3_records" -type f | wc -l) -eq 82 ]] || refuse "fixture record count is not 82"
  "$PY" "$GATE" --read-record "$scratch/o5_si3_records/o5_kind01_step0001_iter000_rank0000.bin" >/dev/null
}

reader_plants() {
  "$PY" "$GATE" --self-test >/dev/null
  local plant
  for plant in header truncation checksum time-level wet-1ulp; do
    if "$PY" "$GATE" --self-test --plant "$plant" >/dev/null 2>&1; then
      refuse "reader plant $plant did not refuse"
    fi
  done
}

verify_real_excerpt() {
  "$PY" - "$REAL_EXCERPT" <<'PY'
import pathlib, sys
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o1_orca2_sas_gate import verify_ocean_output_log
p=pathlib.Path(sys.argv[1]); lines=p.read_text().splitlines()
src=next(x.split(':',1)[1].strip() for x in lines if x.startswith('# Source path:'))
sha=next(x.split(':',1)[1].strip() for x in lines if x.startswith('# Complete source SHA-256:'))
spans=next(x.split(':',1)[1].strip().rstrip('.') for x in lines if x.startswith('# Source lines:'))
import hashlib
source=pathlib.Path(src)
if hashlib.sha256(source.read_bytes()).hexdigest()!=sha:
    raise SystemExit('STATUS PLANT-FIRED: committed real excerpt provenance mismatch')
numbers=[]
for span in spans.split(','):
    ends=[int(x.strip()) for x in span.split('-')]
    numbers.extend(range(ends[0],ends[-1]+1))
body=[x.rstrip() for x in lines if not x.startswith('#')]
source_lines=source.read_text(errors='replace').splitlines()
if body != [source_lines[n-1].rstrip() for n in numbers]:
    raise SystemExit('STATUS PLANT-FIRED: committed excerpt body differs from cited real output lines')
verify_ocean_output_log(p, expect_icedyn=False, expect_icethd=False)
print('PASS_REAL_OCEAN_OUTPUT_EXCERPT')
PY
}

run_preflight() {
  local scratch; scratch=$(mktemp -d); ACTIVE_SCRATCH=$scratch
  printf 'PREFLIGHT 1/8 deck and one-rank three-step overlay\n'; verify_deck
  printf 'PREFLIGHT 2/8 source patches, call-point landings, misplacement plants\n'; check_patches_and_landings "$scratch"
  printf 'PREFLIGHT 3/8 bounds-checked observer fixture\n'; compile_fixture "$scratch/fixture"
  printf 'PREFLIGHT 4/8 header/truncation/checksum/time-level/wet-ULP plants\n'; reader_plants
  printf 'PREFLIGHT 5/8 committed real ocean.output excerpt\n'; verify_real_excerpt
  printf 'PREFLIGHT 6/8 SAS time level derived from Nnn source\n'
  grep -q 'Nbb = 1; Nnn = 2; Naa = 3; Nrhs = Naa' "$NEMO_ROOT/src/SAS/nemogcm.F90"
  grep -q 'CALL sbc    ( kstp, Nbb, Nnn )' "$NEMO_ROOT/src/SAS/step.F90"
  printf 'PREFLIGHT 7/8 memory/disk/timeout/trap gates present\n'; ram_gate 1; disk_gate "$EVIDENCE" 1024
  grep -q 'timeout "$RUN_TIMEOUT_SEC" mpirun -np 1' "$0"
  printf 'PREFLIGHT 8/8 default is read-only; operator block not entered\n'
  if [[ -n "${PREFLIGHT_JSON:-}" ]]; then
    "$PY" - "$PREFLIGHT_JSON" <<'PY'
import json,pathlib,sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({"status":"PREFLIGHT_OK","nemo_run":False,"plants":17},indent=2)+"\n")
PY
  fi
  printf 'PREFLIGHT_OK: no NEMO build or integration was run\n'
  rm -rf -- "$scratch"
  ACTIVE_SCRATCH=""
}

fp_census() {
  local ref_bld=$1 rec_bld=$2 out=$3
  "$PY" - "$ref_bld" "$rec_bld" "$out" <<'PY'
import json,pathlib,re,subprocess,sys
ref,rec,out=map(pathlib.Path,sys.argv[1:])
cfg=(rec/'arch_nemo.fcm').read_text(errors='ignore')
if re.search(r'(^|\s)-O0(\s|$)',cfg): raise SystemExit('STATUS PLANT-FIRED: record build uses -O0')
units=('icedyn_rhg_evp','icedyn_rdgrft','icedyn')
pat=re.compile(r'^(?:v)?(?:add|sub|mul|div|sqrt|max|min|fmadd|fmsub|fnmadd|fnmsub|cvt).*(?:ss|sd|ps|pd)$|^f(?:add|sub|mul|div|sqrt)')
def one(p):
    raw=subprocess.run(['objdump','-d',str(p)],check=True,text=True,capture_output=True).stdout
    ans={}
    for line in raw.splitlines():
        cols=line.split()
        for token in cols[2:5]:
            m=token.lower()
            if pat.match(m): ans[m]=ans.get(m,0)+1; break
    return ans
report={u:{'reference':one(ref/'obj'/f'{u}.o'),'record':one(rec/'obj'/f'{u}.o')} for u in units}
bad=[u for u,v in report.items() if v['reference']!=v['record']]
if bad: raise SystemExit(f'STATUS PLANT-FIRED: FP instruction census changed: {bad}')
out.write_text(json.dumps({'status':'PASS_FP_CENSUS_REAL_FLAGS','units':report},indent=2)+"\n")
PY
}

same_program() {
  "$PY" "$GATE" --same-program "$REC_ROOT/BLD" --reference-bld "$REF_ROOT/BLD" \
    --rhg-patch "$RHG_PATCH" --rdgrft-patch "$RDG_PATCH" --dyn-patch "$DYN_PATCH" --writer "$WRITER" \
    --new-cfg "$REC_CFG" --reference-cfg "$REF_CFG" --json "$EVIDENCE/manifest/same_program.json"
  fp_census "$REF_ROOT/BLD" "$REC_ROOT/BLD" "$EVIDENCE/manifest/fp_census.json"
}

admit() {
  local commit=$1
  "$PY" "$GATE" --admit-evidence "$EVIDENCE" --expect-commit "$commit" --json "$EVIDENCE/o5_admission.json"
  local plant
  for plant in stamp binary-checksum checksum time-level selector; do
    if "$PY" "$GATE" --admit-evidence "$EVIDENCE" --expect-commit "$commit" --plant "$plant" >/dev/null 2>&1; then
      refuse "retained-evidence plant $plant did not refuse"
    fi
  done
}

if [[ "$mode" == --preflight-only ]]; then run_preflight; exit 0; fi
if [[ "$mode" == --admit-existing ]]; then
  run_preflight
  [[ -f "$EVIDENCE/manifest/producer_commit.txt" ]] || refuse "missing producer commit"
  commit=$(tr -d '[:space:]' < "$EVIDENCE/manifest/producer_commit.txt")
  admit "$commit"; printf 'ACQUISITION_ADMITTED: %s\n' "$EVIDENCE"; exit 0
fi

# ---- Operator-only --run mode ----
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] || refuse '--run requires clean committed tree'
readonly COMMIT=$(git -C "$REPO" rev-parse HEAD)
run_preflight
[[ ! -e "$REF_ROOT" && ! -e "$REC_ROOT" && ! -e "$EVIDENCE" ]] || refuse 'target config/evidence already exists'
ram_gate "$MIN_FREE_RAM_GB"; disk_gate "$EVIDENCE" "$MIN_FREE_DISK_KB"
mkdir -p "$EVIDENCE/manifest" "$EVIDENCE/input"
ACTIVE_RUN=$EVIDENCE
printf '%s\n' "$COMMIT" > "$EVIDENCE/manifest/producer_commit.txt"
sha256sum "$ORCA2_BUNDLE" | grep -q "^$ORCA2_BUNDLE_SHA256 "
sha256sum "$SAS_BUNDLE" | grep -q "^$SAS_BUNDLE_SHA256 "
tar -xzf "$ORCA2_BUNDLE" --strip-components=1 -C "$EVIDENCE/input"
tar -xzf "$SAS_BUNDLE" --strip-components=1 -C "$EVIDENCE/input"

build_target() {
  local cfg=$1 instrument=$2 root=$NEMO_ROOT/cfgs/$1
  [[ "$cfg" == "$REF_CFG" ]] && CREATED_REF=1 || CREATED_REC=1
  cd "$NEMO_ROOT"
  ./makenemo -r "$SOURCE_CFG" -n "$cfg" -m conda-scalarmath del_key 'key_xios'
  if [[ "$instrument" -eq 1 ]]; then
    cp "$NEMO_ROOT/src/ICE/icedyn_rhg_evp.F90" "$root/MY_SRC/icedyn_rhg_evp.F90"
    cp "$NEMO_ROOT/src/ICE/icedyn_rdgrft.F90" "$root/MY_SRC/icedyn_rdgrft.F90"
    cp "$NEMO_ROOT/src/ICE/icedyn.F90" "$root/MY_SRC/icedyn.F90"
    cp "$WRITER" "$root/MY_SRC/o5_si3_writer.F90"
    patch --fuzz=0 "$root/MY_SRC/icedyn_rhg_evp.F90" < "$RHG_PATCH"
    patch --fuzz=0 "$root/MY_SRC/icedyn_rdgrft.F90" < "$RDG_PATCH"
    patch --fuzz=0 "$root/MY_SRC/icedyn.F90" < "$DYN_PATCH"
    ./makenemo -n "$cfg" -m conda-scalarmath del_key 'key_xios'
  fi
  [[ -x "$root/BLD/bin/nemo.exe" ]] || refuse "$cfg lacks nemo.exe"
}

stage_run() {
  local cfg=$1 dir=$2 root=$NEMO_ROOT/cfgs/$1
  mkdir -p "$dir/o5_si3_records"
  cp "$NML" "$dir/namelist_cfg"; cp "$ICE_NML" "$dir/namelist_ice_cfg"
  cp -L "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$dir/namelist_ref"
  cp -L "$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref" "$dir/namelist_ice_ref"
  for f in "$EVIDENCE/input"/*.nc; do ln -s "$f" "$dir/$(basename "$f")"; done
  cp "$root/BLD/bin/nemo.exe" "$dir/nemo"
  (cd "$dir" && printf 'RUN_STARTED\n' > run.user.time.log && \
    timeout "$RUN_TIMEOUT_SEC" mpirun -np 1 --oversubscribe ./nemo > run.user.stdout.log 2>&1 && \
    printf 'MPIRUN_RC=0\nRUN DONE\n' >> run.user.time.log)
  touch "$dir/RUN_COMPLETE"
}

build_target "$REF_CFG" 0
build_target "$REC_CFG" 1
same_program
mkdir -p "$EVIDENCE/manifest/binaries"
cp "$REF_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.ref"
cp "$REC_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.rec"
sha256sum "$EVIDENCE/manifest/binaries/nemo.ref" "$EVIDENCE/manifest/binaries/nemo.rec" > "$EVIDENCE/manifest/binaries.sha256"
stage_run "$REF_CFG" "$EVIDENCE/reference_3step"
stage_run "$REC_CFG" "$EVIDENCE/record_3step"
(cd "$EVIDENCE/record_3step" && find o5_si3_records -type f -print0 | sort -z | xargs -0 sha256sum > records.sha256)
touch "$EVIDENCE/RUN_COMPLETE"; RUN_COMPLETE=1
admit "$COMMIT"
printf 'ACQUISITION_DONE: %s\n' "$EVIDENCE"
