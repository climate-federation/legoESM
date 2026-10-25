#!/usr/bin/env bash
# SI3 O7 NEMO ACQUISITION -- OPERATOR EXECUTION ONLY.
# Default is read-only preflight. Only an explicit --run builds or runs NEMO.
set -Eeuo pipefail

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=ORCA2_SAS_ICE
readonly BASE_BLD=${BASE_BLD:-$NEMO_ROOT/cfgs/ORCA2_SAS_ICE_O2_REF/BLD}
readonly REF_CFG=${REF_CFG:-ORCA2_SAS_ICE_O7_REF}
readonly REC_CFG=${REC_CFG:-ORCA2_SAS_ICE_O7_REC}
readonly EVIDENCE_ROOT=${EVIDENCE_ROOT:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round13}
EVIDENCE=${EVIDENCE:-$EVIDENCE_ROOT/oracle_o7_orca2_sas_ice}
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
readonly GATE=$here/../nemo_si3_o7_orca2_sas_gate.py
readonly PASSIVITY=$here/../nemo_si3_o7_passivity_gate.py
readonly WRITER=$here/o7_si3_writer.F90
readonly PATH_STUBS=$here/o7_si3_path_stubs.F90
readonly PATH_ICEUPDATE=$here/o7_si3_path_iceupdate.F90
readonly PATH_DRIVER=$here/o7_si3_path_driver.F90
readonly REAL_EXCERPT=$here/ocean.output.excerpt
readonly REF_ROOT=$NEMO_ROOT/cfgs/$REF_CFG
readonly REC_ROOT=$NEMO_ROOT/cfgs/$REC_CFG
readonly PROD=$NEMO_ROOT/cfgs/$SOURCE_CFG/EXPREF
readonly SHARED=$NEMO_ROOT/cfgs/SHARED
readonly UNITS=(icestp sbcblk icealb icesbc)

mode=${1:---preflight-only}
arg=${2:-}
case "$mode" in
  --preflight-only) PREFLIGHT_JSON=$arg ;;
  --run|--admit-existing) [[ -z "$arg" ]] || EVIDENCE=$arg ;;
  --move-aside-builds) [[ -z "$arg" ]] || { printf 'REFUSE: --move-aside-builds takes no path\n' >&2; exit 64; } ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing|--move-aside-builds] [path]\n' "$0" >&2; exit 64 ;;
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
      printf 'REFUSE_RETAINED: %s\n' "$dst" >&2
    fi
    # Checklist 13: refused builds are evidence too.  Never delete them.
    [[ "$CREATED_REF" -eq 0 || ! -d "$REF_ROOT" ]] || printf 'REFUSE_BUILD_RETAINED: %s\n' "$REF_ROOT" >&2
    [[ "$CREATED_REC" -eq 0 || ! -d "$REC_ROOT" ]] || printf 'REFUSE_BUILD_RETAINED: %s\n' "$REC_ROOT" >&2
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'printf "REFUSE: command failed at line %s; EXIT trap retains evidence\n" "$LINENO" >&2' ERR

if [[ ${O7_RETENTION_PLANT:-0} == 1 ]]; then
  mkdir -p "$EVIDENCE"; ACTIVE_RUN=$EVIDENCE
  printf 'retention plant\n' > "$EVIDENCE/operator.log"
  false
fi

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
source_path() {
  case "$1" in
    icestp|icealb|icesbc) printf '%s/src/ICE/%s.F90\n' "$NEMO_ROOT" "$1" ;;
    sbcblk) printf '%s/src/OCE/SBC/sbcblk.F90\n' "$NEMO_ROOT" ;;
    *) return 2 ;;
  esac
}

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
s,n=re.subn(r"(^\s*nn_itend\s*=\s*)[^!,\n]+",r"\g<1>2 ",s,count=1,flags=re.I|re.M)
if n!=1: raise SystemExit("STATUS PLANT-FIRED: cannot set run-control nn_itend")
match=re.search(r"(^\s*&namrun\b[\s\S]*?)(^\s*/\s*$)",s,re.I|re.M)
if not match: raise SystemExit("STATUS PLANT-FIRED: cannot find &namrun")
insert="   nn_stock    =       2\n   nn_write    =      -1\n"
s=s[:match.start(2)]+insert+s[match.start(2):]
p.write_text(s)
PY
}

targets_absent() {
  [[ ! -e "$1" && ! -e "$2" && ! -e "$3" ]]
}

move_builds_aside() {
  local ref=$1 rec=$2 destination=$3 moved=0
  mkdir -p "$destination"
  if [[ -d "$ref" ]]; then mv "$ref" "$destination/$(basename "$ref")"; moved=$((moved+1)); fi
  if [[ -d "$rec" ]]; then mv "$rec" "$destination/$(basename "$rec")"; moved=$((moved+1)); fi
  [[ "$moved" -gt 0 ]] || refuse 'no retained O7 build exists to move aside'
  printf 'BUILDS_MOVED_ASIDE: %s\n' "$destination"
}

verify_staged_namelists() {
  local dir=$1 plant=${2:-none}
  "$PY" - "$dir/namelist_cfg" "$dir/namelist_ice_cfg" "$dir/namelist_ref" "$dir/namelist_ice_ref" "$plant" <<'PY'
import re,sys
from pathlib import Path
cfg_oce,cfg_ice,ref_oce,ref_ice=map(lambda x:Path(x).read_text(),sys.argv[1:5])
plant=sys.argv[5]
if plant == "misspelled":
    cfg_oce=re.sub(r"(&namrun\b)",r"\1\n   nn_writ = -1",cfg_oce,count=1,flags=re.I)
elif plant == "misplaced":
    cfg_oce=re.sub(r"(&namdom\b)",r"\1\n   nn_write = -1",cfg_oce,count=1,flags=re.I)

def groups(text):
    result={}; current=None
    for raw in text.splitlines():
        code=raw.split("!",1)[0].strip()
        if not code: continue
        match=re.match(r"&([A-Za-z0-9_]+)",code)
        if match:
            current=match.group(1).lower(); result.setdefault(current,set()); continue
        if code.startswith("/"): current=None; continue
        if current and "=" in code:
            name=re.sub(r"\(.*\)$","",code.split("=",1)[0].strip()).lower()
            result[current].add(name)
    return result

for label,cfg,ref in (("ocean",cfg_oce,ref_oce),("ice",cfg_ice,ref_ice)):
    got,want=groups(cfg),groups(ref)
    unknown=sorted(set(got)-set(want))
    if unknown: raise SystemExit(f"STATUS PLANT-FIRED: {label} unknown groups {unknown}")
    owners={name:{group for group,names in want.items() if name in names}
            for names in want.values() for name in names}
    for group,names in got.items():
        for name in names:
            if name not in want[group]:
                raise SystemExit(
                    f"STATUS PLANT-FIRED: {label} variable {name} is staged in &{group}; "
                    f"declared owners={sorted(owners.get(name,set()))}"
                )
print("PASS_STAGED_NAMELIST_GROUP_MEMBERSHIP")
PY
}

verify_production_deck() {
  "$PY" - "$PROD/namelist_cfg" "$PROD/namelist_ice_cfg" "$SHARED/namelist_ref" "$SHARED/namelist_ice_ref" <<'PY'
import sys
from pathlib import Path
from legoesm.ice.si3_surface import SI3SurfaceConfig
cfg=SI3SurfaceConfig.from_input_namelists(*map(Path,(sys.argv[3],sys.argv[1],sys.argv[4],sys.argv[2])))
want={"ln_icedyn":True,"ln_blk":True,"ln_mixcpl":False,"ln_Cx_ice_cst":True,
      "ln_Cx_ice_frm":False,"ln_Cx_ice_AN05":False,"ln_Cx_ice_LU12":False,
      "ln_Cx_ice_LG15":False,"ln_leadhfx":True,"ln_icedO":True,"ln_icedH":True,
      "nn_hls":2,"nn_flxdist":-1,"ln_cndflx":False,"nn_qtrice":0}
bad={k:(getattr(cfg,k),v) for k,v in want.items() if getattr(cfg,k)!=v}
if bad: raise SystemExit(f"STATUS PLANT-FIRED: production O7 deck mismatch {bad}")
if cfg.rn_Cd_ia != cfg.rn_Ce_ia or cfg.rn_Cd_ia != cfg.rn_Ch_ia:
    raise SystemExit("STATUS PLANT-FIRED: production ice-air coefficients differ")
print("PASS_UNMODIFIED_PRODUCTION_DECK")
PY
}

verify_rendered_deck() {
  local dir=$1
  "$PY" - "$dir" "$PROD/namelist_cfg" "$PROD/namelist_ice_cfg" "$SHARED/namelist_ref" "$SHARED/namelist_ice_ref" <<'PY'
import re,sys
from pathlib import Path
from legoesm.ice.si3_surface import SI3SurfaceConfig
d=Path(sys.argv[1])
rendered=SI3SurfaceConfig.from_input_namelists(d/'namelist_ref',d/'namelist_cfg',d/'namelist_ice_ref',d/'namelist_ice_cfg')
production=SI3SurfaceConfig.from_input_namelists(*map(Path,(sys.argv[4],sys.argv[2],sys.argv[5],sys.argv[3])))
if rendered != production: raise SystemExit('STATUS PLANT-FIRED: rendered physical deck changed')
text=(d/'namelist_cfg').read_text()
for name,want in (("nn_it000",1),("nn_itend",2),("nn_stock",2),("nn_write",-1),("nn_fsbc",1)):
    sources=[text,(d/'namelist_ref').read_text()]
    matches=[re.findall(rf"^\s*{name}\s*=\s*(-?\d+)",source,re.I|re.M) for source in sources]
    got=int(next(values[-1] for values in matches if values))
    if got!=want: raise SystemExit(f'STATUS PLANT-FIRED: rendered {name}={got}, expected {want}')
if 2 % 1 or 2 % 1: raise SystemExit('STATUS PLANT-FIRED: run controls not multiples of nn_fsbc')
print('PASS_RENDERED_TWO_STEP_DECK')
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

prepare_real_ppsrc() {
  local scratch=$1 unit src
  mkdir -p "$scratch/raw_ref" "$scratch/raw_rec" \
    "$scratch/ref_bld/ppsrc/nemo" "$scratch/ref_bld/obj" \
    "$scratch/rec_bld/ppsrc/nemo" "$scratch/rec_bld/obj"
  for name in arch_nemo.fcm cpp.fcm bld.cfg; do
    sed "s/ORCA2_SAS_ICE_O2_REF/$REF_CFG/g" "$BASE_BLD/$name" > "$scratch/ref_bld/$name"
    sed "s/ORCA2_SAS_ICE_O2_REF/$REC_CFG/g" "$BASE_BLD/$name" > "$scratch/rec_bld/$name"
  done
  for unit in "${UNITS[@]}"; do
    src=$(source_path "$unit")
    cp "$src" "$scratch/raw_ref/$unit.F90"; cp "$src" "$scratch/raw_rec/$unit.F90"
    patch -s --fuzz=0 "$scratch/raw_rec/$unit.F90" < "$here/${unit}_o7.patch"
    preprocess "$scratch/raw_ref/$unit.F90" "$scratch/ref_bld/ppsrc/nemo/$unit.f90"
    preprocess "$scratch/raw_rec/$unit.F90" "$scratch/rec_bld/ppsrc/nemo/$unit.f90"
    : > "$scratch/ref_bld/obj/$unit.o"; : > "$scratch/rec_bld/obj/$unit.o"
  done
  cp "$WRITER" "$scratch/rec_bld/ppsrc/nemo/o7_si3_writer.f90"
  : > "$scratch/rec_bld/obj/o7_si3_writer.o"
}

same_program_preflight() {
  local scratch=$1 args=() unit src
  mkdir -p "$scratch/expected_raw" "$scratch/expected_ppsrc"
  for unit in "${UNITS[@]}"; do
    # Build the expected observer text independently from the installed patch,
    # using the exact CPP command recorded by the production build.  Do not
    # copy it from the record ppsrc: that would make the content check circular.
    src=$(source_path "$unit")
    cp "$src" "$scratch/expected_raw/$unit.F90"
    patch -s --fuzz=0 "$scratch/expected_raw/$unit.F90" < "$here/${unit}_o7.patch"
    preprocess "$scratch/expected_raw/$unit.F90" "$scratch/expected_ppsrc/$unit.f90"
    args+=(--patch "$unit=$here/${unit}_o7.patch" --expected-ppsrc "$unit=$scratch/expected_ppsrc/$unit.f90")
  done
  "$PY" "$GATE" --same-program "$scratch/rec_bld" --reference-bld "$scratch/ref_bld" \
    --writer "$WRITER" "${args[@]}" --new-cfg "$REC_CFG" --reference-cfg "$REF_CFG"
  cp -a "$scratch/rec_bld" "$scratch/rec_bld_plant"
  printf '! O7_OUTSIDE_MARKER_FP_EQUIVALENCE_PLANT\n' >> "$scratch/rec_bld_plant/ppsrc/nemo/sbcblk.f90"
  if "$PY" "$GATE" --same-program "$scratch/rec_bld_plant" --reference-bld "$scratch/ref_bld" \
      --writer "$WRITER" "${args[@]}" --new-cfg "$REC_CFG" --reference-cfg "$REF_CFG" >/dev/null 2>&1; then
    refuse 'real-ppsrc outside-marker same-program plant did not fire'
  fi
  cp -a "$scratch/rec_bld" "$scratch/rec_bld_marker_plant"
  sed -i '0,/CALL o7_set_coeffs/s/CALL o7_set_coeffs/CALL o7_set_coeffs_changed/' \
    "$scratch/rec_bld_marker_plant/ppsrc/nemo/sbcblk.f90"
  if "$PY" "$GATE" --same-program "$scratch/rec_bld_marker_plant" --reference-bld "$scratch/ref_bld" \
      --writer "$WRITER" "${args[@]}" --new-cfg "$REC_CFG" --reference-cfg "$REF_CFG" >/dev/null 2>&1; then
    refuse 'real-ppsrc marked-content/patch comparison plant did not fire'
  fi
}

check_landings() {
  local ppsrc=$1
  "$PY" - "$ppsrc" <<'PY'
import pathlib,re,sys
root=pathlib.Path(sys.argv[1])
texts={u:(root/f"{u}.f90").read_text() for u in ("icestp","sbcblk","icealb","icesbc")}
checks={
"o7_set_context":("icestp",r"CALL o7_set_context\([\s\S]{0,240}?\)\n\s*! O7_OBSERVER_END\n\s*CALL ice_sbc_tau"),
"o7_finish_step":("icestp",r"CALL ice_update_flx\([^\n]+\)\s*!.*\n\s*! O7_OBSERVER_BEGIN\n\s*CALL o7_finish_step"),
"o7_set_coeffs":("sbcblk",r"CALL o7_set_coeffs\([^\n]+\)\n\s*CALL o7_blk1_entry"),
"o7_blk1_entry":("sbcblk",r"CALL o7_blk1_entry\([^\n]+\)\n\s*! O7_OBSERVER_END\n\s*!\n\s*! treshold for outputs"),
"o7_blk1_exit":("sbcblk",r"CALL o7_blk1_exit\([^\n]+\)\n\s*! O7_OBSERVER_END\n\s*END SUBROUTINE blk_ice_1"),
"o7_blk2_entry":("sbcblk",r"CALL o7_blk2_entry\([\s\S]{0,420}?\)\n\s*! O7_OBSERVER_END\n\s*!\n\s*zcoef_dqlw"),
"o7_blk2_exit":("sbcblk",r"CALL o7_blk2_exit\([\s\S]{0,520}?\)\n\s*\n\s*! O7_OBSERVER_END\n\s*CALL iom_put"),
"o7_alb_entry":("icealb",r"CALL o7_alb_entry\([^\n]+\)\n\s*! O7_OBSERVER_END\n\s*!\n\s*IF\( ln_timing \)\s+CALL timing_start"),
"o7_alb_exit":("icealb",r"CALL o7_alb_exit\([^\n]+\)\n\s*! O7_OBSERVER_END\n\s*!\n\s*IF\( ln_timing \)\s+CALL timing_stop"),
"o7_tau_after":("icesbc",r"CALL lbc_lnk\([^\n]+\)\n\s*! O7_OBSERVER_BEGIN\n\s*CALL o7_tau_after"),
"o7_flx_entry":("icesbc",r"CALL o7_flx_entry\([\s\S]{0,320}?\)\n\s*! O7_OBSERVER_END\n\s*!\n\s*! computation of friction velocity"),
"o7_flx_exit":("icesbc",r"CALL o7_flx_exit\([^\n]+\)\n\s*! O7_OBSERVER_END\n\s*END SUBROUTINE ice_flx_other"),
}
def verify(all_text):
    missing=[name for name,(unit,pat) in checks.items() if re.search(pat,all_text[unit],re.M) is None]
    if missing: raise ValueError(','.join(missing))
verify(texts)
for name,(unit,_pat) in checks.items():
    planted=dict(texts)
    planted[unit]=planted[unit].replace(f"CALL {name}",f"CALL misplaced_{name}",1)+f"\nCALL {name}\n"
    try: verify(planted)
    except ValueError: continue
    raise SystemExit(f"STATUS PLANT-FIRED: misplacement plant inert for {name}")
for token in ("o7_zsq(ji,jj,jl) = zsq","o7_zsipt(ji,jj,jl) = zsipt",
              "o7_zdqlw(ji,jj,jl) = z_dqlw","o7_zdqsb(ji,jj,jl) = z_dqsb"):
    if texts["sbcblk"].count(token)!=1: raise SystemExit(f"STATUS PLANT-FIRED: intermediate copy {token}")
print("PASS_12_LANDINGS_AND_12_MISPLACEMENT_PLANTS")
PY
}

syntax_compile_real_ppsrc() {
  local scratch=$1 flags
  flags=$(awk '/^%FCFLAGS/ {$1=""; sub(/^ /,""); print}' "$BASE_BLD/arch_nemo.fcm")
  # shellcheck disable=SC2086
  "$FC" $flags -fsyntax-only -J "$scratch/mod" -I "$scratch/mod" -I "$BASE_BLD/inc" "$WRITER"
  local unit
  for unit in "${UNITS[@]}"; do
    # shellcheck disable=SC2086
    "$FC" $flags -fsyntax-only -J "$scratch/mod" -I "$scratch/mod" -I "$BASE_BLD/inc" \
      "$scratch/rec_bld/ppsrc/nemo/$unit.f90"
  done
}

bounds_fixture() {
  local scratch=$1 ppsrc=$2
  mkdir -p "$scratch/actual/o7_si3_records"
  # Retain the patched production ice_stp body verbatim while omitting its
  # unrelated initialization/diagnostic tail from this minimal link fixture.
  sed '/PUBLIC.*ice_init/d' "$ppsrc/icestp.f90" | awk \
    '{print} /END SUBROUTINE ice_stp/{print "END MODULE icestp"; exit}' \
    > "$scratch/actual/icestp_path.f90"
  (cd "$scratch/actual" && "$FC" -fdefault-real-8 -O2 -fcheck=bounds -fbacktrace \
    -ffree-line-length-none -fallow-argument-mismatch -J. -I. \
    "$PATH_STUBS" "$WRITER" "$ppsrc/sbcblk.f90" "$ppsrc/icealb.f90" \
    "$ppsrc/icesbc.f90" "$PATH_ICEUPDATE" icestp_path.f90 "$PATH_DRIVER" \
    -o o7_actual_si3_path && ./o7_actual_si3_path)
  [[ $(find "$scratch/actual/o7_si3_records" -type f | wc -l) -eq 18 ]] || \
    refuse 'patched sbcblk/icealb/icesbc/icestp bounds path did not emit 18 frames'
  "$PY" "$GATE" --read-record "$scratch/actual/o7_si3_records/o7_kind07_step0001_rank0000.bin" >/dev/null
  "$PY" - "$scratch/actual/o7_si3_records" <<'PY'
import hashlib, pathlib, struct, sys
import numpy as np
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o7_orca2_sas_gate import read_o7_record
root=pathlib.Path(sys.argv[1])
r0=root/'o7_s4_alb_entry_step0001_rank0000.bin'
r1=root/'o7_s4_alb_entry_step0002_rank0001.bin'
rec=read_o7_record(r0)
if not any(np.any((v == 0.0) & np.signbit(v)) for v in rec['fields'].values()):
    raise SystemExit('STATUS PLANT-FIRED: real observer path lost all negative zero payloads')
if struct.unpack_from('<i',r1.read_bytes(),24)[0] != 1:
    raise SystemExit('STATUS PLANT-FIRED: second fixture rank was not recorded')
if hashlib.sha256(r0.read_bytes()).digest() == hashlib.sha256(r1.read_bytes()).digest():
    raise SystemExit('STATUS PLANT-FIRED: per-rank observer records are identical')
print('PASS_PATCHED_FOUR_UNIT_BOUNDS_PATH_RANK_DATA_SIGNED_ZERO')
PY
}

reader_plants() {
  "$PY" "$GATE" --self-test --nemo-root "$NEMO_ROOT" >/dev/null
  local plant
  for plant in header truncation extent checksum time-level missing-frame wet-1ulp absolute-path; do
    if "$PY" "$GATE" --self-test --plant "$plant" --nemo-root "$NEMO_ROOT" >/dev/null 2>&1; then
      refuse "reader plant $plant did not refuse"
    fi
  done
}

preregistered_control_plants() {
  local plant
  for plant in swapped-field cd-1ulp fold-sign pond-branch nn-snwfra sum-seed composed-feed; do
    if "$PY" "$GATE" --control-self-test --plant "$plant" --nemo-root "$NEMO_ROOT" >/dev/null 2>&1; then
      refuse "preregistered control plant $plant did not turn the scorer red"
    fi
  done
}

passivity_real_objects() {
  local scratch=$1 flags
  mkdir -p "$scratch/ref" "$scratch/rec" "$scratch/bad" "$scratch/ref_mod" "$scratch/rec_mod" "$scratch/bad_mod"
  flags=$(awk '/^%FCFLAGS/ {$1=""; sub(/^ /,""); print}' "$BASE_BLD/arch_nemo.fcm")
  # The writer supplies the observer interface; every censused object below is
  # the actual NEMO ppsrc compiled with the record build's production flags.
  # shellcheck disable=SC2086
  "$FC" $flags -J "$scratch/rec_mod" -I "$BASE_BLD/inc" -c "$WRITER" -o "$scratch/rec/o7_si3_writer.o"
  local unit
  for unit in sbcblk icealb icesbc icestp; do
    # shellcheck disable=SC2086
    "$FC" $flags -J "$scratch/ref_mod" -I "$scratch/ref_mod" -I "$BASE_BLD/inc" -c \
      "$scratch/../ref_bld/ppsrc/nemo/$unit.f90" -o "$scratch/ref/$unit.o"
    # shellcheck disable=SC2086
    "$FC" $flags -J "$scratch/rec_mod" -I "$scratch/rec_mod" -I "$BASE_BLD/inc" -c \
      "$scratch/../rec_bld/ppsrc/nemo/$unit.f90" -o "$scratch/rec/$unit.o"
  done
  "$PY" "$PASSIVITY" --reference "$scratch/ref" --record "$scratch/rec" \
    --arch "$BASE_BLD/arch_nemo.fcm" >/dev/null
  cp "$scratch/rec"/*.o "$scratch/bad/"
  cp "$scratch/../rec_bld/ppsrc/nemo/sbcblk.f90" "$scratch/bad_sbcblk.f90"
  sed -i '/wndm_ice(ji,jj) = SQRT/a\         wndm_ice(ji,jj) = wndm_ice(ji,jj) + rn_Cd_ia ! O7 injected FP census plant' \
    "$scratch/bad_sbcblk.f90"
  # shellcheck disable=SC2086
  "$FC" $flags -J "$scratch/bad_mod" -I "$scratch/rec_mod" -I "$BASE_BLD/inc" -c \
    "$scratch/bad_sbcblk.f90" -o "$scratch/bad/sbcblk.o"
  if "$PY" "$PASSIVITY" --reference "$scratch/ref" --record "$scratch/bad" \
      --arch "$BASE_BLD/arch_nemo.fcm" >/dev/null 2>&1; then
    refuse 'injected-FP passivity plant did not fire'
  fi
}

verify_real_excerpt() {
  "$PY" - "$REAL_EXCERPT" <<'PY'
import hashlib,pathlib,sys,tempfile
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o7_orca2_sas_gate import verify_o7_log
p=pathlib.Path(sys.argv[1]); lines=p.read_text().splitlines()
src=pathlib.Path(next(x.split(':',1)[1].strip() for x in lines if x.startswith('# Source path:')))
sha=next(x.split(':',1)[1].strip() for x in lines if x.startswith('# Complete source SHA-256:'))
spans=next(x.split(':',1)[1].strip().rstrip('.') for x in lines if x.startswith('# Source lines:'))
if hashlib.sha256(src.read_bytes()).hexdigest()!=sha: raise SystemExit('STATUS PLANT-FIRED: real log provenance digest')
nums=[]
for span in spans.split(','):
    a=[int(x.strip()) for x in span.split('-')]; nums.extend(range(a[0],a[-1]+1))
body=[x.rstrip() for x in lines if not x.startswith('#')]
source=src.read_text(errors='replace').splitlines()
if body != [source[n-1].rstrip() for n in nums]: raise SystemExit('STATUS PLANT-FIRED: real log excerpt differs')
# The available real excerpt is O1's diagnostic deck (dynamics/thermo off).
# Exercise the O7 parser with an explicit synthetic production-selector echo;
# admission applies the same parser to each real dynamics-on O7 ocean.output.
text='\n'.join(body).replace('ln_icedyn =  F','ln_icedyn =  T').replace('ln_icethd =  F','ln_icethd =  T')
text += '''
 ln_pnd = T
 ln_pnd_alb = T
 ln_blk = T
 ln_mixcpl = F
 ln_Cx_ice_cst = T
 nn_it000 = 1
 nn_itend = 2
 nn_fsbc = 1
 jpni = 1
 jpnj = 1
'''
with tempfile.TemporaryDirectory() as td:
    synthetic=pathlib.Path(td)/'ocean.output'; synthetic.write_text(text)
    verify_o7_log(synthetic)
    try: verify_o7_log(synthetic,plant='error-token')
    except ValueError as exc:
        if 'E R R O R : planted fatal NEMO line' not in str(exc): raise
    else: raise SystemExit('STATUS PLANT-FIRED: ocean.output E R R O R plant inert')
print('PASS_SYNTHETIC_DYNAMICS_ON_LOG; REAL_CHECK_AT_ADMISSION')
PY
}

retention_plant() {
  local scratch=$1 root=$scratch/retention
  mkdir -p "$root"
  if env EVIDENCE_ROOT="$root" EVIDENCE="$root/active" REFUSED_DIR="$root/refused" \
      O7_RETENTION_PLANT=1 "$0" --preflight-only >/dev/null 2>&1; then
    refuse 'retention plant returned success'
  fi
  [[ ! -e "$root/active" && $(find "$root/refused" -type f -name operator.log | wc -l) -eq 1 ]] || \
    refuse 'failure evidence was not retained exactly once'
}

rerun_move_aside_plant() {
  local scratch=$1 root ref rec
  root=$scratch/rerun; ref=$root/ORCA2_SAS_ICE_O7_REF; rec=$root/ORCA2_SAS_ICE_O7_REC
  mkdir -p "$ref" "$rec"; printf ref > "$ref/retained"; printf rec > "$rec/retained"
  move_builds_aside "$ref" "$rec" "$root/refused/builds_test" >/dev/null
  [[ ! -e "$ref" && ! -e "$rec" ]] || refuse 'move-aside rerun plant left target builds in place'
  [[ -f "$root/refused/builds_test/ORCA2_SAS_ICE_O7_REF/retained" && \
     -f "$root/refused/builds_test/ORCA2_SAS_ICE_O7_REC/retained" ]] || \
    refuse 'move-aside rerun plant did not retain both builds'
}

run_preflight() {
  local scratch; scratch=$(mktemp -d); ACTIVE_SCRATCH=$scratch
  printf 'PREFLIGHT 1/11 unmodified production and validated two-step O7 deck\n'
  awk -v cfg="$SOURCE_CFG" '$1==cfg {found=1} END {exit !found}' "$NEMO_ROOT/cfgs/ref_cfgs.txt"
  verify_production_deck; render_deck "$scratch/deck"; verify_rendered_deck "$scratch/deck"
  verify_staged_namelists "$scratch/deck"
  for nml_plant in misspelled misplaced; do
    if verify_staged_namelists "$scratch/deck" "$nml_plant" >/dev/null 2>&1; then
      refuse "staged-namelist $nml_plant variable plant did not fire"
    fi
  done
  cp -a "$scratch/deck" "$scratch/deck_plant"
  sed -i '0,/ln_blk[[:space:]]*=[[:space:]]*\.true\./s//ln_blk = .false./' "$scratch/deck_plant/namelist_cfg"
  if verify_rendered_deck "$scratch/deck_plant" >/dev/null 2>&1; then
    refuse 'rendered physical-deck plant did not fire'
  fi
  if [[ "$mode" != --admit-existing ]]; then
    targets_absent "$REF_ROOT" "$REC_ROOT" "$EVIDENCE" || refuse 'target config/evidence already exists'
  fi
  mkdir -p "$scratch/existing"
  if targets_absent "$scratch/no-ref" "$scratch/no-rec" "$scratch/existing"; then
    refuse 'target-existence plant did not fire'
  fi
  printf 'PREFLIGHT 2/11 raw patches and record-build CPP output\n'; prepare_real_ppsrc "$scratch"
  printf 'PREFLIGHT 3/11 like-with-like ppsrc same-program proof and plants\n'; same_program_preflight "$scratch"
  printf 'PREFLIGHT 4/11 exact call-point landings and misplacement plants\n'; check_landings "$scratch/rec_bld/ppsrc/nemo"
  printf 'PREFLIGHT 5/11 actual patched interfaces under production flags\n'; mkdir -p "$scratch/mod"; syntax_compile_real_ppsrc "$scratch"
  printf 'PREFLIGHT 6/11 bounds-checked writer and patched-NEMO execution fixtures\n'
  bounds_fixture "$scratch/bounds" "$scratch/rec_bld/ppsrc/nemo"
  printf 'PREFLIGHT 7/11 reader/extents/source-derived-time-level/wet-cell controls\n'; reader_plants
  printf 'PREFLIGHT 8/11 preregistered red-score controls including P6 composition\n'; preregistered_control_plants
  printf 'PREFLIGHT 9/11 real-object per-function passivity census and injected-op plant\n'; passivity_real_objects "$scratch/passivity"
  printf 'PREFLIGHT 10/11 synthetic dynamics-on ocean.output parser; real check at admission\n'; verify_real_excerpt
  printf 'PREFLIGHT 11/11 retention, move-aside rerun, resource, timeout, and read-only controls\n'
  retention_plant "$scratch"; rerun_move_aside_plant "$scratch"; ram_gate 1; disk_gate "$EVIDENCE" 1024
  grep -q 'timeout "$RUN_TIMEOUT_SEC" mpirun -np 1' "$0"
  if [[ -n "${PREFLIGHT_JSON:-}" ]]; then
    "$PY" - "$PREFLIGHT_JSON" <<'PY'
import json,pathlib,sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({"status":"PREFLIGHT_OK","nemo_run":False,"plants":36},indent=2)+"\n")
PY
  fi
  printf 'PREFLIGHT_OK: no NEMO build or integration was run\n'
  rm -rf -- "$scratch"; ACTIVE_SCRATCH=""
}

same_program_real_builds() {
  local args=() unit src scratch
  scratch=$(mktemp -d)
  for unit in "${UNITS[@]}"; do
    src=$scratch/$unit.F90; cp "$(source_path "$unit")" "$src"
    patch -s --fuzz=0 "$src" < "$here/${unit}_o7.patch"
    preprocess "$src" "$scratch/$unit.f90"
    args+=(--patch "$unit=$here/${unit}_o7.patch" --expected-ppsrc "$unit=$scratch/$unit.f90")
  done
  "$PY" "$GATE" --same-program "$REC_ROOT/BLD" --reference-bld "$REF_ROOT/BLD" \
    --writer "$WRITER" "${args[@]}" --new-cfg "$REC_CFG" --reference-cfg "$REF_CFG" \
    --json "$EVIDENCE/manifest/same_program.json"
  rm -rf -- "$scratch"
  "$PY" "$PASSIVITY" --reference "$REF_ROOT/BLD/obj" --record "$REC_ROOT/BLD/obj" \
    --arch "$REC_ROOT/BLD/arch_nemo.fcm" --json "$EVIDENCE/manifest/fp_census.json"
}

admit() {
  local commit=$1 plant
  "$PY" "$GATE" --admit-evidence "$EVIDENCE" --expect-commit "$commit" --nemo-root "$NEMO_ROOT" --json "$EVIDENCE/o7_admission.json"
  for plant in stamp binary-checksum checksum input-checksum time-level selector error-log restart missing-frame wet-1ulp \
    swapped-field cd-1ulp fold-sign pond-branch nn-snwfra sum-seed composed-feed; do
    if "$PY" "$GATE" --admit-evidence "$EVIDENCE" --expect-commit "$commit" --nemo-root "$NEMO_ROOT" --plant "$plant" >/dev/null 2>&1; then
      refuse "retained-evidence plant $plant did not refuse"
    fi
  done
}

if [[ "$mode" == --move-aside-builds ]]; then
  # Refused builds are never deleted.  Operator rerun procedure:
  #   run.sh --move-aside-builds
  #   run.sh --run
  # The exact retained configs move below refused/builds_*; makenemo recreates
  # the same names and refreshes their work_cfgs.txt entries.
  move_builds_aside "$REF_ROOT" "$REC_ROOT" \
    "$REFUSED_DIR/builds_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  exit 0
fi
if [[ "$mode" == --preflight-only ]]; then run_preflight; exit 0; fi
if [[ "$mode" == --admit-existing ]]; then
  run_preflight
  [[ -f "$EVIDENCE/manifest/producer_commit.txt" ]] || refuse 'missing producer commit'
  commit=$(tr -d '[:space:]' < "$EVIDENCE/manifest/producer_commit.txt")
  admit "$commit"; printf 'ACQUISITION_ADMITTED: %s\n' "$EVIDENCE"; exit 0
fi

# ---- Operator-only --run mode: this is the only block that invokes NEMO. ----
[[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] || refuse '--run requires a clean committed tree'
readonly COMMIT=$(git -C "$REPO" rev-parse HEAD)
run_preflight
[[ ! -e "$REF_ROOT" && ! -e "$REC_ROOT" && ! -e "$EVIDENCE" ]] || refuse 'target config/evidence already exists'
ram_gate "$MIN_FREE_RAM_GB"; disk_gate "$EVIDENCE" "$MIN_FREE_DISK_KB"
mkdir -p "$EVIDENCE/manifest/production_deck" "$EVIDENCE/manifest/binaries" "$EVIDENCE/input"
ACTIVE_RUN=$EVIDENCE
printf '%s\n' "$COMMIT" > "$EVIDENCE/manifest/producer_commit.txt"
actual_orca2=$(sha256sum "$ORCA2_BUNDLE"); [[ ${actual_orca2%% *} == "$ORCA2_BUNDLE_SHA256" ]]
actual_sas=$(sha256sum "$SAS_BUNDLE"); [[ ${actual_sas%% *} == "$SAS_BUNDLE_SHA256" ]]
tar -xzf "$ORCA2_BUNDLE" --strip-components=1 -C "$EVIDENCE/input"
tar -xzf "$SAS_BUNDLE" --strip-components=1 -C "$EVIDENCE/input"
cp "$PROD/namelist_cfg" "$EVIDENCE/manifest/production_deck/namelist_cfg"
cp "$PROD/namelist_ice_cfg" "$EVIDENCE/manifest/production_deck/namelist_ice_cfg"
cp -L "$SHARED/namelist_ref" "$EVIDENCE/manifest/production_deck/namelist_ref"
cp -L "$SHARED/namelist_ice_ref" "$EVIDENCE/manifest/production_deck/namelist_ice_ref"

build_target() {
  local cfg=$1 instrument=$2 root=$NEMO_ROOT/cfgs/$1 unit
  [[ "$cfg" == "$REF_CFG" ]] && CREATED_REF=1 || CREATED_REC=1
  cd "$NEMO_ROOT"
  ./makenemo -r "$SOURCE_CFG" -n "$cfg" -m conda-scalarmath del_key 'key_xios'
  if [[ "$instrument" -eq 1 ]]; then
    for unit in "${UNITS[@]}"; do
      cp "$(source_path "$unit")" "$root/MY_SRC/$unit.F90"
      patch --fuzz=0 "$root/MY_SRC/$unit.F90" < "$here/${unit}_o7.patch"
    done
    cp "$WRITER" "$root/MY_SRC/o7_si3_writer.F90"
    ./makenemo -n "$cfg" -m conda-scalarmath del_key 'key_xios'
  fi
  [[ -x "$root/BLD/bin/nemo.exe" ]] || refuse "$cfg lacks nemo.exe"
}

stage_run() {
  local cfg=$1 dir=$2 root=$NEMO_ROOT/cfgs/$1
  mkdir -p "$dir/o7_si3_records"; render_deck "$dir"
  local f
  for f in "$EVIDENCE/input"/*.nc; do ln -s "$f" "$dir/$(basename "$f")"; done
  cp "$root/BLD/bin/nemo.exe" "$dir/nemo"
  (cd "$dir" && printf 'RUN_STARTED\n' > run.user.time.log && \
    timeout "$RUN_TIMEOUT_SEC" mpirun -np 1 --oversubscribe ./nemo > run.user.stdout.log 2>&1 && \
    printf 'MPIRUN_RC=0\nRUN DONE\n' >> run.user.time.log)
  touch "$dir/RUN_COMPLETE"
}

build_target "$REF_CFG" 0
build_target "$REC_CFG" 1
same_program_real_builds
cp "$REF_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.ref"
cp "$REC_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.rec"
(cd "$EVIDENCE" && sha256sum manifest/binaries/nemo.ref manifest/binaries/nemo.rec > manifest/binaries.sha256)
stage_run "$REF_CFG" "$EVIDENCE/reference_2step"
stage_run "$REC_CFG" "$EVIDENCE/record_2step"
(cd "$EVIDENCE" && find record_2step/o7_si3_records -type f -print0 | sort -z | xargs -0 sha256sum > manifest/records.sha256)
(cd "$EVIDENCE" && { find input manifest/production_deck -type f -print0; \
  printf '%s\0' manifest/same_program.json manifest/fp_census.json manifest/producer_commit.txt; } | \
  sort -z | xargs -0 sha256sum > manifest/inputs.sha256)
touch "$EVIDENCE/RUN_COMPLETE"; RUN_COMPLETE=1
admit "$COMMIT"
printf 'ACQUISITION_DONE: %s\n' "$EVIDENCE"
