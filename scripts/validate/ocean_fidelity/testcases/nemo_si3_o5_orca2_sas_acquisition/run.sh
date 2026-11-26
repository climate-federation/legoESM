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
readonly O2_REF_BLD=$NEMO_ROOT/cfgs/ORCA2_SAS_ICE_O2_REF/BLD
readonly O2_REC_BLD=$NEMO_ROOT/cfgs/ORCA2_SAS_ICE_O2_REC/BLD

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
    if [[ -n "$ACTIVE_RUN" && -d "$ACTIVE_RUN" && "$RUN_COMPLETE" -eq 0 ]]; then
      local dst="$REFUSED_DIR/$(basename "$ACTIVE_RUN")_$(date -u +%Y%m%dT%H%M%SZ)_rc${rc}_$$"
      mv "$ACTIVE_RUN" "$dst"
      printf 'REFUSE_RETained: %s\n' "$dst" >&2
    elif [[ -n "$ACTIVE_RUN" && -d "$ACTIVE_RUN" ]]; then
      printf 'REFUSE_RETAINED_IN_PLACE: completed run was not relocated: %s\n' "$ACTIVE_RUN" >&2
    fi
    [[ "$CREATED_REF" -eq 0 || ! -d "$REF_ROOT" ]] || printf 'REFUSE_RETAINED_BUILD: %s\n' "$REF_ROOT" >&2
    [[ "$CREATED_REC" -eq 0 || ! -d "$REC_ROOT" ]] || printf 'REFUSE_RETAINED_BUILD: %s\n' "$REC_ROOT" >&2
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
  local ocean_cfg=${1:-$NML} ice_cfg=${2:-$ICE_NML}
  "$PY" - "$ocean_cfg" "$ice_cfg" "$NEMO_ROOT/cfgs/$SOURCE_CFG/EXPREF/namelist_cfg" \
    "$NEMO_ROOT/cfgs/$SOURCE_CFG/EXPREF/namelist_ice_cfg" "$NEMO_ROOT/cfgs/SHARED/namelist_ref" \
    "$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref" <<'PY'
import pathlib, re, sys
paths = list(map(pathlib.Path, sys.argv[1:]))

def groups(path):
    parsed, group = {}, None
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        code = raw.split("!", 1)[0].strip()
        if not code:
            continue
        start = re.match(r"&([a-z][a-z0-9_]*)\b", code, re.I)
        if start:
            group = start.group(1).lower()
            if group in parsed:
                raise SystemExit(f"STATUS PLANT-FIRED: duplicate &{group} in {path}:{line_number}")
            parsed[group] = {}
            continue
        if group is not None and code.startswith("/"):
            group = None
            continue
        if group is None:
            continue
        assignment = re.match(
            r"([a-z][a-z0-9_]*(?:%[a-z][a-z0-9_]*)*)(?:\s*\([^=]*\))?\s*=\s*(.*)",
            code,
            re.I,
        )
        if assignment:
            parsed[group][assignment.group(1).lower()] = (
                assignment.group(2).strip().rstrip(",").lower(), line_number
            )
        elif "=" in code:
            raise SystemExit(f"STATUS PLANT-FIRED: unparsed assignment in {path}:{line_number}: {code}")
    if group is not None:
        raise SystemExit(f"STATUS PLANT-FIRED: unterminated &{group} in {path}")
    return parsed

o, i, bo, bi, ro, ri = map(groups, paths)

def validate_membership(deck, reference, path):
    for group, variables in deck.items():
        if group not in reference:
            raise SystemExit(f"STATUS PLANT-FIRED: unknown namelist group &{group} in {path}")
        for variable, (_, line_number) in variables.items():
            if variable not in reference[group]:
                raise SystemExit(
                    f"STATUS PLANT-FIRED: unknown namelist variable {variable} in &{group} "
                    f"at {path}:{line_number}"
                )

validate_membership(o, ro, paths[0])
validate_membership(i, ri, paths[1])

def merged(*layers):
    result = {}
    for layer in layers:
        for group, variables in layer.items():
            result.setdefault(group, {}).update(variables)
    return result

eo, ei = merged(ro, bo, o), merged(ri, bi, i)
want_o = {
    ("namrun", "nn_it000"): "1", ("namrun", "nn_itend"): "3",
    ("namrun", "nn_stock"): "3", ("namsbc", "nn_fsbc"): "1",
    ("namsbc", "ln_ice_embd"): ".false.",
}
want_i = {
    ("nampar", "ln_icedyn"): ".true.", ("nampar", "ln_icethd"): ".false.",
    ("namdyn", "ln_dynall"): ".true.", ("namdyn", "ln_dynrhgadv"): ".false.",
    ("namdyn_rdgrft", "ln_ridging"): ".false.",
    ("namdyn_rdgrft", "ln_rafting"): ".false.",
    ("namdyn_rdgrft", "ln_str_h79"): ".true.",
    ("namdyn_rdgrft", "ln_str_smooth"): ".true.",
    ("namdyn_rhg", "ln_aevp"): ".true.", ("namdyn_rhg", "nn_nevp"): "100",
}
for resolved, wanted in ((eo, want_o), (ei, want_i)):
    for (group, variable), value in wanted.items():
        got = resolved.get(group, {}).get(variable, (None, None))[0]
        if got != value:
            raise SystemExit(
                f"STATUS PLANT-FIRED: &{group} {variable}={got}, expected {value}"
            )

def deviations(deck, base):
    changed = set()
    for group in deck.keys() | base.keys():
        deck_vars, base_vars = deck.get(group, {}), base.get(group, {})
        for variable in deck_vars.keys() | base_vars.keys():
            if deck_vars.get(variable, (None,))[0] != base_vars.get(variable, (None,))[0]:
                changed.add((group, variable))
    return changed

allowed_o = {
    ("namrun", "nn_itend"), ("namrun", "nn_stock"), ("namrun", "nn_write"),
    ("namsbc", "ln_ice_embd"),
}
allowed_i = {
    ("nampar", "ln_icedyn"), ("nampar", "ln_icethd"),
    ("namdyn", "ln_dynall"), ("namdyn", "ln_dynrhgadv"),
    ("namdyn_rdgrft", "ln_ridging"), ("namdyn_rdgrft", "ln_rafting"),
}
for label, got, wanted in (
    ("ocean", deviations(o, bo), allowed_o), ("ice", deviations(i, bi), allowed_i)
):
    if got != wanted:
        raise SystemExit(
            f"STATUS PLANT-FIRED: {label} deck has unregistered deviations: "
            f"got={sorted(got)}, expected={sorted(wanted)}"
        )
print("PASS_NAMELIST_REF_GROUP_MEMBERSHIP_AND_RESOLVED_OVERLAY")
PY
}

check_patches_and_landings() {
  local scratch=$1
  local raw=$scratch/raw pp=$scratch/pp mods=$scratch/mods fpp_keys
  mkdir -p "$raw" "$pp" "$mods"
  cp "$NEMO_ROOT/src/ICE/icedyn_rhg_evp.F90" "$raw/icedyn_rhg_evp.F90"
  cp "$NEMO_ROOT/src/ICE/icedyn_rdgrft.F90" "$raw/icedyn_rdgrft.F90"
  cp "$NEMO_ROOT/src/ICE/icedyn.F90" "$raw/icedyn.F90"
  patch -s --fuzz=0 "$raw/icedyn_rhg_evp.F90" < "$RHG_PATCH"
  patch -s --fuzz=0 "$raw/icedyn_rdgrft.F90" < "$RDG_PATCH"
  patch -s --fuzz=0 "$raw/icedyn.F90" < "$DYN_PATCH"
  fpp_keys=$(awk '/bld::tool::fppkeys/ {for(i=2;i<=NF;i++) printf " -D%s",$i}' "$O2_REF_BLD/cpp.fcm")
  for unit in icedyn_rhg_evp icedyn_rdgrft icedyn; do
    cpp $fpp_keys -P -traditional -I "$NEMO_ROOT/cfgs/ORCA2_SAS_ICE_O2_REF/WORK" \
      "$raw/$unit.F90" -o "$pp/$unit.f90"
  done
  "$FC" -c -O3 -fdefault-real-8 -ffree-line-length-none -I "$O2_REF_BLD/inc" \
    -J "$mods" "$WRITER" -o "$scratch/o5_si3_writer.o"
  for unit in icedyn_rdgrft icedyn_rhg_evp icedyn; do
    "$FC" -fsyntax-only -fdefault-real-8 -fcray-pointer -ffree-line-length-none \
      -fallow-argument-mismatch -I "$mods" -I "$O2_REF_BLD/inc" \
      -I /home/dbalwada/miniconda3/envs/nemo-build/include -J "$mods" "$pp/$unit.f90"
  done
  sed 's/ncom_stp, nit000/ncom_stp, o5_undeclared_identifier/' "$pp/icedyn_rhg_evp.f90" > "$scratch/undefined_plant.f90"
  if "$FC" -fsyntax-only -fdefault-real-8 -fcray-pointer -ffree-line-length-none \
       -fallow-argument-mismatch -I "$mods" -I "$O2_REF_BLD/inc" -J "$mods" \
       -I /home/dbalwada/miniconda3/envs/nemo-build/include \
       "$scratch/undefined_plant.f90" >/dev/null 2>&1; then
    refuse 'patched-ppsrc syntax plant did not fail'
  fi
  "$PY" - "$pp" "$FIXTURE" <<'PY'
import pathlib, re, sys
root=pathlib.Path(sys.argv[1])
text="\n".join(p.read_text() for p in (root/'icedyn_rhg_evp.f90',root/'icedyn_rdgrft.f90',root/'icedyn.f90'))
checks={
"o5_context":r"CALL o5_context\([\s\S]{0,220}?\)\n\s*IF\( kt == nit000",
"o5_fimask":r"CALL lbc_lnk\( 'icedyn_rhg_evp', fimask[^\n]+\)\n\s*CALL o5_fimask",
"o5_strength_entry":r"CALL o5_strength_entry\([\s\S]{0,500}?\)\n[\s\S]{0,100}?CALL ice_strength",
"o5_strength_smooth":r"CALL ice_strength\n\s*CALL o5_strength_smooth",
"o5_aevp_entry_state":r"zsshdyn\(:,:\) = ice_var_sshdyn[^\n]+\n\s*CALL o5_aevp_entry_state",
"o5_frame_delta":r"CALL lbc_lnk\( 'icedyn_rhg_evp', zdelta[^\n]+\n[\s\S]{0,180}?CALL o5_frame_delta",
"o5_frame_stress":r"CALL o5_frame_stress[\s\S]{0,600}?\n\s*! --- Ice internal stresses",
"o5_frame_velocity":r"CALL lbc_lnk\( 'icedyn_rhg_evp', u_ice[^\n]+\n[\s\S]{0,100}?ENDIF\n\s*IF\([^\n]+\n[^\n]+CALL o5_frame_velocity",
"o5_aevp_exit":r"pstress12_i\(:,:\) = zs12\(:,:\)\n\s*CALL o5_aevp_exit",
"o5_strength_raw":r"CALL o5_strength_raw[^\n]+\n\s*CASE \( np_strcst \)",
"o5_rdgrft_entry":r"CALL o5_rdgrft_entry[^\n]+\n\s*CALL ice_dyn_rdgrft",
"o5_rdgrft_exit":r"CALL ice_dyn_rdgrft[^\n]+\n\s*CALL o5_rdgrft_exit",
}

def verify(value):
    missing=[name for name,pat in checks.items() if re.search(pat,value,re.M) is None]
    if missing: raise ValueError(','.join(missing))
verify(text)
patched_calls=set(re.findall(r"\bCALL\s+(o5_[a-z0-9_]+)",text,re.I))
markers={"o5_"+name.lower() for name in re.findall(r"O5_PATCHED_([A-Z0-9_]+)",pathlib.Path(sys.argv[2]).read_text())}
if patched_calls != markers:
    raise SystemExit(f"STATUS PLANT-FIRED: patched/fixture observer coverage differs: {sorted(patched_calls ^ markers)}")
for name in checks:
    planted=text.replace(f"CALL {name}", f"CALL misplaced_{name}", 1)+f"\nCALL {name}\n"
    try: verify(planted)
    except ValueError: continue
    raise SystemExit(f"STATUS PLANT-FIRED: misplacement plant inert for {name}")
print("PASS_PATCHED_PPSRC_FIXTURE_COVERAGE_AND_12_MISPLACEMENT_PLANTS")
PY
}

preflight_real_same_program() {
  local scratch=$1
  local root=$scratch/real_same_program
  local ref=$root/ORCA2_SAS_ICE_O2_REF/BLD
  local rec=$root/ORCA2_SAS_ICE_O2_REC/BLD
  local raw=$root/raw fpp_keys unit name
  mkdir -p "$ref/ppsrc/nemo" "$ref/obj" "$rec/ppsrc/nemo" "$rec/obj" "$raw"
  for name in arch_nemo.fcm cpp.fcm bld.cfg; do
    cp "$O2_REF_BLD/$name" "$ref/$name"
    cp "$O2_REC_BLD/$name" "$rec/$name"
  done
  fpp_keys=$(awk '/bld::tool::fppkeys/ {for(i=2;i<=NF;i++) printf " -D%s",$i}' "$O2_REC_BLD/cpp.fcm")
  for unit in icedyn_rhg_evp icedyn_rdgrft icedyn; do
    cp "$NEMO_ROOT/src/ICE/$unit.F90" "$raw/$unit.ref.F90"
    cp "$raw/$unit.ref.F90" "$raw/$unit.rec.F90"
  done
  patch -s --fuzz=0 "$raw/icedyn_rhg_evp.rec.F90" < "$RHG_PATCH"
  patch -s --fuzz=0 "$raw/icedyn_rdgrft.rec.F90" < "$RDG_PATCH"
  patch -s --fuzz=0 "$raw/icedyn.rec.F90" < "$DYN_PATCH"
  for unit in icedyn_rhg_evp icedyn_rdgrft icedyn; do
    cpp $fpp_keys -P -traditional -I "$O2_REC_BLD/../WORK" \
      "$raw/$unit.ref.F90" -o "$ref/ppsrc/nemo/$unit.f90"
    cpp $fpp_keys -P -traditional -I "$O2_REC_BLD/../WORK" \
      "$raw/$unit.rec.F90" -o "$rec/ppsrc/nemo/$unit.f90"
    cp "$O2_REF_BLD/obj/$unit.o" "$ref/obj/$unit.o"
    cp "$O2_REC_BLD/obj/$unit.o" "$rec/obj/$unit.o"
  done
  cp "$O2_REF_BLD/ppsrc/nemo/icevar.f90" "$ref/ppsrc/nemo/icevar.f90"
  cp "$O2_REC_BLD/ppsrc/nemo/icevar.f90" "$rec/ppsrc/nemo/icevar.f90"
  cp "$O2_REF_BLD/obj/icevar.o" "$ref/obj/icevar.o"
  cp "$O2_REF_BLD/obj/icevar.o" "$rec/obj/icevar.o"
  cpp $fpp_keys -P -traditional -I "$O2_REC_BLD/../WORK" \
    "$WRITER" -o "$rec/ppsrc/nemo/o5_si3_writer.f90"
  cp "$O2_REC_BLD/obj/o2_si3_state_writer.o" "$rec/obj/o5_si3_writer.o"
  "$PY" "$GATE" --same-program "$rec" --reference-bld "$ref" \
    --rhg-patch "$RHG_PATCH" --rdgrft-patch "$RDG_PATCH" --dyn-patch "$DYN_PATCH" \
    --writer "$WRITER" --new-cfg ORCA2_SAS_ICE_O2_REC \
    --reference-cfg ORCA2_SAS_ICE_O2_REF --json "$root/same_program.json"

  sed 's/CALL o5_fimask( fimask )/CALL o5_fimask( -fimask )/' \
    "$raw/icedyn_rhg_evp.rec.F90" > "$raw/icedyn_rhg_evp.plant.F90"
  cpp $fpp_keys -P -traditional -I "$O2_REC_BLD/../WORK" \
    "$raw/icedyn_rhg_evp.plant.F90" -o "$rec/ppsrc/nemo/icedyn_rhg_evp.f90"
  if "$PY" "$GATE" --same-program "$rec" --reference-bld "$ref" \
       --rhg-patch "$RHG_PATCH" --rdgrft-patch "$RDG_PATCH" --dyn-patch "$DYN_PATCH" \
       --writer "$WRITER" --new-cfg ORCA2_SAS_ICE_O2_REC \
       --reference-cfg ORCA2_SAS_ICE_O2_REF >/dev/null 2>&1; then
    refuse 'real-preprocessed same-program insertion plant did not fail'
  fi
  printf 'PASS_REAL_PREPROCESSED_SAME_PROGRAM_AND_INSERTION_PLANT\n'
}

compile_fixture() {
  local scratch=$1 pp=$2
  mkdir -p "$scratch/o5_si3_records"
  "$PY" - "$pp" "$FIXTURE" "$scratch/o5_patched_fixture.F90" <<'PY'
import pathlib,re,sys
pp,template,out=map(pathlib.Path,sys.argv[1:])
calls={}
for source in (pp/'icedyn_rhg_evp.f90',pp/'icedyn_rdgrft.f90',pp/'icedyn.f90'):
    lines=source.read_text().splitlines()
    index=0
    while index < len(lines):
        match=re.search(r'\bCALL\s+(o5_[a-z0-9_]+)\s*\(',lines[index],re.I)
        if not match:
            index+=1; continue
        name=match.group(1).lower()
        start=index-1 if lines[index].lstrip().startswith('&') and index > 0 else index
        block=lines[start:index+1]
        depth=sum(line.count('(')-line.count(')') for line in block)
        while depth > 0:
            index+=1
            if index >= len(lines): raise SystemExit(f'STATUS PLANT-FIRED: unterminated {name} call')
            block.append(lines[index]); depth+=lines[index].count('(')-lines[index].count(')')
        if name in calls: raise SystemExit(f'STATUS PLANT-FIRED: duplicate patched call {name}')
        calls[name]='\n'.join(block)
        index+=1
text=template.read_text()
markers=re.findall(r'^! O5_PATCHED_([A-Z0-9_]+)$',text,re.M)
expected={f'o5_{marker.lower()}' for marker in markers}
if set(calls) != expected:
    raise SystemExit(f'STATUS PLANT-FIRED: patched source/template calls differ: {sorted(set(calls)^expected)}')
for marker in markers:
    text=text.replace(f'! O5_PATCHED_{marker}',calls[f'o5_{marker.lower()}'])
if re.search(r'^! O5_PATCHED_',text,re.M): raise SystemExit('STATUS PLANT-FIRED: unexpanded patched-source marker')
out.write_text(text)
print('PASS_GENERATED_FIXTURE_USES_PATCHED_PPSRC_CALLS')
PY
  (cd "$scratch" && "$FC" -O2 -fcheck=bounds -fbacktrace -finit-real=snan -Wall -Wextra -J. -I. \
    "$STUBS" "$WRITER" "$scratch/o5_patched_fixture.F90" -o o5_fixture && ./o5_fixture)
  [[ $(find "$scratch/o5_si3_records" -type f | wc -l) -eq 164 ]] || refuse "fixture record count is not 164"
  "$PY" "$GATE" --verify-layout-fixture "$scratch/o5_si3_records"
  local plant
  for plant in rank-collapse signed-zero extent; do
    if "$PY" "$GATE" --verify-layout-fixture "$scratch/o5_si3_records" --plant "$plant" >/dev/null 2>&1; then
      refuse "layout fixture plant $plant did not refuse"
    fi
  done
}

reader_plants() {
  "$PY" "$GATE" --self-test >/dev/null
  local plant
  for plant in header truncation checksum time-level wet-1ulp swapped-field; do
    if "$PY" "$GATE" --self-test --plant "$plant" >/dev/null 2>&1; then
      refuse "reader plant $plant did not refuse"
    fi
  done
}

verify_real_excerpt() {
  "$PY" - "$REAL_EXCERPT" <<'PY'
import pathlib, sys
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o1_orca2_sas_gate import verify_ocean_output_log
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o5_orca2_sas_gate import verify_o5_log
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
try:
    verify_ocean_output_log(p, plant='error-token')
except ValueError as exc:
    message=str(exc)
    if 'ocean.output contains E R R O R' not in message or 'planted fatal error in ocean.output' not in message:
        raise SystemExit('STATUS PLANT-FIRED: ocean.output error refusal did not quote NEMO error lines')
else:
    raise SystemExit('STATUS PLANT-FIRED: ocean.output E R R O R plant did not refuse')
# This provenance-bearing historical excerpt predates O5 and has dynamics off.
# Exercise O5's overlay parser with an explicitly synthetic dynamics-on variant;
# admission always checks the complete, real record-run ocean.output itself.
synthetic='\n'.join(body).replace('ln_icedyn =  F', 'ln_icedyn =  T')
synthetic += '''
 ln_ice_embd = F
 ln_dynALL = T
 ln_dynRHGADV = F
 ln_ridging = F
 ln_rafting = F
 ln_str_H79 = T
 ln_str_smooth = T
 ln_aEVP = T
 nn_nevp = 100
'''
import tempfile
with tempfile.TemporaryDirectory() as td:
    candidate=pathlib.Path(td)/'synthetic.ocean.output'
    candidate.write_text(synthetic)
    verify_o5_log(candidate)
    try: verify_o5_log(candidate, plant='selector')
    except ValueError: pass
    else: raise SystemExit('STATUS PLANT-FIRED: synthetic O5 selector plant did not fail')
print('PASS_REAL_EXCERPT_PROVENANCE_AND_SYNTHETIC_O5_LOG; real O5 log is checked at admission')
PY
}

run_preflight() {
  local scratch; scratch=$(mktemp -d); ACTIVE_SCRATCH=$scratch
  printf 'PREFLIGHT 1/10 deck and one-rank three-step overlay\n'; verify_deck
  awk '{print} /nn_itend[[:space:]]*=/ {print "   nn_itned    =       3   ! misspelled-variable plant"}' \
    "$NML" > "$scratch/namelist_misspelled_cfg"
  local namelist_plant_output
  if namelist_plant_output=$(verify_deck "$scratch/namelist_misspelled_cfg" "$ICE_NML" 2>&1); then
    refuse 'misspelled namelist variable plant did not refuse'
  fi
  grep -F 'unknown namelist variable nn_itned in &namrun' <<< "$namelist_plant_output" >/dev/null || \
    refuse "misspelled namelist variable plant refused for the wrong reason: $namelist_plant_output"
  printf 'PASS_MISSPELLED_NAMELIST_VARIABLE_PLANT\n'
  printf 'PREFLIGHT 2/10 source patches, preprocessed call-point landings, misplacement plants\n'; check_patches_and_landings "$scratch"
  printf 'PREFLIGHT 3/10 same-program on patched/unpatched real sources preprocessed with record-build CPP flags\n'
  preflight_real_same_program "$scratch"
  printf 'PREFLIGHT 4/10 bounds-checked patched-ppsrc fixture, two ranks, real extents and signed zero\n'; compile_fixture "$scratch/fixture" "$scratch/pp"
  printf 'PREFLIGHT 5/10 reader and admission payload plants\n'; reader_plants
  printf 'PREFLIGHT 6/10 real excerpt provenance and synthetic dynamics-on O5 overlay\n'; verify_real_excerpt
  printf 'PREFLIGHT 7/10 SAS time level derived from Nnn source\n'
  grep -q 'Nbb = 1; Nnn = 2; Naa = 3; Nrhs = Naa' "$NEMO_ROOT/src/SAS/nemogcm.F90"
  grep -q 'CALL sbc    ( kstp, Nbb, Nnn )' "$NEMO_ROOT/src/SAS/step.F90"
  printf 'PREFLIGHT 8/10 real-flag FP census and injected-operation plant\n'
  fp_census "$O2_REF_BLD" "$O2_REC_BLD" "$scratch/fp_census.json"
  "$FC" -x f95 -c -O3 -fdefault-real-8 -o "$scratch/fp_plant.o" - <<'F90'
SUBROUTINE o5_injected_fp_operation(x, y)
  REAL, VOLATILE, INTENT(INOUT) :: x, y
  y = x * 1.0000000000000002 + y
END SUBROUTINE o5_injected_fp_operation
F90
  if fp_census "$O2_REF_BLD" "$O2_REC_BLD" "$scratch/fp_census_plant.json" "$scratch/fp_plant.o"; then
    refuse 'injected floating-point operation census plant did not fail'
  fi
  printf 'PREFLIGHT 9/10 memory/disk/timeout/trap gates present\n'; ram_gate 1; disk_gate "$EVIDENCE" 1024
  grep -q 'timeout "$RUN_TIMEOUT_SEC" mpirun -np 1' "$0"
  printf 'PREFLIGHT 10/10 default is read-only; operator block not entered\n'
  if [[ -n "${PREFLIGHT_JSON:-}" ]]; then
    "$PY" - "$PREFLIGHT_JSON" <<'PY'
import json,pathlib,sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({"status":"PREFLIGHT_OK","nemo_run":False,"plants":32},indent=2)+"\n")
PY
  fi
  printf 'PREFLIGHT_OK: no NEMO build or integration was run\n'
  rm -rf -- "$scratch"
  ACTIVE_SCRATCH=""
}

fp_census() {
  local ref_bld=$1 rec_bld=$2 out=$3 plant_obj=${4:-}
  "$PY" - "$ref_bld" "$rec_bld" "$out" "$plant_obj" <<'PY'
import json,pathlib,re,shlex,subprocess,sys
ref,rec,out=map(pathlib.Path,sys.argv[1:4]); plant=pathlib.Path(sys.argv[4]) if sys.argv[4] else None
def fortran_flags(bld):
    values={}
    for line in (bld/'arch_nemo.fcm').read_text(errors='ignore').splitlines():
        match=re.match(r'^%(FCFLAGS|FFLAGS)\s+(.*)$',line)
        if match: values[match.group(1)]=match.group(2).strip()
    text=values.get('FFLAGS',values.get('FCFLAGS','')).replace('%FCFLAGS',values.get('FCFLAGS',''))
    flags=shlex.split(text)
    if '-O0' in flags or not any(re.fullmatch(r'-O[1-9gs]',flag) for flag in flags):
        raise SystemExit(f'STATUS PLANT-FIRED: observed Fortran flags are not optimized: {flags}')
    return flags
flags={'reference':fortran_flags(ref),'record':fortran_flags(rec)}
if flags['reference'] != flags['record']:
    raise SystemExit('STATUS PLANT-FIRED: reference/record Fortran flags differ')
units=('icedyn_rhg_evp','icedyn_rdgrft','icedyn')
header=re.compile(r'^\s*[0-9a-f]+\s+<([^>]+)>:$')
instruction=re.compile(r'^\s*[0-9a-f]+:\s+(?:(?:[0-9a-f]{2})\s+)+\s*([a-z][a-z0-9.]*)\b')
fp=re.compile(r'^(?:v)?(?:add|sub|mul|div|sqrt|max|min|fmadd|fmsub|fnmadd|fnmsub|cvt|cmp|comi|ucomi|and|andn|xor).*?(?:ss|sd|ps|pd)(?:x|y|z)?$|^f(?:add|sub|mul|div|sqrt)')
def one(path):
    raw=subprocess.run(['objdump','-d',str(path)],check=True,text=True,capture_output=True).stdout
    function='<outside>'; counts={}
    for line in raw.splitlines():
        found=header.match(line)
        if found: function=found.group(1); continue
        found=instruction.match(line)
        if found and fp.match(found.group(1).lower()): counts[function]=counts.get(function,0)+1
    return counts
report={}
for unit in units:
    reference=one(ref/'obj'/f'{unit}.o'); record=one(rec/'obj'/f'{unit}.o')
    if plant is not None and unit == 'icedyn':
        for name,count in one(plant).items(): record[f'PLANT::{name}']=count
    report[unit]={'reference':reference,'record':record,'reference_total':sum(reference.values()),'record_total':sum(record.values())}
bad=[unit for unit,value in report.items() if value['reference'] != value['record']]
zero=[unit for unit,value in report.items() if value['reference_total'] == 0 or value['record_total'] == 0]
payload={'status':'PASS_FP_CENSUS_REAL_FLAGS','fortran_flags':flags,'units':report}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
if zero: raise SystemExit(f'STATUS PLANT-FIRED: zero FP census: {zero}')
if bad: raise SystemExit(f'STATUS PLANT-FIRED: per-function FP instruction census changed: {bad}')
print('PASS_FP_CENSUS_REAL_FLAGS '+ ' '.join(f"{u}={report[u]['reference_total']}" for u in units))
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
  for plant in stamp binary-checksum checksum time-level selector missing-complete missing-stop0 ref-rec-diff \
    wet-1ulp swapped-field fold-sign-fimask fold-sign-zdelta fold-sign-zp_delt fold-sign-u_ice \
    fold-sign-v_ice fold-sign-zs1 fold-sign-zs2 fold-sign-zs12 fold-sign-strength \
    exchange-before-smooth unmasked-smooth parity-swap; do
    if "$PY" "$GATE" --admit-evidence "$EVIDENCE" --expect-commit "$commit" --plant "$plant" >/dev/null 2>&1; then
      refuse "retained-evidence plant $plant did not refuse"
    fi
  done
}

if [[ "$mode" == --preflight-only ]]; then run_preflight; exit 0; fi
if [[ "$mode" == --admit-existing ]]; then
  run_preflight
  [[ -f "$EVIDENCE/manifest/producer_commit.txt" ]] || refuse "missing producer commit"
  existing_commit=$(tr -d '[:space:]' < "$EVIDENCE/manifest/producer_commit.txt")
  admit "$existing_commit"; printf 'ACQUISITION_ADMITTED: %s\n' "$EVIDENCE"; exit 0
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
  local run_rc=0
  (cd "$dir" && printf 'RUN_STARTED\n' > run.user.time.log && \
    timeout "$RUN_TIMEOUT_SEC" mpirun -np 1 --oversubscribe ./nemo > run.user.stdout.log 2>&1) || run_rc=$?
  if [[ -f "$dir/ocean.output" ]] && \
     grep -E 'E[[:space:]]+R[[:space:]]+R[[:space:]]+O[[:space:]]+R' "$dir/ocean.output" >/dev/null; then
    "$PY" "$GATE" --verify-log "$dir/ocean.output" >&2 || true
    [[ "$run_rc" -ne 0 ]] || run_rc=1
  fi
  [[ "$run_rc" -eq 0 ]] || return "$run_rc"
  printf 'MPIRUN_RC=0\nRUN DONE\n' >> "$dir/run.user.time.log"
  touch "$dir/RUN_COMPLETE"
}

build_target "$REF_CFG" 0
build_target "$REC_CFG" 1
same_program
mkdir -p "$EVIDENCE/manifest/binaries"
cp "$REF_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.ref"
cp "$REC_ROOT/BLD/bin/nemo.exe" "$EVIDENCE/manifest/binaries/nemo.rec"
(cd "$EVIDENCE/manifest" && sha256sum binaries/nemo.ref binaries/nemo.rec > binaries.sha256)
stage_run "$REF_CFG" "$EVIDENCE/reference_3step"
stage_run "$REC_CFG" "$EVIDENCE/record_3step"
(cd "$EVIDENCE/record_3step" && find o5_si3_records -type f -print0 | sort -z | xargs -0 sha256sum > records.sha256)
touch "$EVIDENCE/RUN_COMPLETE"; RUN_COMPLETE=1
admit "$COMMIT"
printf 'ACQUISITION_DONE: %s\n' "$EVIDENCE"
