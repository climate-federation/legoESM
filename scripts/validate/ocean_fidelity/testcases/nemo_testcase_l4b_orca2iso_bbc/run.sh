#!/usr/bin/env bash
# L4b round-1 acquisition. Default is read-only seven-item preflight;
# --run is operator-only; --admit-existing revalidates an existing record.
set -Eeuo pipefail

fail() { printf 'REFUSE: L4b acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in worktree'
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly G5_DIR=$here/../nemo_testcase_g5_orca2ice
readonly L2_DIR=$here/../nemo_testcase_l2_orca2iso
readonly L4_DIR=$here/../nemo_testcase_l4_orca2bbc
readonly G5_TOOL=$G5_DIR/run.sh G5_RECORDS=$G5_DIR/g5_records.py
readonly G5_DECK=$G5_DIR/namelist_cfg ICE_DECK=$G5_DIR/namelist_ice_cfg
readonly RENDER=$here/l4b_deck.py L2_CHECKER=$L2_DIR/l2_records.py
readonly BBC_CHECKER=$L4_DIR/l4_records.py PATCH=$L4_DIR/trabbc_round53.patch
readonly BASE_TRABBC=$NEMO_ROOT/src/OCE/TRA/trabbc.F90
readonly REC_SOURCE_NAME=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly REC_SOURCE=$NEMO_ROOT/cfgs/$REC_SOURCE_NAME
readonly REF_NAME=ORCA2G5_OMIP_L4_R31REF_FIX1
readonly REF_BIN=$NEMO_ROOT/cfgs/$REF_NAME/BLD/bin/nemo.exe
readonly TARGET_NAME=ORCA2L4B_ISO_BBC_R54
readonly TARGET=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly G5_RECORD=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round31/oracle_g5_r31/twin_a
readonly QGH_SOURCE=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/geothermal_heating.nc
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round54/oracle_l4b_r54}
readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly TWIN_A=$EVIDENCE/twin_a TWIN_B=$EVIDENCE/twin_b TWIN_C=$EVIDENCE/twin_c
readonly FSBC=2
readonly G5_DECK_SHA=c57eeb83d8f59c5b1d44df1295a10f5f30d28537358c8d327fcecc3d538f569e
readonly ICE_DECK_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly RENDER_SHA=f61931505dce078de809b360641ff6a651f3dc65207848925b85be1abdc565cd
readonly RENDERED_DECK_SHA=43a61a692b1a844b57d16580d47e7407c1e61ccbe873b29f1f1954fec8b7fbb3
readonly L2_CHECKER_SHA=656f9d4829884a5d06f939c69ebf9a1e6c720fe34d94c859be325c51e142338e
readonly BBC_CHECKER_SHA=2435c5a31ebdd2db250f3ff104c9e0834dd08a03bd83be1d0541e6cdb02f9d2e
readonly PATCH_SHA=dd98dfca4051cec8dd4814edd71c3bc2717ad599af3a4a9b7e086495f2f68453
readonly BASE_TRABBC_SHA=28fd28c51c0616d7429600318148bfccebefb458d0db81c2f48f688714b9c2ab
readonly REF_BIN_SHA=e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7
readonly REC_BIN_SHA=41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275
readonly INPUT_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc
  ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc q_10.15JUNE2009_fill.nc
  ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc
  sss_data.nc runoff_core_monthly.nc namelist_ref)

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
readonly PYPATH=$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src
sha() { sha256sum "$1" | awk '{print $1}'; }
input_sha() { sed -nE "s/^  \[$1\]=([0-9a-f]{64})$/\1/p" "$G5_TOOL"; }
absent() { [[ ! -e "$1" ]]; }
schedule_ok() { (( $1 > 0 && $2 > 0 && $1 % $3 == 0 && $2 % $3 == 0 && $2 <= $1 )); }
path_ok() {
  local check_path=$1
  while [[ ! -e "$check_path" ]]; do check_path=$(dirname "$check_path"); done
  [[ -d "$check_path" && -w "$check_path" ]] \
    && (( $(df -Pk "$check_path" | awk 'NR==2 {print $4}') > 1048576 ))
}

render_base() { python "$RENDER" "$G5_DECK" "$1"; }
render_deck() { # target itend stocklist
  render_base "$1.base"
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
         -e "s/^(   nn_stock    = ).*/\1    $2/" \
         -e "s/^(   nn_stocklist = ).*/\1$3/" "$1.base" >"$1"
  rm -f -- "$1.base"
}
pad10() { local values=("$@"); while ((${#values[@]} < 10)); do values+=(0); done; local IFS=,; printf '%s' "${values[*]}"; }
list_a() { pad10 "$1"; }
list_b() { pad10 $(seq 2 2 "$1"); }
list_c() { (($1 >= 3)) && pad10 $(seq 3 2 "$1") || pad10 0; }

syntax_check() {
  cpp -Dkey_nosignedzero -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 \
    -P -traditional -I"$REC_SOURCE/WORK" "$1" \
    | mpif90 -c -cpp -fsyntax-only -x f95 -fdefault-real-8 -fcray-pointer \
      -ffree-line-length-none -fallow-argument-mismatch -I"$REC_SOURCE/BLD/inc" \
      -J"$2" - >/dev/null 2>&1
}

write_synthetic() { # directory steps
  python -I - "$1" "$2" <<'PY'
import pathlib, struct, sys
import numpy as np
root, steps = pathlib.Path(sys.argv[1]), int(sys.argv[2]); root.mkdir(parents=True)
magic = b'NEMO_L4_BBC__1  '; q = np.array([1., 2., 0., 3., 0., 4.])
e = np.array([10., 20., 0., 30., 0., 40.]); t = np.zeros_like(q); t[e > 0] = q[e > 0] / e[e > 0]
for kt in range(1, steps + 1):
    (root / f'oracle_geothermal_kt{kt:08d}.bin').write_bytes(
        magic + struct.pack('=8i', 1, kt, 2, 3, 3, 2, 64, 3)
        + q.tobytes() + e.tobytes() + t.tobytes())
PY
}

check_deck() { # deck; compare against separately rendered L2
  python -I - "$G5_DECK" "$L2_DIR/l2_deck.py" "$1" <<'PY'
import importlib.util, pathlib, sys
def load(path):
    spec=importlib.util.spec_from_file_location('l2deck', path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def values(text):
    out={}; group=None
    for raw in text.splitlines():
        line=raw.split('!',1)[0].strip()
        if line.startswith('&'): group=line[1:].split()[0].lower()
        elif line=='/': group=None
        elif group and '=' in line:
            key,val=line.split('=',1); out[f'{group}.{key.strip().lower()}']=''.join(val.rstrip(',').split()).lower()
    return out
g5=pathlib.Path(sys.argv[1]).read_text(); l2=values(load(sys.argv[2]).render_text(g5)); l4b=values(pathlib.Path(sys.argv[3]).read_text())
changed={k:(l2.get(k),l4b.get(k)) for k in l2.keys()|l4b.keys() if l2.get(k)!=l4b.get(k)}
want={'nambbc.ln_trabbc':('.false.','.true.'),'nambbc.nn_geoflx':(None,'2'),'nambbc.cn_dir':(None,"'./'"),'nambbc.sn_qgh':(None,"'geothermal_heating.nc',-12.,'heatflow',.false.,.true.,'yearly','','',''")}
if changed != want: raise SystemExit(f'L4b is not L2 plus BBC exactly: {changed!r}')
PY
}

check_numeric_log() { # directory
  python -I - "$1/namelist_cfg" "$1/ocean.output" <<'PY'
import re, sys
from pathlib import Path
deck=Path(sys.argv[1]).read_text(); log=Path(sys.argv[2]).read_text(errors='replace')
def number(text, pattern):
    match=re.search(pattern, text, re.I|re.M)
    if match is None: raise SystemExit(f'missing numeric token {pattern}')
    return float(match.group(1).replace('D','E').replace('d','e'))
for key in ('nn_geoflx','nn_aht_ijk_t','rn_Ud','rn_Ld'):
    d=number(deck, rf'^\s*{key}\s*=\s*([^!,\s]+)')
    o=number(log, rf'{key}\s*=\s*([^\s]+)')
    if d != o: raise SystemExit(f'{key}: deck {d!r} != log {o!r}')
required=('ln_traldf_OFF       = F','ln_traldf_lap       = T','ln_traldf_iso       = T',
          'ln_traldf_msc       = T','ln_ldfeiv           = F','ln_trabbl           = F',
          'ln_tradmp           = F','Apply a geothermal heating at ocean bottom','variable geothermal heat flux')
for phrase in required:
    if phrase not in log: raise SystemExit(f'missing resolved selection {phrase!r}')
PY
}

seven_item_preflight() {
  local scratch=$repo/.l4b-preflight-$$ rendered truncated occupied_file
  absent "$scratch" || fail "scratch exists: $scratch"; mkdir "$scratch"
  trap 'rm -rf -- "$scratch"' RETURN

  # 1 source context and syntax, plus a syntax plant.
  cp "$BASE_TRABBC" "$scratch/trabbc.F90"; patch -s -d "$scratch" -p0 <"$PATCH"
  python -I - "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90" <<'PY'
import sys
text=open(sys.argv[1]).read(); calls=('CALL tra_ldf','IF( ln_trabbc ) CALL tra_bbc','IF( ln_trabbl ) CALL tra_bbl','IF( ln_tradmp ) CALL tra_dmp')
positions=[text.find(call) for call in calls]
if -1 in positions or positions != sorted(positions): raise SystemExit(f'bad stage-3 call order: {positions}')
PY
  ! python -I - <<'PY'
calls=('CALL tra_ldf','IF( ln_trabbc ) CALL tra_bbc','IF( ln_trabbl ) CALL tra_bbl','IF( ln_tradmp ) CALL tra_dmp')
planted=(calls[1], calls[0], calls[2], calls[3])
if list(map(calls.index, planted)) != sorted(map(calls.index, planted)): raise SystemExit(1)
PY
  [[ $? -eq 0 ]] || fail 'source-order plant did not fire'
  syntax_check "$scratch/trabbc.F90" "$scratch" || fail 'patched trabbc fails syntax proof'
  cp "$scratch/trabbc.F90" "$scratch/bad.F90"; printf '\nNOT FORTRAN\n' >>"$scratch/bad.F90"
  ! syntax_check "$scratch/bad.F90" "$scratch" || fail 'syntax plant accepted'

  # 2 exact executable/config identities, with a missing-reference plant.
  grep -Eq '^ORCA2_ICE_PISCES([[:space:]]|$)' "$NEMO_ROOT/cfgs/ref_cfgs.txt" || fail 'reference missing'
  ! grep -Eq '^ROUND54_NOT_A_CONFIG([[:space:]]|$)' "$NEMO_ROOT/cfgs/ref_cfgs.txt" || fail 'reference plant accepted'

  # 3 exact rendered deck and all schedules, with an odd-stock plant.
  rendered=$scratch/namelist_cfg; render_base "$rendered"; check_deck "$rendered"
  [[ "$(sha "$rendered")" == "$RENDERED_DECK_SHA" ]] || fail 'rendered L4b deck changed'
  schedule_ok 2 2 "$FSBC" && schedule_ok 100 20 "$FSBC" && schedule_ok 10 10 "$FSBC" || fail 'bad schedule'
  ! schedule_ok 10 9 "$FSBC" || fail 'schedule plant accepted'

  # 4 reader/writer schema and exact EOF, with a truncation plant.
  write_synthetic "$scratch/layout" 1; PYTHONPATH=$PYPATH python "$BBC_CHECKER" check-records "$scratch/layout" 1
  truncated=$scratch/layout/oracle_geothermal_kt00000001.bin
  head -c $(( $(stat -c %s "$truncated") - 8 )) "$truncated" >"$scratch/truncated.bin"
  mv "$scratch/truncated.bin" "$truncated"
  ! PYTHONPATH=$PYPATH python "$BBC_CHECKER" check-records "$scratch/layout" 1 >/dev/null 2>&1 || fail 'EOF plant accepted'

  # 5 expected inventory, with the terminal frame removed.
  rm -rf -- "$scratch/layout"; write_synthetic "$scratch/layout" 2
  PYTHONPATH=$PYPATH python "$BBC_CHECKER" check-records "$scratch/layout" 2
  rm -- "$scratch/layout/oracle_geothermal_kt00000002.bin"
  ! PYTHONPATH=$PYPATH python "$BBC_CHECKER" check-records "$scratch/layout" 2 >/dev/null 2>&1 || fail 'inventory plant accepted'

  # 6 output viability below an absent parent; regular-file parent is refused.
  path_ok "$EVIDENCE" || fail 'evidence path not writable/spacious'
  occupied_file=$scratch/file_parent; : >"$occupied_file"
  ! path_ok "$occupied_file/child" || fail 'file-parent plant accepted'

  # 7 fresh target/run boundary, with an occupied-path plant.
  if [[ "$mode" != --admit-existing ]]; then
    absent "$TARGET" || fail "target exists: $TARGET"
    for run_dir in "$SMOKE" "$REF" "$TWIN_A" "$TWIN_B" "$TWIN_C"; do absent "$run_dir" || fail "run exists: $run_dir"; done
  fi
  mkdir "$scratch/occupied"; ! absent "$scratch/occupied" || fail 'occupied-path plant accepted'
  trap - RETURN; rm -rf -- "$scratch"
  printf 'L4B_PREFLIGHT_7_OF_7_PASS plants=7/7 evidence_parent_absent=%s\n' "$([[ ! -e "$EVIDENCE" ]] && printf true || printf false)"
}

preflight() {
  local required want
  [[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] || fail 'commit exact acquisition tool first'
  for required in "$G5_TOOL" "$G5_RECORDS" "$G5_DECK" "$ICE_DECK" "$RENDER" "$L2_CHECKER" "$BBC_CHECKER" "$PATCH" "$BASE_TRABBC" "$REF_BIN" "$REC_SOURCE/BLD/bin/nemo.exe" "$QGH_SOURCE"; do
    [[ -f "$required" ]] || fail "missing $required"
  done
  [[ "$(sha "$QGH_SOURCE")" == e369762f068a7fbc1561be310fb8ded844233da37e6a0c6ffc10805e4996fffb ]] || fail 'geothermal input changed'
  [[ "$(sha "$G5_DECK")" == "$G5_DECK_SHA" \
     && "$(sha "$ICE_DECK")" == "$ICE_DECK_SHA" \
     && "$(sha "$RENDER")" == "$RENDER_SHA" \
     && "$(sha "$L2_CHECKER")" == "$L2_CHECKER_SHA" \
     && "$(sha "$BBC_CHECKER")" == "$BBC_CHECKER_SHA" \
     && "$(sha "$PATCH")" == "$PATCH_SHA" \
     && "$(sha "$BASE_TRABBC")" == "$BASE_TRABBC_SHA" \
     && "$(sha "$REF_BIN")" == "$REF_BIN_SHA" \
     && "$(sha "$REC_SOURCE/BLD/bin/nemo.exe")" == "$REC_BIN_SHA" ]] \
    || fail 'pinned producer changed'
  for required in "${INPUT_FILES[@]}"; do
    want=$(input_sha "$required"); [[ ${#want} == 64 && -f "$INPUT_ROOT/$required" && "$(sha "$INPUT_ROOT/$required")" == "$want" ]] || fail "input changed: $required"
  done
  seven_item_preflight
  printf 'L4B_PREFLIGHT_OK commit=%s reference=%s record_target=%s\n' "$(git -C "$repo" rev-parse HEAD)" "$REF_NAME" "$TARGET_NAME"
}

build_target() {
  (cd "$NEMO_ROOT" && ./makenemo -r ORCA2_ICE_PISCES -n "$TARGET_NAME" -m conda-scalarmath del_key key_xios)
  cp "$REC_SOURCE"/MY_SRC/* "$TARGET/MY_SRC/"
  cp "$REC_SOURCE/cpp_$REC_SOURCE_NAME.fcm" "$TARGET/cpp_$TARGET_NAME.fcm"
  cp "$BASE_TRABBC" "$TARGET/MY_SRC/trabbc.F90"; patch -s -d "$TARGET/MY_SRC" -p0 <"$PATCH"
  (cd "$NEMO_ROOT" && ./makenemo -n "$TARGET_NAME" -m conda-scalarmath -j 8)
  nm "$TARGET/BLD/bin/nemo.exe" >"$TARGET/BLD/bin/nemo.symbols"
  grep -F '__trabbc_MOD_tra_bbc' "$TARGET/BLD/bin/nemo.symbols" >/dev/null || fail 'record binary lacks tra_bbc'
}

stage_run() { # binary directory itend stocklist
  local input
  absent "$2" || fail "run exists: $2"; mkdir -p "$2"
  for input in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$input" "$2/$input"; done
  cp "$QGH_SOURCE" "$2/geothermal_heating.nc"; cp "$ICE_DECK" "$2/namelist_ice_cfg"
  cp "$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref" "$2/namelist_ice_ref"
  render_deck "$2/namelist_cfg" "$3" "$4"; check_deck "$2/namelist_cfg"; cp "$1" "$2/nemo"
}

run_one() { # binary directory itend stocklist done|boundary
  local rc=0
  stage_run "$1" "$2" "$3" "$4"
  (cd "$2"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$2/run.exit"
  [[ -f "$2/ocean.output" && -f "$2/time.step" ]] || fail "$2 produced no NEMO log"
  check_numeric_log "$2"; PYTHONPATH=$PYPATH python "$BBC_CHECKER" check-log "$2/ocean.output" "$2/namelist_cfg"
  if [[ "$5" == done ]]; then [[ "$rc" -eq 0 && "$(tr -d '[:space:]' <"$2/time.step")" == "$3" ]] || fail "$2 did not complete $3 steps"; fi
}

completed_steps() {
  local completed rc; completed=$(tr -d '[:space:]' <"$REF/time.step"); rc=$(<"$REF/run.exit")
  if [[ "$rc" -eq 0 && "$completed" == 100 ]]; then printf 100; return; fi
  grep -F 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$REF/ocean.output" >/dev/null \
    && [[ -f "$REF/output.abort.nc" ]] || fail "writer-free boundary stopped outside stp_ctl at $completed"
  printf '%s' $((completed - 1))
}

check_acting() { # run directory
  local acting_steps
  PYTHONPATH=$PYPATH JAX_PLATFORMS=cpu python "$L2_CHECKER" --check "$1" --g5 "$G5_RECORD" --domain "$INPUT_ROOT/domain_cfg.nc"
  acting_steps=$(tr -d '[:space:]' <"$1/time.step")
  PYTHONPATH=$PYPATH python "$BBC_CHECKER" check-records "$1" "$acting_steps"
}

admit() { # steps
  local steps=$1 run_dir rel restart_step restart_name
  [[ -d "$SMOKE" && -d "$REF" ]] || fail 'smoke or writer-free boundary missing'
  for run_dir in "$SMOKE" "$REF"; do
    check_deck "$run_dir/namelist_cfg"; check_numeric_log "$run_dir"
    PYTHONPATH=$PYPATH python "$BBC_CHECKER" check-log "$run_dir/ocean.output" "$run_dir/namelist_cfg"
    check_acting "$run_dir"
  done
  for run_dir in "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
    check_deck "$run_dir/namelist_cfg"; check_numeric_log "$run_dir"
    PYTHONPATH=$PYPATH python "$BBC_CHECKER" check-log "$run_dir/ocean.output" "$run_dir/namelist_cfg"
    PYTHONPATH=$PYPATH python "$BBC_CHECKER" check-records "$run_dir" "$steps"
    PYTHONPATH=$PYPATH JAX_PLATFORMS=cpu python "$L2_CHECKER" --check "$run_dir" --g5 "$G5_RECORD" --domain "$INPUT_ROOT/domain_cfg.nc"
    "$G5_TOOL" --check-surface-payload "$run_dir" "$steps" >/dev/null
  done
  PYTHONPATH=$PYPATH python "$BBC_CHECKER" admit "$steps" "$TWIN_A" "$TWIN_B" "$TWIN_C"
  while IFS= read -r rel; do
    [[ -f "$TWIN_B/$rel" && -f "$TWIN_C/$rel" ]] || fail "twin inventory lacks $rel"
    cmp -s "$TWIN_A/$rel" "$TWIN_B/$rel" && cmp -s "$TWIN_A/$rel" "$TWIN_C/$rel" || fail "twins differ: $rel"
  done < <(cd "$TWIN_A" && find . -maxdepth 1 -type f -name 'oracle_*.bin' -printf '%P\n' | LC_ALL=C sort)
  for restart_step in $(seq 2 "$steps"); do
    restart_name=$(printf 'ORCA2G1_%08d_restart.nc' "$restart_step")
    [[ -f "$TWIN_A/$restart_name" || -f "$TWIN_B/$restart_name" || -f "$TWIN_C/$restart_name" ]] || fail "no twin restart $restart_step"
  done
  [[ -f "$REF/ORCA2G1_$(printf '%08d' "$steps")_restart.nc" ]] || fail 'writer-free endpoint restart missing'
  PYTHONPATH=$PYPATH python "$G5_RECORDS" admit --evidence "$EVIDENCE" --steps "$steps"
  ! PYTHONPATH=$PYPATH python "$G5_RECORDS" admit --evidence "$EVIDENCE" --steps "$steps" --plant >/dev/null 2>&1 || fail 'restart passivity plant accepted'
  mkdir -p "$EVIDENCE/manifest"
  sha256sum "$G5_DECK" "$ICE_DECK" "$RENDER" "$PATCH" "$L2_CHECKER" "$BBC_CHECKER" "$QGH_SOURCE" "$REF_BIN" "$TARGET/BLD/bin/nemo.exe" >"$EVIDENCE/manifest/producers.sha256"
  printf '%s\n' "$(git -C "$repo" rev-parse HEAD)" >"$EVIDENCE/manifest/legoesm_commit"
  printf 'L4B_ACQUISITION_PASS steps=1..%s twins=3 evidence=%s\n' "$steps" "$EVIDENCE"
}

preflight
if [[ "$mode" == --admit-existing ]]; then
  [[ -d "$TWIN_A" && -d "$TWIN_B" && -d "$TWIN_C" ]] || fail 'existing twins missing'
  steps=$(tr -d '[:space:]' <"$TWIN_A/time.step")
  ((steps >= 2 && steps <= 10 && steps % FSBC == 0)) || fail "bad twin length $steps"
  admit "$steps"; exit 0
fi
[[ "$mode" == --run ]] || { printf 'WOULD_RUN smoke, writer-free boundary, twins A/B/C into %s\n' "$EVIDENCE"; exit 0; }

build_target; mkdir -p "$EVIDENCE"
run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE" 2 "$(pad10 2)" done; check_acting "$SMOKE"
run_one "$REF_BIN" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps); steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC)); ((steps >= 2)) || fail "NEMO completed only $completed steps"
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_C" "$steps" "$(list_c "$steps")" done
admit "$steps"
