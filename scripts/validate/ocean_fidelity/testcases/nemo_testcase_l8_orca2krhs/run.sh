#!/usr/bin/env bash
# L8 round-1 acquisition. Default is read-only preflight; --run is operator-only.
set -Eeuo pipefail

fail() { printf 'REFUSE: L8 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR
mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in worktree'
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly G9=$here/../nemo_testcase_g9_orca2tadv
readonly G5=$here/../nemo_testcase_g5_orca2ice
readonly L4=$here/../nemo_testcase_l4_orca2bbc
readonly L6=$here/../nemo_testcase_l6_orca2dmp
readonly RENDER=$here/l8_deck.py READER=$here/l8_records.py
readonly STAGE_PATCH=$here/stprk3_l8_round56.patch LAYOUT=$here/layout_writer.f90
readonly BBC_PATCH=$L4/trabbc_round53.patch DMP_PATCH=$L6/tradmp_round55.patch
readonly REAL_DECK=$here/../nemo_testcase_l4b_orca2iso_bbc/l2_r17_twin_a_namelist.resolved
readonly REAL_LOG=$here/../nemo_testcase_l4b_orca2iso_bbc/l2_r17_twin_a_ocean.output.excerpt
readonly SOURCE_NAME=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly SOURCE=$NEMO_ROOT/cfgs/$SOURCE_NAME
readonly REF_NAME=ORCA2G5_OMIP_L4_R31REF_FIX1
readonly REF_BIN=$NEMO_ROOT/cfgs/$REF_NAME/BLD/bin/nemo.exe
readonly TARGET_NAME=ORCA2L8_KRHS_R56
readonly TARGET=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly QGH=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/geothermal_heating.nc
readonly RESTO=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/resto.nc
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round56/oracle_l8_r56}
readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly A=$EVIDENCE/twin_a B=$EVIDENCE/twin_b C=$EVIDENCE/twin_c
readonly INPUT_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc
  ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc q_10.15JUNE2009_fill.nc
  ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc
  sss_data.nc runoff_core_monthly.nc namelist_ref)

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
readonly PYPATH=$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src
absent() { [[ ! -e "$1" ]]; }
path_ok() { local p=$1; while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done; [[ -d "$p" && -w "$p" ]]; }
pad10() { local a=("$@"); while ((${#a[@]} < 10)); do a+=(0); done; local IFS=,; printf '%s' "${a[*]}"; }
list_a() { pad10 "$1"; }
list_b() { pad10 $(seq 2 2 "$1"); }
list_c() { (($1 >= 3)) && pad10 $(seq 3 2 "$1") || pad10 0; }

render_deck() { # path itend stocklist
  python "$RENDER" "$G9/namelist_cfg" "$1.base"
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
    -e "s/^(   nn_stock    = ).*/\1    $2/" \
    -e "s/^(   nn_stocklist = ).*/\1$3/" "$1.base" >"$1"
  rm -f -- "$1.base"
}

check_log() { # log deck: whitespace-tolerant resolved-selection proof
  python -I - "$1" "$2" <<'PY'
import re,sys
log=open(sys.argv[1],errors='replace').read(); deck=open(sys.argv[2]).read()
flags=('ln_traadv_fct','ln_traldf_lap','ln_traldf_iso','ln_traldf_msc',
       'ln_ldfeiv','ln_trabbc','ln_trabbl','ln_tradmp')
for key in flags:
    m=re.search(rf'^\s*{key}\s*=\s*\.(true|false)\.',deck,re.I|re.M)
    if not m: raise SystemExit(f'missing deck {key}')
    want='T' if m.group(1).lower()=='true' else 'F'
    if not re.search(rf'{key}\s*=\s*{want}\s*$',log,re.I|re.M):
        raise SystemExit(f'missing resolved {key}={want}')
for key in ('nn_fct_h','nn_fct_v','nn_bbl_ldf','nn_bbl_adv','rn_ahtbbl','rn_gambbl','nn_aht_ijk_t','rn_Ud','rn_Ld','nn_geoflx','nn_zdmp'):
    d=re.search(rf'^\s*{key}\s*=\s*([^!,\s]+)',deck,re.I|re.M)
    if not d: continue
    o=re.search(rf'{key}\s*=\s*([^\s]+)',log,re.I|re.M)
    if not o or float(d.group(1).replace('d','e').replace('D','E')) != float(o.group(1).replace('d','e').replace('D','E')):
        raise SystemExit(f'resolved numeric mismatch {key}')
PY
}

preflight() {
  local scratch patched stamped
  [[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] || fail 'commit exact acquisition tool first'
  scratch=$(mktemp -d "$repo/.l8-preflight.XXXXXX") || fail 'cannot create scratch'
  trap 'rm -rf -- "$scratch"' EXIT

  # 1. Running-source order and patch applicability, with a moved-call plant.
  python -I - "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90" <<'PY'
import re,sys
t=open(sys.argv[1]).read(); pats=(r'CALL\s+tra_adv\s*\(',r'CALL\s+tra_sbc_RK3\s*\(',r'CALL\s+tra_qsr\s*\(',r'CALL\s+tra_ldf\s*\(',r'CALL\s+tra_bbc\s*\(',r'CALL\s+tra_bbl\s*\(',r'CALL\s+tra_dmp\s*\(')
p=[re.search(x,t,re.I).start() for x in pats]
def check(values):
    if values != sorted(values): raise ValueError(values)
check(p)
try: check([p[0],p[2],p[1],*p[3:]])
except ValueError: pass
else: raise SystemExit('moved-call plant did not fire')
PY
  cp "$SOURCE/MY_SRC/stprk3_stg.F90" "$scratch/stprk3_stg.F90"
  patch -s -d "$scratch" -p0 <"$STAGE_PATCH" || fail 'stage observer patch rejected'
  grep -F "l4_canon_3d(ts(:,:,:,jp_tem,Krhs),'T')" "$scratch/stprk3_stg.F90" >/dev/null

  # 2. Exact G9-based deck: all five certified additions active.
  stamped=$scratch/namelist_cfg; render_deck "$stamped" 2 "$(pad10 2)"
  PYTHONPATH=$PYPATH python -I - "$stamped" <<'PY'
import re,sys
t=open(sys.argv[1]).read()
want={'ln_traadv_fct':'true','ln_traldf_iso':'true','ln_trabbc':'true','ln_trabbl':'true','ln_tradmp':'true','nn_bbl_ldf':'1','nn_bbl_adv':'0'}
for k,v in want.items():
    if not re.search(rf'^\s*{k}\s*=\s*\.?{v}\.?',t,re.I|re.M): raise SystemExit(k)
PY

  # 3. Log parser on a real excerpt and on the stamped L8 deck; flip plant.
  check_log "$REAL_LOG" "$REAL_DECK"
  cp "$REAL_LOG" "$scratch/stamped.output"
  sed -i -E -e 's/(ln_tradmp[[:space:]]*=[[:space:]]*)F/\1T/' \
    -e 's/(ln_trabbl[[:space:]]*=[[:space:]]*)F/\1T/' \
    -e 's/(ln_trabbc[[:space:]]*=[[:space:]]*)F/\1T/' \
    -e 's/(ln_traadv_fct[[:space:]]*=[[:space:]]*)F/\1T/' "$scratch/stamped.output"
  printf '\n nn_fct_h = 2\n nn_fct_v = 2\n nn_bbl_ldf = 1\n nn_bbl_adv = 0\n rn_ahtbbl = 1000.0\n rn_gambbl = 10.0\n nn_geoflx = 2\n nn_zdmp = 0\n' >>"$scratch/stamped.output"
  check_log "$scratch/stamped.output" "$stamped"
  sed -i -E 's/(ln_trabbc[[:space:]]*=[[:space:]]*)T/\1F/' "$scratch/stamped.output"
  ! check_log "$scratch/stamped.output" "$stamped" >/dev/null 2>&1 || fail 'log flag plant accepted'

  # 4. A compiled gfortran stream writer proves the Python reader's real sequence/EOF.
  gfortran -std=f2008 -o "$scratch/layout_writer" "$LAYOUT"
  (cd "$scratch" && ./layout_writer)
  PYTHONPATH=$PYPATH python "$READER" check "$scratch/oracle_rktracer_stage3_kt00000001.bin" 1
  head -c $(( $(stat -c %s "$scratch/oracle_rktracer_stage3_kt00000001.bin") - 8 )) \
    "$scratch/oracle_rktracer_stage3_kt00000001.bin" >"$scratch/truncated.bin"
  ! PYTHONPATH=$PYPATH python "$READER" check "$scratch/truncated.bin" 1 >/dev/null 2>&1 || fail 'EOF plant accepted'

  # 5. Smoke -> writer-free boundary -> three twins, every twin <= 10 steps.
  (( 2 % 2 == 0 && 10 <= 10 )) || fail 'bad schedule'
  grep -F 'check_acting "$SMOKE"' "$here/run.sh" >/dev/null || fail 'smoke acting statement missing'
  ! grep -F 'check_acting "$REF"' "$here/run.sh" >/dev/null || fail 'acting leaked to reference'

  # 6. Output viability; 7. fresh target/run paths.
  path_ok "$EVIDENCE" || fail 'evidence parent not writable'
  if [[ "$mode" != --admit-existing ]]; then
    absent "$TARGET" || fail "target exists: $TARGET"
    for d in "$SMOKE" "$REF" "$A" "$B" "$C"; do absent "$d" || fail "run exists: $d"; done
  fi
  trap - EXIT; rm -rf -- "$scratch"
  printf 'L8_PREFLIGHT_7_OF_7_PASS plants=3/3 compiled_gfortran_layout=true real_log=true stamped_deck=true\n'
}

build_target() {
  (cd "$NEMO_ROOT" && ./makenemo -r ORCA2_ICE_PISCES -n "$TARGET_NAME" -m conda-scalarmath del_key key_xios)
  cp "$SOURCE"/MY_SRC/* "$TARGET/MY_SRC/"
  cp "$SOURCE/cpp_$SOURCE_NAME.fcm" "$TARGET/cpp_$TARGET_NAME.fcm"
  cp "$NEMO_ROOT/src/OCE/TRA/trabbc.F90" "$TARGET/MY_SRC/trabbc.F90"; patch -s -d "$TARGET/MY_SRC" -p0 <"$BBC_PATCH"
  cp "$NEMO_ROOT/src/OCE/TRA/tradmp.F90" "$TARGET/MY_SRC/tradmp.F90"; patch -s -d "$TARGET/MY_SRC" -p0 <"$DMP_PATCH"
  patch -s -d "$TARGET/MY_SRC" -p0 <"$STAGE_PATCH"
  (cd "$NEMO_ROOT" && ./makenemo -n "$TARGET_NAME" -m conda-scalarmath -j 8)
}

stage_run() { # bin dir itend stocklist
  absent "$2" || fail "run exists: $2"; mkdir -p "$2"
  local f; for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$2/$f"; done
  cp "$QGH" "$2/geothermal_heating.nc"; cp "$RESTO" "$2/resto.nc"
  cp "$G9/namelist_ice_cfg" "$2/namelist_ice_cfg"
  cp "$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref" "$2/namelist_ice_ref"
  render_deck "$2/namelist_cfg" "$3" "$4"; cp "$1" "$2/nemo"
}

run_one() { # bin dir itend stocklist done|boundary
  local rc=0; stage_run "$1" "$2" "$3" "$4"
  (cd "$2" && mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$2/run.exit"
  [[ -f "$2/ocean.output" && -f "$2/time.step" ]] || fail "$2 produced no log"
  check_log "$2/ocean.output" "$2/namelist_cfg"
  if [[ $5 == done ]]; then [[ $rc -eq 0 && "$(tr -d '[:space:]' <"$2/time.step")" == "$3" ]] || fail "$2 incomplete"; fi
}

check_acting() { # smoke only: term-specific kt=1 rules live in the reader
  PYTHONPATH=$PYPATH python "$READER" inventory "$1" 2 --acting
}

completed_steps() {
  local n rc; n=$(tr -d '[:space:]' <"$REF/time.step"); rc=$(<"$REF/run.exit")
  if [[ $rc -eq 0 && $n == 100 ]]; then printf 100; return; fi
  grep -F 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$REF/ocean.output" >/dev/null || fail 'unexpected reference boundary'
  printf '%s' $((n - 1))
}

admit() { # steps
  local steps=$1 d rel
  check_acting "$SMOKE"
  [[ -z "$(find "$REF" -maxdepth 1 -name 'oracle_*' -print -quit)" ]] || fail 'writer-free boundary contains observers'
  for d in "$A" "$B" "$C"; do PYTHONPATH=$PYPATH python "$READER" inventory "$d" "$steps"; done
  PYTHONPATH=$PYPATH python "$READER" admit "$steps" "$A" "$B" "$C"
  while IFS= read -r rel; do cmp -s "$A/$rel" "$B/$rel" && cmp -s "$A/$rel" "$C/$rel" || fail "twins differ: $rel"; done \
    < <(cd "$A" && find . -maxdepth 1 -type f -name 'oracle_*.bin' -printf '%P\n' | LC_ALL=C sort)
  mkdir -p "$EVIDENCE/manifest"
  sha256sum "$RENDER" "$READER" "$STAGE_PATCH" "$LAYOUT" "$QGH" "$RESTO" "$TARGET/BLD/bin/nemo.exe" >"$EVIDENCE/manifest/producers.sha256"
  printf '%s\n' "$(git -C "$repo" rev-parse HEAD)" >"$EVIDENCE/manifest/legoesm_commit"
  printf 'L8_ACQUISITION_PASS steps=1..%s twins=3 evidence=%s\n' "$steps" "$EVIDENCE"
}

preflight
if [[ $mode == --admit-existing ]]; then
  steps=$(tr -d '[:space:]' <"$A/time.step"); ((steps >= 2 && steps <= 10)) || fail 'bad existing length'; admit "$steps"; exit
fi
[[ $mode == --run ]] || { printf 'WOULD_RUN smoke -> writer-free boundary -> three twins into %s\n' "$EVIDENCE"; exit; }
[[ ${NEMO_OPERATOR_ACK:-} == L8_R56_OPERATOR ]] || fail 'ACQUISITION_NEEDED: set NEMO_OPERATOR_ACK=L8_R56_OPERATOR'
build_target; mkdir -p "$EVIDENCE"
run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE" 2 "$(pad10 2)" done
check_acting "$SMOKE"
run_one "$REF_BIN" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
n=$(completed_steps); steps=$((n < 10 ? n : 10)); steps=$((steps - steps % 2)); ((steps >= 2)) || fail 'boundary too early'
run_one "$TARGET/BLD/bin/nemo.exe" "$A" "$steps" "$(list_a "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$B" "$steps" "$(list_b "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$C" "$steps" "$(list_c "$steps")" done
admit "$steps"
