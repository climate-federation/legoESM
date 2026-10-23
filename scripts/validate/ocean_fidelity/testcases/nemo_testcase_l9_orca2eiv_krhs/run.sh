#!/usr/bin/env bash
# L9 round-1 acquisition. Default is read-only preflight; --run is operator-only.
set -Eeuo pipefail

fail() { printf 'REFUSE: L9 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR
mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in worktree'
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_CFG=ORCA2_ICE_PISCES TARGET_NAME=ORCA2L9_EIV_KRHS_R57
readonly TARGET=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly SOURCE=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly ORCA2=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4
readonly REF_NAME=ORCA2G5_OMIP_L4_R31REF_FIX1
readonly REF_BIN=$NEMO_ROOT/cfgs/$REF_NAME/BLD/bin/nemo.exe
readonly G9=$here/../nemo_testcase_g9_orca2tadv
readonly L3=$here/../nemo_testcase_l3_orca2eiv
readonly L4=$here/../nemo_testcase_l4_orca2bbc
readonly L6=$here/../nemo_testcase_l6_orca2dmp
readonly L8=$here/../nemo_testcase_l8_orca2krhs
readonly RENDER=$here/l9_deck.py READER=$here/l9_records.py
readonly INSTRUMENT=$L3/instrument_ldftra.py LAYOUT=$here/layout_writer.f90
readonly STAGE_PATCH=$L8/stprk3_l8_round56.patch
readonly BBC_PATCH=$L4/trabbc_round53.patch DMP_PATCH=$L6/tradmp_round55.patch
readonly REAL_DECK=$here/../nemo_testcase_l4b_orca2iso_bbc/l2_r17_twin_a_namelist.resolved
readonly REAL_LOG=$here/../nemo_testcase_l4b_orca2iso_bbc/l2_r17_twin_a_ocean.output.excerpt
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly QGH=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/geothermal_heating.nc
readonly RESTO=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/resto.nc
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round57/oracle_l9_r57}
readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly A=$EVIDENCE/twin_a B=$EVIDENCE/twin_b C=$EVIDENCE/twin_c
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly INC=$ORCA2/BLD/inc
readonly INPUT_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc
  ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc q_10.15JUNE2009_fill.nc
  ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc
  sss_data.nc runoff_core_monthly.nc namelist_ref)

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
readonly PYPATH=$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src
scratch=

absent() { [[ ! -e "$1" ]]; }
ref_is_listed() { awk -v n="$1" '$1==n {found=1} END{exit !found}' "$NEMO_ROOT/cfgs/ref_cfgs.txt"; }
path_ok() { local p=$1; while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done; [[ -d "$p" && -w "$p" ]]; }
pad10() { local a=("$@"); while ((${#a[@]} < 10)); do a+=(0); done; local IFS=,; printf '%s' "${a[*]}"; }
list_a() { pad10 "$1"; }
list_b() { pad10 $(seq 2 2 "$1"); }
list_c() { (($1 >= 3)) && pad10 $(seq 3 2 "$1") || pad10 0; }

check_clock() { # rendered deck; NEMO sbcmod.f90:342-347
  "$PY" - "$1" <<'PY'
import re,sys
t=open(sys.argv[1]).read()
def one(name, kind=r"[0-9]+"):
    x=re.findall(rf"^\s*{name}\s*=\s*({kind})",t,re.I|re.M)
    if len(x)!=1: raise SystemExit(name)
    return x[0]
fsbc,it000,itend,stock=(int(one(x)) for x in ("nn_fsbc","nn_it000","nn_itend","nn_stock"))
listed=one("ln_rst_list",r"\.(?:true|false)\.").lower()==".true."
if fsbc!=2 or (itend-it000+1)%fsbc or (not listed and stock%fsbc): raise SystemExit(1)
PY
}

render_deck() { # path itend stocklist
  "$PY" "$RENDER" "$G9/namelist_cfg" "$1.base"
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
    -e "s/^(   nn_stock    = ).*/\1    $2/" \
    -e "s/^(   nn_stocklist = ).*/\1$3/" "$1.base" >"$1"
  rm -f -- "$1.base"
  check_clock "$1" || fail "bad rendered deck $1"
}

check_log() { # log deck: whitespace-tolerant logical/numeric comparison
  "$PY" - "$1" "$2" <<'PY'
import re,sys
log=open(sys.argv[1],errors='replace').read(); deck=open(sys.argv[2]).read()
flags=('ln_traadv_fct','ln_traldf_lap','ln_traldf_iso','ln_traldf_msc',
       'ln_ldfeiv','ln_eke_equ','ln_trabbc','ln_trabbl','ln_tradmp')
for key in flags:
    d=re.findall(rf'^\s*{key}\s*=\s*\.(true|false)\.',deck,re.I|re.M)
    if not d: continue
    want='T' if d[-1].lower()=='true' else 'F'
    got=re.findall(rf'{key}\s*=\s*([TF])\s*$',log,re.I|re.M)
    if not got or got[-1].upper()!=want: raise SystemExit(f'{key}: {got} != {want}')
number=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?'
for key in ('nn_fct_h','nn_fct_v','nn_bbl_ldf','nn_bbl_adv','rn_ahtbbl',
            'rn_gambbl','nn_aht_ijk_t','rn_Ue','rn_Le','nn_aht_ijk_t','rn_Ud',
            'rn_Ld','nn_geoflx','nn_zdmp'):
    d=re.findall(rf'^\s*{key}\s*=\s*({number})',deck,re.I|re.M)
    if not d: continue
    got=re.findall(rf'{key}\s*=\s*({number})',log,re.I|re.M)
    cv=lambda x: float(x.replace('d','e').replace('D','E'))
    if not got or cv(got[-1])!=cv(d[-1]): raise SystemExit(f'{key}: {got} != {d[-1]}')
PY
}

check_writers() { # patched stprk3, patched ldftra
  [[ $(grep -c "WRITE(l2_tracer_unit) l4_canon_3d" "$1") -eq 9 ]] || return 1
  local tag
  for tag in 'BEFORE_U        ' 'BEFORE_V        ' 'AFTER_U         ' \
             'AFTER_V         ' 'DELTA_U         ' 'DELTA_V         '; do
    [[ $(grep -F -c "WRITE(l3_unit) '$tag'" "$2") -eq 1 ]] || return 1
  done
}

preflight() {
  local stamped f
  [[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] || fail 'commit exact acquisition tool first'
  scratch=$(mktemp -d "$repo/.l9-preflight.XXXXXX") || fail 'cannot create scratch'
  trap 'rm -rf -- "$scratch"' EXIT

  # 1. Apply and syntax-check every patched NEMO file; continuation plant.
  cp "$SOURCE/MY_SRC/stprk3_stg.F90" "$scratch/stprk3_stg.F90"
  patch --batch -s -d "$scratch" -p0 <"$STAGE_PATCH" || fail 'stage patch rejected'
  "$PY" "$INSTRUMENT" "$ORCA2/BLD/ppsrc/nemo/ldftra.f90" "$scratch/ldftra.f90"
  cp "$NEMO_ROOT/src/OCE/TRA/trabbc.F90" "$scratch/trabbc.F90"
  patch --batch -s -d "$scratch" -p0 <"$BBC_PATCH" || fail 'BBC patch rejected'
  cp "$NEMO_ROOT/src/OCE/TRA/tradmp.F90" "$scratch/tradmp.F90"
  patch --batch -s -d "$scratch" -p0 <"$DMP_PATCH" || fail 'DMP patch rejected'
  for f in stprk3_stg.F90 ldftra.f90 trabbc.F90 tradmp.F90; do
    "$FC" -cpp -ffree-line-length-none -fsyntax-only -I"$INC" -J"$scratch" "$scratch/$f" \
      || fail "$f syntax failed"
  done
  if "$PY" "$INSTRUMENT" --plant-after-continuation \
       "$ORCA2/BLD/ppsrc/nemo/ldftra.f90" "$scratch/continued.f90" >/dev/null 2>&1; then
    fail "continued-'&' insertion plant accepted"
  fi

  # The live call graph: EIV augments stage-3 transport before Krhs is cleared;
  # the remaining calls keep L8's exact order. A moved-call plant is refused.
  "$PY" - "$ORCA2/BLD/ppsrc/nemo/traadv.f90" "$ORCA2/BLD/ppsrc/nemo/stprk3_stg.f90" <<'PY'
import re,sys
adv=open(sys.argv[1]).read(); stg=open(sys.argv[2]).read()
if not re.search(r'IF\(\s*ln_ldfeiv.*?CALL\s+ldf_eiv_trp.*?CALL\s+wzv',adv,re.I|re.S): raise SystemExit('EIV order')
pats=(r'ts\(:,:,:,jn,Krhs\)\s*=\s*0',r'CALL\s+tra_adv\s*\(',r'CALL\s+tra_sbc_RK3\s*\(',r'CALL\s+tra_qsr\s*\(',r'CALL\s+tra_ldf\s*\(',r'CALL\s+tra_bbc\s*\(',r'CALL\s+tra_bbl\s*\(',r'CALL\s+tra_dmp\s*\(',r'CALL\s+tra_zdf\s*\(')
p=[re.search(x,stg,re.I).start() for x in pats]
if p!=sorted(p): raise SystemExit('stage order')
bad=[*p]; bad[5],bad[6]=bad[6],bad[5]
try:
    if bad!=sorted(bad): raise ValueError
except ValueError: pass
else: raise SystemExit('moved-call plant did not fire')
PY

  # 2. makenemo -r must name a listed reference; plant a false name.
  ref_is_listed "$REF_CFG" || fail "$REF_CFG not listed in ref_cfgs.txt"
  ! ref_is_listed ORCA2_L9_NOT_A_REFERENCE || fail 'bad reference plant accepted'

  # 3. Render every possible run deck and enforce NEMO's clock rules; plants.
  render_deck "$scratch/smoke" 2 "$(pad10 2)"
  render_deck "$scratch/ref" 100 '2,4,6,8,10,12,14,16,18,20'
  local n
  for n in 2 4 6 8 10; do
    render_deck "$scratch/a$n" "$n" "$(list_a "$n")"
    render_deck "$scratch/b$n" "$n" "$(list_b "$n")"
    render_deck "$scratch/c$n" "$n" "$(list_c "$n")"
  done
  sed -E 's/^( *nn_itend *= *).*/\1 3/' "$scratch/c10" >"$scratch/bad-clock"
  ! check_clock "$scratch/bad-clock" || fail 'odd clock plant accepted'
  sed -E -e 's/^( *nn_stock *= *).*/\1 1/' \
    -e 's/^( *ln_rst_list *= *)\.true\./\1.false./' "$scratch/c10" >"$scratch/bad-stock"
  ! check_clock "$scratch/bad-stock" || fail 'odd stock plant accepted'

  # 4. Python selftest plus compiled Fortran writer for both exact streams.
  PYTHONPATH=$PYPATH "$PY" "$READER" selftest
  "$FC" -std=f2008 -o "$scratch/layout_writer" "$LAYOUT"
  (cd "$scratch" && ./layout_writer)
  PYTHONPATH=$PYPATH "$PY" "$READER" layout "$scratch"
  head -c $(( $(stat -c %s "$scratch/oracle_l3_eiv_kt00000001.bin") - 8 )) \
    "$scratch/oracle_l3_eiv_kt00000001.bin" >"$scratch/truncated-eiv.bin"
  ! "$PY" "$L3/l3_records.py" --check "$scratch/truncated-eiv.bin" \
      --mesh "$INPUT_ROOT/domain_cfg.nc" >/dev/null 2>&1 || fail 'EOF plant accepted'

  # 5. Every declared frame has a writer; deleting either family is refused.
  check_writers "$scratch/stprk3_stg.F90" "$scratch/ldftra.f90" || fail 'writer inventory incomplete'
  sed "/WRITE(l3_unit) 'DELTA_V/d" "$scratch/ldftra.f90" >"$scratch/missing-eiv.f90"
  ! check_writers "$scratch/stprk3_stg.F90" "$scratch/missing-eiv.f90" || fail 'missing EIV frame accepted'
  sed '0,/WRITE(l2_tracer_unit) l4_canon_3d/d' "$scratch/stprk3_stg.F90" >"$scratch/missing-stage.F90"
  ! check_writers "$scratch/missing-stage.F90" "$scratch/ldftra.f90" || fail 'missing stage frame accepted'

  # Whitespace-tolerant checker on a real committed excerpt and stamped L9 deck.
  check_log "$REAL_LOG" "$REAL_DECK"
  stamped=$scratch/stamped.output
  cp "$REAL_LOG" "$stamped"
  cat >>"$stamped" <<'LOG'
 ln_traadv_fct = T
 ln_traldf_lap = T
 ln_traldf_iso = T
 ln_traldf_msc = T
 ln_ldfeiv = T
 ln_eke_equ = F
 ln_trabbc = T
 ln_trabbl = T
 ln_tradmp = T
 nn_fct_h = 2
 nn_fct_v = 2
 nn_bbl_ldf = 1
 nn_bbl_adv = 0
 rn_ahtbbl = 1000.0
 rn_gambbl = 10.0
 nn_aei_ijk_t = 21
 rn_Ue = 2.9999999999999999E-002
 rn_Le = 200000.0
 nn_aht_ijk_t = 20
 rn_Ud = 0.018
 rn_Ld = 200000.0
 nn_geoflx = 2
 nn_zdmp = 0
LOG
  check_log "$stamped" "$scratch/smoke"
  sed -i -E 's/(ln_ldfeiv[[:space:]]*=[[:space:]]*)T/\1F/' "$stamped"
  ! check_log "$stamped" "$scratch/smoke" >/dev/null 2>&1 || fail 'log plant accepted'

  # Smoke-only acting call, schedule, output paths, and fresh targets/runs.
  (( 2 % 2 == 0 && 10 <= 10 )) || fail 'bad schedule'
  grep -F 'check_acting "$SMOKE"' "$here/run.sh" >/dev/null || fail 'smoke acting statement missing'
  ! awk '!/grep -F/ {print}' "$here/run.sh" | grep -F 'check_acting "$REF"' >/dev/null \
    || fail 'acting leaked to writer-free boundary'
  path_ok "$EVIDENCE" || fail 'evidence parent not writable'
  mkdir -p "$scratch/new/absent/evidence"; : >"$scratch/new/absent/evidence/probe"
  : >"$scratch/blocked-parent"
  ! mkdir -p "$scratch/blocked-parent/child" 2>/dev/null || fail 'file-parent plant accepted'
  if [[ "$mode" != --admit-existing ]]; then
    absent "$TARGET" || fail "target exists: $TARGET"
    for f in "$SMOKE" "$REF" "$A" "$B" "$C"; do absent "$f" || fail "run exists: $f"; done
    mkdir "$scratch/existing-target" "$scratch/existing-run"
    ! absent "$scratch/existing-target" || fail 'existing-target plant accepted'
    ! absent "$scratch/existing-run" || fail 'existing-run plant accepted'
  fi
  printf 'L9_PREFLIGHT_7_OF_7_PASS plants=all_refused compiled_gfortran_layout=true patched_syntax=4 real_log=true stamped_deck=true\n'
}

build_target() {
  (cd "$NEMO_ROOT" && ./makenemo -r "$REF_CFG" -n "$TARGET_NAME" -m conda-scalarmath del_key key_xios)
  cp -a "$SOURCE/MY_SRC/." "$TARGET/MY_SRC/"
  cp "$ORCA2/cpp_ORCA2_OMIP_L4.fcm" "$TARGET/cpp_$TARGET_NAME.fcm"
  cp "$NEMO_ROOT/src/OCE/LDF/ldftra.F90" "$TARGET/MY_SRC/ldftra.F90"
  "$PY" "$INSTRUMENT" "$TARGET/MY_SRC/ldftra.F90" "$TARGET/MY_SRC/ldftra.new.F90"
  mv "$TARGET/MY_SRC/ldftra.new.F90" "$TARGET/MY_SRC/ldftra.F90"
  cp "$NEMO_ROOT/src/OCE/TRA/trabbc.F90" "$TARGET/MY_SRC/trabbc.F90"
  patch --batch -s -d "$TARGET/MY_SRC" -p0 <"$BBC_PATCH"
  cp "$NEMO_ROOT/src/OCE/TRA/tradmp.F90" "$TARGET/MY_SRC/tradmp.F90"
  patch --batch -s -d "$TARGET/MY_SRC" -p0 <"$DMP_PATCH"
  patch --batch -s -d "$TARGET/MY_SRC" -p0 <"$STAGE_PATCH"
  (cd "$NEMO_ROOT" && ./makenemo -n "$TARGET_NAME" -m conda-scalarmath -j 8)
  [[ -x "$TARGET/BLD/bin/nemo.exe" ]] || fail 'target produced no executable'
}

stage_run() { # binary dir itend stocklist
  absent "$2" || fail "run exists: $2"; mkdir -p "$2"
  local f; for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$2/$f"; done
  cp "$QGH" "$2/geothermal_heating.nc"; cp "$RESTO" "$2/resto.nc"
  cp "$G9/namelist_ice_cfg" "$2/namelist_ice_cfg"
  cp "$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref" "$2/namelist_ice_ref"
  render_deck "$2/namelist_cfg" "$3" "$4"; cp "$1" "$2/nemo"
}

run_one() { # binary dir itend stocklist done|boundary
  local rc=0; stage_run "$1" "$2" "$3" "$4"
  (cd "$2" && mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$2/run.exit"
  [[ -f "$2/ocean.output" && -f "$2/time.step" ]] || fail "$2 produced no log"
  check_log "$2/ocean.output" "$2/namelist_cfg"
  if [[ $5 == done ]]; then [[ $rc -eq 0 && "$(tr -d '[:space:]' <"$2/time.step")" == "$3" ]] || fail "$2 incomplete"; fi
}

check_acting() { # smoke only; kt=1 rules live in the combined reader
  PYTHONPATH=$PYPATH "$PY" "$READER" acting "$1" 2 "$1/mesh_mask.nc"
}

completed_steps() {
  local n rc; n=$(tr -d '[:space:]' <"$REF/time.step"); rc=$(<"$REF/run.exit")
  if [[ $rc -eq 0 && $n == 100 ]]; then printf 100; return; fi
  grep -F 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$REF/ocean.output" >/dev/null || fail 'unexpected reference boundary'
  printf '%s' $((n - 1))
}

admit() { # steps
  local steps=$1 d rel step restart
  check_acting "$SMOKE"
  [[ -z "$(find "$REF" -maxdepth 1 -name 'oracle_*' -print -quit)" ]] || fail 'writer-free boundary contains observers'
  for d in "$A" "$B" "$C"; do
    PYTHONPATH=$PYPATH "$PY" "$READER" inventory "$d" "$steps" "$d/mesh_mask.nc"
  done
  PYTHONPATH=$PYPATH "$PY" "$READER" admit "$steps" "$A/mesh_mask.nc" "$A" "$B" "$C"
  while IFS= read -r rel; do
    cmp -s "$A/$rel" "$B/$rel" && cmp -s "$A/$rel" "$C/$rel" || fail "twins differ: $rel"
  done < <(cd "$A" && find . -maxdepth 1 -type f -name 'oracle_*.bin' -printf '%P\n' | LC_ALL=C sort)
  for step in $(seq 2 "$steps"); do
    restart=$(printf 'ORCA2G1_%08d_restart.nc' "$step")
    [[ -f "$A/$restart" || -f "$B/$restart" || -f "$C/$restart" ]] || fail "no restart for step $step"
  done
  mkdir -p "$EVIDENCE/manifest"
  sha256sum "$RENDER" "$READER" "$INSTRUMENT" "$STAGE_PATCH" "$BBC_PATCH" \
    "$DMP_PATCH" "$LAYOUT" "$QGH" "$RESTO" "$TARGET/BLD/bin/nemo.exe" \
    >"$EVIDENCE/manifest/producers.sha256"
  printf '%s\n' "$(git -C "$repo" rev-parse HEAD)" >"$EVIDENCE/manifest/legoesm_commit"
  printf 'L9_ACQUISITION_PASS steps=1..%s twins=3 evidence=%s\n' "$steps" "$EVIDENCE"
}

preflight
if [[ $mode == --admit-existing ]]; then
  steps=$(tr -d '[:space:]' <"$A/time.step"); ((steps >= 2 && steps <= 10)) || fail 'bad existing length'
  admit "$steps"; exit
fi
[[ $mode == --run ]] || { printf 'WOULD_RUN smoke -> writer-free boundary -> three twins into %s\n' "$EVIDENCE"; exit; }
[[ ${NEMO_OPERATOR_ACK:-} == L9_R57_OPERATOR ]] || fail 'ACQUISITION_NEEDED: set NEMO_OPERATOR_ACK=L9_R57_OPERATOR'
build_target; mkdir -p "$EVIDENCE"
run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE" 2 "$(pad10 2)" done
check_acting "$SMOKE"
run_one "$REF_BIN" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
n=$(completed_steps); steps=$((n < 10 ? n : 10)); steps=$((steps - steps % 2)); ((steps >= 2)) || fail 'boundary too early'
run_one "$TARGET/BLD/bin/nemo.exe" "$A" "$steps" "$(list_a "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$B" "$steps" "$(list_b "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$C" "$steps" "$(list_c "$steps")" done
admit "$steps"
