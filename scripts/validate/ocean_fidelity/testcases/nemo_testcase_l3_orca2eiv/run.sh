#!/usr/bin/env bash
# L3 acquisition: certified G9 + certified L2 + ORCA2 EIV.  Default mode is
# read-only preflight.  NEMO creation/build/run under --run is operator-only.
set -Eeuo pipefail

fail() { printf 'REFUSE: L3 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in a worktree'
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_CFG=ORCA2_ICE_PISCES TARGET_NAME=ORCA2L3_EIV_R20
readonly REF_TARGET_NAME=ORCA2L3_EIV_R20_REF
readonly TARGET=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly REF_TARGET=$NEMO_ROOT/cfgs/$REF_TARGET_NAME
readonly BASE=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly REF_BASE=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31REF_FIX1
readonly ORCA2=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4
readonly SOURCE=$NEMO_ROOT/src/OCE/LDF/ldftra.F90
readonly PP_SOURCE=$ORCA2/BLD/ppsrc/nemo/ldftra.f90
readonly INSTRUMENT=$here/instrument_ldftra.py READER=$here/l3_records.py
readonly DECK=$here/namelist_cfg ICE_DECK=$here/namelist_ice_cfg
readonly DECK_SHA=e782a21545b75a781193a706fdfeec91c4f6f32a644a4f41a3f8b2cfc099b457
readonly ICE_DECK_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly G5_TOOL=$here/../nemo_testcase_g5_orca2ice/run.sh
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref
readonly G9_RECORD=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round46/oracle_g9_r46/twin_a
readonly L2_RECORD=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round17/oracle_l2_r17/twin_a
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round20/oracle_l3_r20}
readonly FAILED1_LOG=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round20/oracle_l3_r20_failed1/twin_a/ocean.output
readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly TWIN_A=$EVIDENCE/twin_a TWIN_B=$EVIDENCE/twin_b TWIN_C=$EVIDENCE/twin_c
readonly FSBC=2
readonly INPUT_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc
  ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc q_10.15JUNE2009_fill.nc
  ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc
  sss_data.nc runoff_core_monthly.nc namelist_ref)
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

sha() { sha256sum "$1" | awk '{print $1}'; }
input_sha() { sed -nE "s/^  \[$1\]=([0-9a-f]{64})$/\1/p" "$G5_TOOL"; }
require_absent() { [[ ! -e "$1" ]] || return 1; }
ref_is_listed() { awk -v n="$1" '$1==n {found=1} END{exit !found}' "$NEMO_ROOT/cfgs/ref_cfgs.txt"; }

check_namelist_clock() { # NEMO's rule, sbcmod.f90:342-347
  "$PY" - "$1" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
def one(name, kind=r"[0-9]+"):
    hits = re.findall(rf"^\s*{name}\s*=\s*({kind})", text, re.M | re.I)
    if len(hits) != 1: raise SystemExit(1)
    return hits[0]
fsbc, it000, itend, stock = (int(one(n)) for n in ("nn_fsbc", "nn_it000", "nn_itend", "nn_stock"))
rst_list = one("ln_rst_list", r"\.(?:true|false)\.").lower() == ".true."
if fsbc != 2 or (itend - it000 + 1) % fsbc or (not rst_list and stock % fsbc):
    raise SystemExit(1)
PY
}

check_frame_writes() {
  local file=$1 tag
  [[ $(grep -c "WRITE(l3_unit) 1, kt, Kmm, Krhs" "$file") -eq 1 ]] || return 1
  for tag in 'BEFORE_U        ' 'BEFORE_V        ' 'AFTER_U         ' \
             'AFTER_V         ' 'DELTA_U         ' 'DELTA_V         '; do
    [[ $(grep -F -c "WRITE(l3_unit) '$tag'" "$file") -eq 1 ]] || return 1
  done
}

check_log_values() { # ocean.output, rendered namelist
  "$PY" - "$1" "$2" <<'PY'
import re, sys

log = open(sys.argv[1]).read()
deck = open(sys.argv[2]).read()

number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
def values(text, name, anchor):
    return re.findall(rf"(?im){anchor}\b{re.escape(name)}\s*=\s*({number})", text)

for name in ("nn_aei_ijk_t", "rn_Ue", "rn_Le"):
    wanted = values(deck, name, r"^\s*")
    got = values(log, name, "")   # ocean.output prints a description first
    if len(wanted) != 1 or not got:
        raise SystemExit(f"missing or ambiguous {name}: deck={wanted} log={got}")
    parse = lambda value: float(value.replace("D", "E").replace("d", "e"))
    expected = parse(wanted[0])
    if any(parse(value) != expected for value in got):
        raise SystemExit(f"{name} differs: deck={wanted[0]} log={got}")
PY
}

probe_output_parent() {
  local dir=$1
  mkdir -p "$dir" 2>/dev/null || return 1
  : >"$dir/.l3-write-probe" 2>/dev/null || return 1
  rm -f "$dir/.l3-write-probe"
}

preflight() {
  local tmp rendered planted f want
  [[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] \
    || fail 'commit the exact acquisition tool before running it'
  for f in "$SOURCE" "$PP_SOURCE" "$INSTRUMENT" "$READER" "$DECK" \
           "$ICE_DECK" "$G5_TOOL" "$ICE_REF" "$FC"; do
    [[ -f "$f" ]] || fail "missing $f"
  done
  [[ "$(sha "$DECK")" == "$DECK_SHA" ]] || fail 'L3 deck changed'
  [[ "$(sha "$ICE_DECK")" == "$ICE_DECK_SHA" ]] || fail 'ice deck changed'
  for f in "${INPUT_FILES[@]}"; do
    want=$(input_sha "$f"); [[ ${#want} == 64 ]] || fail "no input SHA for $f"
    [[ -f "$INPUT_ROOT/$f" && "$(sha "$INPUT_ROOT/$f")" == "$want" ]] \
      || fail "missing or changed input $f"
  done
  tmp=$(mktemp -d /tmp/orca2l3-preflight.XXXXXX)
  trap 'rm -rf -- "$tmp"' RETURN

  # 1: exact transform, refusal after '&', and gfortran syntax of every
  # patched NEMO file (ldftra is the only one).
  "$PY" "$INSTRUMENT" "$PP_SOURCE" "$tmp/ldftra.f90"
  "$FC" -ffree-line-length-none -fsyntax-only -I"$ORCA2/BLD/inc" \
    -J"$tmp" "$tmp/ldftra.f90" || fail 'instrumented ldftra syntax failed'
  if "$PY" "$INSTRUMENT" --plant-after-continuation "$PP_SOURCE" \
       "$tmp/continued.f90" >/dev/null 2>&1; then
    fail "continued-'&' insertion plant was accepted"
  fi

  # 2: makenemo reference is a listed ref_cfgs first token; planted name fails.
  ref_is_listed "$REF_CFG" || fail "$REF_CFG is not in ref_cfgs.txt"
  if ref_is_listed ORCA2_L3_NOT_A_REFERENCE; then fail 'bad -r plant was accepted'; fi

  # 3: every deck --run renders passes NEMO's clock rule (sbcmod.f90:342-347);
  # a run length off nn_fsbc and an unlisted odd nn_stock are refused.
  local n
  render_deck "$tmp/smoke" 2 "$(pad10 2)"
  render_deck "$tmp/ref" 100 '2,4,6,8,10,12,14,16,18,20'
  for n in 2 4 6 8 10; do
    render_deck "$tmp/a$n" "$n" "$(list_a "$n")"
    render_deck "$tmp/b$n" "$n" "$(list_b "$n")"
    render_deck "$tmp/c$n" "$n" "$(list_c "$n")"
  done
  rendered=$tmp/smoke
  sed -E 's/^( *nn_itend *= *).*/\1 3/' "$tmp/c10" >"$tmp/bad-clock"
  if check_namelist_clock "$tmp/bad-clock"; then fail 'odd run-length plant was accepted'; fi
  sed -E -e 's/^( *nn_stock *= *).*/\1 1/' \
    -e 's/^( *ln_rst_list *= *)\.true\./\1.false./' "$tmp/c10" >"$tmp/bad-stock"
  if check_namelist_clock "$tmp/bad-stock"; then fail 'odd nn_stock plant was accepted'; fi

  # 4: reader/writer byte identity, truncation/tag plants, and 16-byte literals.
  "$PY" "$READER" --selftest || fail 'record layout selftest failed'
  if "$PY" "$INSTRUMENT" --plant-short-literal "$PP_SOURCE" \
       "$tmp/short-literal.f90" >/dev/null 2>&1; then
    fail 'short character-literal plant was accepted'
  fi
  if "$PY" "$INSTRUMENT" --plant-raw-observer "$PP_SOURCE" \
       "$tmp/raw-observer.f90" >/dev/null 2>&1; then
    fail 'raw undefined-cell observer plant was accepted'
  fi

  # 5: all six declared frames have a writer; deleting one must be refused.
  check_frame_writes "$tmp/ldftra.f90" || fail 'expected writer frame missing'
  sed "/WRITE(l3_unit) 'DELTA_V/d" "$tmp/ldftra.f90" >"$tmp/missing-frame.f90"
  if check_frame_writes "$tmp/missing-frame.f90"; then fail 'missing-frame plant was accepted'; fi

  # 6: output creation works with absent parents; a regular-file parent fails.
  probe_output_parent "$tmp/new/absent/evidence" || fail 'absent output parent unusable'
  : >"$tmp/blocked-parent"
  if probe_output_parent "$tmp/blocked-parent/child"; then fail 'file-parent plant was accepted'; fi

  # The namelist/log admission parses Fortran numbers and compares their
  # binary64 values exactly.  The plant is NEMO's own ldf_tra_init print
  # (description first, gfortran's 0.03): the old rn_Ue text grep refused it,
  # the numeric check must accept it, and a neighbouring float must fail.
  printf '%s\n' \
    '          type of time-space variation            nn_aei_ijk_t  =           21' \
    '          lateral diffusive velocity (if cst)     rn_Ue         =    2.9999999999999999E-002  m/s' \
    '          lateral diffusive length   (if cst)     rn_Le         =    200000.00000000000       m' \
    >"$tmp/float-log"
  check_log_values "$tmp/float-log" "$rendered" \
    || fail "NEMO's rn_Ue print plant was refused"
  if grep -Eq 'rn_Ue += +3\.000000000000000[0-9]*E-002' "$tmp/float-log"; then
    fail 'old rn_Ue text grep unexpectedly accepted its refusal plant'
  fi
  sed 's/2\.9999999999999999E-002/3.0000000000000006E-002/' \
    "$tmp/float-log" >"$tmp/bad-float-log"
  if check_log_values "$tmp/bad-float-log" "$rendered"; then
    fail 'unequal rn_Ue numeric plant was accepted'
  fi
  if [[ -f "$FAILED1_LOG" ]]; then   # the refused first acquisition's own log
    check_log_values "$FAILED1_LOG" "$rendered" \
      || fail 'the first acquisition log is still refused'
  fi

  # Admission metadata plants.  These use symlinks to the immutable inputs so
  # preflight exercises the exact completed-run audit without copying data.
  plant_audit_run "$tmp/admit-metadata" 2 "$(pad10 2)"
  audit_run "$FC" "$tmp/admit-metadata" 2 "$(pad10 2)" done \
    || fail 'valid admission metadata plant was refused'
  printf '4\n' >"$tmp/admit-metadata/time.step"
  if audit_run "$FC" "$tmp/admit-metadata" 2 "$(pad10 2)" done; then
    fail 'wrong twin-B/C final-step plant was accepted'
  fi
  printf '2\n' >"$tmp/admit-metadata/time.step"
  sed -i 's/nn_itend    =       2/nn_itend    =       4/' \
    "$tmp/admit-metadata/namelist_cfg"
  if audit_run "$FC" "$tmp/admit-metadata" 2 "$(pad10 2)" done; then
    fail 'altered completed-run deck plant was accepted'
  fi
  render_deck "$tmp/admit-metadata/namelist_cfg" 2 "$(pad10 2)"
  sed -i '/ln_ldfeiv/d' "$tmp/admit-metadata/ocean.output"
  if audit_run "$FC" "$tmp/admit-metadata" 2 "$(pad10 2)" done; then
    fail 'altered completed-run log plant was accepted'
  fi
  plant_audit_log >"$tmp/admit-metadata/ocean.output"
  printf '1\n' >"$tmp/admit-metadata/run.exit"
  if audit_run "$FC" "$tmp/admit-metadata" 2 "$(pad10 2)" done; then
    fail 'incomplete smoke-run plant was accepted'
  fi

  # 7: both configurations and the run root must be absent; planted existing
  # paths fail.  --admit-existing deliberately audits completed evidence.
  if [[ "$mode" != --admit-existing ]]; then
    require_absent "$TARGET" || fail "target config exists: $TARGET"
    require_absent "$REF_TARGET" || fail "reference config exists: $REF_TARGET"
    require_absent "$EVIDENCE" || fail "run root exists: $EVIDENCE"
    mkdir "$tmp/existing-target" "$tmp/existing-run"
    if require_absent "$tmp/existing-target" || require_absent "$tmp/existing-run"; then
      fail 'existing target/run plant was accepted'
    fi
  else
    [[ -x "$TARGET/BLD/bin/nemo.exe" && -x "$REF_TARGET/BLD/bin/nemo.exe" \
       && -d "$EVIDENCE" ]] || fail 'existing acquisition is incomplete'
  fi
  rm -rf -- "$tmp"
  trap - RETURN
  printf 'L3_PREFLIGHT_OK checklist=7 plants=all_refused commit=%s\n' \
    "$(git -C "$repo" rev-parse HEAD)"
}

render_deck() { # destination, itend, stock-list
  sed -E -e "s/^(   nn_itend    = ).*/\1      $2/" \
    -e "s/^(   nn_stock    = ).*/\1      $2/" \
    -e "s/^(   nn_stocklist = ).*/\1 $3/" "$DECK" >"$1"
  check_namelist_clock "$1" || fail "bad rendered deck $1"
}

pad10() { local a=("$@"); while ((${#a[@]} < 10)); do a+=(0); done; local IFS=,; printf '%s' "${a[*]}"; }
list_a() { pad10 "$1"; }
list_b() { pad10 $(seq 2 2 "$1"); }
list_c() { (($1 >= 3)) && pad10 $(seq 3 2 "$1") || pad10 0; }

stage_run() { # binary, directory, itend, stock-list
  local f dir=$2
  require_absent "$dir" || fail "run directory exists: $dir"
  mkdir -p "$dir"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$dir/$f"; done
  cp "$ICE_REF" "$dir/namelist_ice_ref"; cp "$ICE_DECK" "$dir/namelist_ice_cfg"
  render_deck "$dir/namelist_cfg" "$3" "$4"
  cp "$1" "$dir/nemo"
}

check_log() {
  local out=$1/ocean.output
  grep -Eq 'ln_traadv_fct += +T$' "$out" \
    && grep -Eq 'ln_traldf_iso += +T$' "$out" \
    && grep -Eq 'ln_traldf_msc += +T$' "$out" \
    && grep -Eq 'ln_ldfeiv += +T$' "$out" \
    && grep -Fq 'ldf_eiv_trp : eddy induced advection' "$out" \
    || return 1
  check_log_values "$out" "$1/namelist_cfg" \
    || return 1
}

check_done() { # directory, expected final step
  local dir=$1 expected=$2 rc step
  [[ -f "$dir/run.exit" && -f "$dir/time.step" ]] || return 1
  rc=$(tr -d '[:space:]' <"$dir/run.exit")
  step=$(tr -d '[:space:]' <"$dir/time.step")
  [[ "$rc" == 0 && "$step" == "$expected" ]]
}

audit_run() { # binary, directory, itend, stock-list, done|boundary
  local binary=$1 dir=$2 itend=$3 stock=$4 completion=$5 expected f
  expected=$(mktemp /tmp/orca2l3-admit-deck.XXXXXX)
  render_deck "$expected" "$itend" "$stock"
  cmp -s "$expected" "$dir/namelist_cfg" || return 1
  rm -f "$expected"
  cmp -s "$ICE_REF" "$dir/namelist_ice_ref" || return 1
  cmp -s "$ICE_DECK" "$dir/namelist_ice_cfg" || return 1
  cmp -s "$binary" "$dir/nemo" || return 1
  for f in "${INPUT_FILES[@]}"; do
    cmp -s "$INPUT_ROOT/$f" "$dir/$f" || return 1
  done
  check_log "$dir" || return 1
  if [[ "$completion" == done ]]; then
    check_done "$dir" "$itend"
  else
    completed_steps "$dir" >/dev/null
  fi
}

plant_audit_log() {
  printf '%s\n' \
    ' ln_traadv_fct = T' ' ln_traldf_iso = T' ' ln_traldf_msc = T' \
    ' ln_ldfeiv = T' ' ldf_eiv_trp : eddy induced advection' \
    ' type of time-space variation nn_aei_ijk_t = 21' \
    ' lateral diffusive velocity rn_Ue = 2.9999999999999999E-002' \
    ' lateral diffusive length rn_Le = 200000.00000000000'
}

plant_audit_run() { # synthetic completed run for preflight-only plants
  local dir=$1 itend=$2 stock=$3 f
  mkdir -p "$dir"
  for f in "${INPUT_FILES[@]}"; do ln -s "$INPUT_ROOT/$f" "$dir/$f"; done
  ln -s "$ICE_REF" "$dir/namelist_ice_ref"
  ln -s "$ICE_DECK" "$dir/namelist_ice_cfg"
  ln -s "$FC" "$dir/nemo"
  render_deck "$dir/namelist_cfg" "$itend" "$stock"
  plant_audit_log >"$dir/ocean.output"
  printf '0\n' >"$dir/run.exit"
  printf '%s\n' "$itend" >"$dir/time.step"
}

run_one() { # binary, directory, itend, stock-list, done|boundary
  local dir=$2 rc=0
  stage_run "$1" "$dir" "$3" "$4"
  (cd "$dir"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$dir/run.exit"
  [[ -f "$dir/ocean.output" && -f "$dir/time.step" ]] \
    || fail "$dir produced no NEMO log"
  check_log "$dir" || fail "$dir did not execute exact L3 selection"
  if [[ "$5" == done ]]; then
    [[ $rc -eq 0 && "$(tr -d '[:space:]' <"$dir/time.step")" == "$3" ]] \
      || fail "$dir did not complete $3 steps"
  fi
}

build_one() { # target name, inherited source, instrument=yes|no
  local name=$1 inherited=$2 target=$NEMO_ROOT/cfgs/$1
  (cd "$NEMO_ROOT" && ./makenemo -r "$REF_CFG" -n "$name" \
    -m conda-scalarmath del_key key_xios)
  cp -a "$inherited/MY_SRC/." "$target/MY_SRC/"
  cp "$ORCA2/cpp_ORCA2_OMIP_L4.fcm" "$target/cpp_$name.fcm"
  if [[ "$3" == yes ]]; then
    "$PY" "$INSTRUMENT" "$SOURCE" "$target/MY_SRC/ldftra.F90"
  fi
  touch "$target/MY_SRC/"*.F90
  (cd "$NEMO_ROOT" && ./makenemo -n "$name" -m conda-scalarmath -y --clean)
  (cd "$NEMO_ROOT" && ./makenemo -n "$name" -m conda-scalarmath del_key key_xios)
  [[ -x "$target/BLD/bin/nemo.exe" ]] || fail "$name produced no executable"
}

build_targets() {
  build_one "$REF_TARGET_NAME" "$REF_BASE" no
  build_one "$TARGET_NAME" "$BASE" yes
  ! grep -Rqs 'oracle_l3_eiv' "$REF_TARGET/BLD/ppsrc/nemo" \
    || fail 'writer-free reference contains the L3 observer'
  grep -Fq 'oracle_l3_eiv_kt00000001.bin' "$TARGET/BLD/ppsrc/nemo/ldftra.f90" \
    || fail 'built source lacks L3 observer'
}

completed_steps() {
  local t rc
  t=$(tr -d '[:space:]' <"$1/time.step"); rc=$(<"$1/run.exit")
  if [[ "$rc" -eq 0 && "$t" == 100 ]]; then printf 100; return; fi
  grep -q 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$1/ocean.output" \
    && [[ -f "$1/output.abort.nc" ]] || fail "reference stopped outside stp_ctl at kt=$t"
  printf '%s' $((t - 1))
}

admit() {
  local completed steps name=oracle_rktracer_stage3_kt00000001.bin dir step restart
  audit_run "$TARGET/BLD/bin/nemo.exe" "$SMOKE" 2 "$(pad10 2)" done \
    || fail 'smoke metadata, deck, log, binary, inputs, or completion differ'
  audit_run "$REF_TARGET/BLD/bin/nemo.exe" "$REF" 100 \
    '2,4,6,8,10,12,14,16,18,20' boundary \
    || fail 'writer-free boundary metadata, deck, log, binary, or inputs differ'
  completed=$(completed_steps "$REF")
  steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
  ((steps >= 2)) || fail "NEMO completed only $completed steps"
  audit_run "$TARGET/BLD/bin/nemo.exe" "$TWIN_A" "$steps" "$(list_a "$steps")" done \
    || fail 'twin A metadata, deck, log, binary, inputs, or final step differ'
  audit_run "$TARGET/BLD/bin/nemo.exe" "$TWIN_B" "$steps" "$(list_b "$steps")" done \
    || fail 'twin B metadata, deck, log, binary, inputs, or final step differ'
  audit_run "$TARGET/BLD/bin/nemo.exe" "$TWIN_C" "$steps" "$(list_c "$steps")" done \
    || fail 'twin C metadata, deck, log, binary, inputs, or final step differ'
  for dir in "$SMOKE" "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
    "$PY" "$READER" --check "$dir/oracle_l3_eiv_kt00000001.bin" \
      --mesh "$dir/mesh_mask.nc" \
      || fail "$dir has no acting EIV witness"
    [[ -s "$dir/$name" ]] || fail "$dir lacks the inherited stage-3 record"
  done
  for dir in "$TWIN_B" "$TWIN_C"; do
    "$PY" "$READER" --compare-defined \
      "$TWIN_A/oracle_l3_eiv_kt00000001.bin" \
      "$dir/oracle_l3_eiv_kt00000001.bin" --mesh "$TWIN_A/mesh_mask.nc" \
      || fail 'twins differ on the defined EIV domain'
    cmp -s "$TWIN_A/$name" "$dir/$name" || fail 'twins differ'
  done
  cmp -s "$TWIN_A/$name" "$G9_RECORD/$name" \
    && fail 'L3 stage-3 tracer record equals G9 baseline'
  cmp -s "$TWIN_A/$name" "$L2_RECORD/$name" \
    && fail 'L3 stage-3 tracer record equals L2 baseline'
  for step in $(seq 2 "$steps"); do
    restart=$(printf 'ORCA2G1_%08d_restart.nc' "$step")
    [[ -f "$TWIN_A/$restart" || -f "$TWIN_B/$restart" \
       || -f "$TWIN_C/$restart" ]] || fail "no twin wrote restart step $step"
  done
  restart=$(printf 'ORCA2G1_%08d_restart.nc' "$steps")
  [[ -f "$REF/$restart" ]] || fail "writer-free reference lacks $restart"
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$(git -C "$repo" rev-parse HEAD)" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$DECK" "$ICE_DECK" "$REF_TARGET/BLD/bin/nemo.exe" \
    "$TARGET/BLD/bin/nemo.exe" \
    >"$EVIDENCE/manifest/producers.sha256"
  printf 'L3_ACQUISITION_PASS twins=3 steps=1..%s reference_completed=%s evidence=%s\n' \
    "$steps" "$(completed_steps "$REF")" "$EVIDENCE"
}

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac
preflight
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD_RUN build writer-free=%s record=%s; smoke, boundary, twins A/B/C into %s\n' \
    "$REF_TARGET_NAME" "$TARGET_NAME" "$EVIDENCE"
  exit 0
fi
if [[ "$mode" == --admit-existing ]]; then
  admit
  exit 0
fi
build_targets
mkdir -p "$EVIDENCE"
run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE" 2 "$(pad10 2)" done
run_one "$REF_TARGET/BLD/bin/nemo.exe" "$REF" 100 \
  '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps "$REF")
steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
((steps >= 2)) || fail "NEMO completed only $completed steps"
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_C" "$steps" "$(list_c "$steps")" done
admit
