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
readonly TARGET=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly BASE=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
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
readonly TWIN_A=$EVIDENCE/twin_a TWIN_B=$EVIDENCE/twin_b TWIN_C=$EVIDENCE/twin_c
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

check_namelist_clock() {
  "$PY" - "$1" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
def one(name):
    hits = re.findall(rf"^\s*{name}\s*=\s*([0-9]+)", text, re.M | re.I)
    if len(hits) != 1: raise SystemExit(1)
    return int(hits[0])
fsbc, itend, stock = one("nn_fsbc"), one("nn_itend"), one("nn_stock")
line = re.findall(r"^\s*nn_stocklist\s*=\s*([^!\n]+)", text, re.M | re.I)
if len(line) != 1: raise SystemExit(1)
stocklist = [int(x) for x in re.findall(r"[0-9]+", line[0]) if int(x)]
if fsbc != 2 or not stocklist or any(x % fsbc for x in [itend, stock, *stocklist]):
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

  # 3: render a two-step deck and enforce every restart time modulo nn_fsbc=2.
  rendered=$tmp/namelist_cfg
  sed -E -e 's/^(   nn_itend    = ).*/\1      2/' \
    -e 's/^(   nn_stock    = ).*/\1      2/' \
    -e 's/^(   nn_stocklist = ).*/\1 2,0,0,0,0,0,0,0,0,0/' "$DECK" >"$rendered"
  check_namelist_clock "$rendered" || fail 'rendered clock violates init rules'
  sed 's/nn_stock    =      2/nn_stock    =      1/' "$rendered" >"$tmp/bad-clock"
  if check_namelist_clock "$tmp/bad-clock"; then fail 'bad clock plant was accepted'; fi

  # 4: reader/writer byte identity, truncation/tag plants, and 16-byte literals.
  "$PY" "$READER" --selftest || fail 'record layout selftest failed'
  if "$PY" "$INSTRUMENT" --plant-short-literal "$PP_SOURCE" \
       "$tmp/short-literal.f90" >/dev/null 2>&1; then
    fail 'short character-literal plant was accepted'
  fi

  # 5: all six declared frames have a writer; deleting one must be refused.
  check_frame_writes "$tmp/ldftra.f90" || fail 'expected writer frame missing'
  sed "/WRITE(l3_unit) 'DELTA_V/d" "$tmp/ldftra.f90" >"$tmp/missing-frame.f90"
  if check_frame_writes "$tmp/missing-frame.f90"; then fail 'missing-frame plant was accepted'; fi

  # 6: output creation works with absent parents; a regular-file parent fails.
  probe_output_parent "$tmp/new/absent/evidence" || fail 'absent output parent unusable'
  : >"$tmp/blocked-parent"
  if probe_output_parent "$tmp/blocked-parent/child"; then fail 'file-parent plant was accepted'; fi

  # 7: both configuration and run root must be absent; planted existing paths fail.
  require_absent "$TARGET" || fail "target config exists: $TARGET"
  require_absent "$EVIDENCE" || fail "run root exists: $EVIDENCE"
  mkdir "$tmp/existing-target" "$tmp/existing-run"
  if require_absent "$tmp/existing-target" || require_absent "$tmp/existing-run"; then
    fail 'existing target/run plant was accepted'
  fi
  rm -rf -- "$tmp"
  trap - RETURN
  printf 'L3_PREFLIGHT_OK checklist=7 plants=all_refused commit=%s\n' \
    "$(git -C "$repo" rev-parse HEAD)"
}

render_deck() { # destination, stock-list
  sed -E -e 's/^(   nn_itend    = ).*/\1      2/' \
    -e 's/^(   nn_stock    = ).*/\1      2/' \
    -e "s/^(   nn_stocklist = ).*/\1 $2/" "$DECK" >"$1"
  check_namelist_clock "$1" || fail "bad rendered deck $1"
}

stage_run() { # directory, stock-list
  local f
  require_absent "$1" || fail "run directory exists: $1"
  mkdir -p "$1"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$1/$f"; done
  cp "$ICE_REF" "$1/namelist_ice_ref"; cp "$ICE_DECK" "$1/namelist_ice_cfg"
  render_deck "$1/namelist_cfg" "$2"
  cp "$TARGET/BLD/bin/nemo.exe" "$1/nemo"
}

check_log() {
  local out=$1/ocean.output
  grep -Eq 'ln_traadv_fct += +T$' "$out" \
    && grep -Eq 'ln_traldf_iso += +T$' "$out" \
    && grep -Eq 'ln_traldf_msc += +T$' "$out" \
    && grep -Eq 'ln_ldfeiv += +T$' "$out" \
    && grep -Eq 'nn_aei_ijk_t += +21$' "$out" \
    && grep -Eq 'rn_Ue += +3\.000000000000000[0-9]*E-002' "$out" \
    && grep -Eq 'rn_Le += +200000\.00000000000' "$out" \
    && grep -Fq 'ldf_eiv_trp : eddy induced advection' "$out" \
    || fail "$1 did not execute exact L3 selection"
}

run_one() {
  local rc=0
  stage_run "$1" "$2"
  (cd "$1"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  [[ $rc -eq 0 && -f "$1/time.step" \
     && "$(tr -d '[:space:]' <"$1/time.step")" == 2 ]] \
    || fail "$1 did not complete two steps"
  check_log "$1"
  "$PY" "$READER" --check "$1/oracle_l3_eiv_kt00000001.bin" \
    || fail "$1 has no acting EIV witness"
  [[ -s "$1/oracle_rktracer_stage3_kt00000001.bin" ]] \
    || fail "$1 lacks the inherited stage-3 tracer record"
}

build_target() {
  (cd "$NEMO_ROOT" && ./makenemo -r "$REF_CFG" -n "$TARGET_NAME" \
    -m conda-scalarmath del_key key_xios)
  cp -a "$BASE/MY_SRC/." "$TARGET/MY_SRC/"
  cp "$ORCA2/cpp_ORCA2_OMIP_L4.fcm" "$TARGET/cpp_$TARGET_NAME.fcm"
  "$PY" "$INSTRUMENT" "$SOURCE" "$TARGET/MY_SRC/ldftra.F90"
  touch "$TARGET/MY_SRC/"*.F90
  (cd "$NEMO_ROOT" && ./makenemo -n "$TARGET_NAME" -m conda-scalarmath -y --clean)
  (cd "$NEMO_ROOT" && ./makenemo -n "$TARGET_NAME" -m conda-scalarmath del_key key_xios)
  [[ -x "$TARGET/BLD/bin/nemo.exe" ]] || fail 'build produced no executable'
  grep -Fq 'oracle_l3_eiv_kt00000001.bin' "$TARGET/BLD/ppsrc/nemo/ldftra.f90" \
    || fail 'built source lacks L3 observer'
}

admit() {
  local name=oracle_rktracer_stage3_kt00000001.bin dir
  for dir in "$TWIN_B" "$TWIN_C"; do
    cmp -s "$TWIN_A/oracle_l3_eiv_kt00000001.bin" "$dir/oracle_l3_eiv_kt00000001.bin" \
      && cmp -s "$TWIN_A/$name" "$dir/$name" || fail 'twins differ'
  done
  cmp -s "$TWIN_A/$name" "$G9_RECORD/$name" \
    && fail 'L3 stage-3 tracer record equals G9 baseline'
  cmp -s "$TWIN_A/$name" "$L2_RECORD/$name" \
    && fail 'L3 stage-3 tracer record equals L2 baseline'
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$(git -C "$repo" rev-parse HEAD)" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$DECK" "$ICE_DECK" "$TARGET/BLD/bin/nemo.exe" \
    >"$EVIDENCE/manifest/producers.sha256"
  printf 'L3_ACQUISITION_PASS twins=3 steps=2 evidence=%s\n' "$EVIDENCE"
}

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run) ;; *) exit 64 ;; esac
preflight
[[ "$mode" == --run ]] || {
  printf 'WOULD_RUN makenemo -r %s -n %s and three 2-step twins into %s\n' \
    "$REF_CFG" "$TARGET_NAME" "$EVIDENCE"
  exit 0
}
build_target
mkdir -p "$EVIDENCE"
run_one "$TWIN_A" '2,0,0,0,0,0,0,0,0,0'
run_one "$TWIN_B" '0,2,0,0,0,0,0,0,0,0'
run_one "$TWIN_C" '2,0,0,0,0,0,0,0,0,0'
admit
