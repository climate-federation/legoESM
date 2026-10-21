#!/usr/bin/env bash
# L5 acquisition: certified G9 plus ORCA2 mixed-layer eddies.  Default mode
# is read-only preflight; NEMO creation/build/run under --run is operator-only.
set -Eeuo pipefail

fail() { printf 'REFUSE: L5 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in a worktree'
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_CFG=ORCA2_ICE_PISCES TARGET_NAME=ORCA2L5_MLE_R21
readonly REF_TARGET_NAME=ORCA2L5_MLE_R21_REF
readonly TARGET=$NEMO_ROOT/cfgs/$TARGET_NAME REF_TARGET=$NEMO_ROOT/cfgs/$REF_TARGET_NAME
readonly BASE=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly REF_BASE=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31REF_FIX1
readonly ORCA2=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4
readonly SOURCE=$NEMO_ROOT/src/OCE/TRA/tramle.F90
readonly PP_SOURCE=$ORCA2/BLD/ppsrc/nemo/tramle.f90
readonly INSTRUMENT=$here/instrument_tramle.py
readonly READER=$here/../nemo_testcase_l3_orca2eiv/l3_records.py
readonly BASE_DECK=$here/../nemo_testcase_g9_orca2tadv/namelist_cfg
readonly ICE_DECK=$here/../nemo_testcase_g9_orca2tadv/namelist_ice_cfg
readonly BASE_DECK_SHA=570db996c34e98ccdabf3c367455c1ee6d0b4f18df9f3bdd7264cf053f1f0720
readonly ICE_DECK_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly G5_TOOL=$here/../nemo_testcase_g5_orca2ice/run.sh
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round21/oracle_l5_r21}
readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly TWIN_A=$EVIDENCE/twin_a TWIN_B=$EVIDENCE/twin_b TWIN_C=$EVIDENCE/twin_c
readonly INPUT_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc
  ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc q_10.15JUNE2009_fill.nc
  ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc
  sss_data.nc runoff_core_monthly.nc namelist_ref)
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly FSBC=2
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

sha() { sha256sum "$1" | awk '{print $1}'; }
input_sha() { sed -nE "s/^  \[$1\]=([0-9a-f]{64})$/\1/p" "$G5_TOOL"; }
require_absent() { [[ ! -e "$1" ]] || return 1; }
ref_is_listed() { awk -v n="$1" '$1==n {found=1} END{exit !found}' "$NEMO_ROOT/cfgs/ref_cfgs.txt"; }
pad10() { local a=("$@"); while ((${#a[@]} < 10)); do a+=(0); done; local IFS=,; printf '%s' "${a[*]}"; }
list_a() { pad10 "$1"; }
list_b() { pad10 $(seq 2 2 "$1"); }
list_c() { (($1 >= 3)) && pad10 $(seq 3 2 "$1") || pad10 0; }

check_namelist_clock() {
  "$PY" - "$1" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
def one(name, kind=r"[0-9]+"):
    hits = re.findall(rf"^\s*{name}\s*=\s*({kind})", text, re.M | re.I)
    if len(hits) != 1: raise SystemExit(1)
    return hits[0]
fsbc, it000, itend, stock = (int(one(n)) for n in ("nn_fsbc", "nn_it000", "nn_itend", "nn_stock"))
rst = one("ln_rst_list", r"\.(?:true|false)\.").lower() == ".true."
if fsbc != 2 or (itend-it000+1) % fsbc or (not rst and stock % fsbc): raise SystemExit(1)
PY
}

render_deck() { # destination, itend, stock-list
  sed -E -e 's/^(   ln_mle      = )\.false\./\1.true./' \
    -e "s/^(   nn_itend    = ).*/\1      $2/" \
    -e "s/^(   nn_stock    = ).*/\1      $2/" \
    -e "s/^(   nn_stocklist = ).*/\1 $3/" "$BASE_DECK" >"$1"
  [[ $(grep -F -c 'ln_mle      = .true.' "$1") -eq 1 ]] || fail 'MLE deck switch did not render once'
  check_namelist_clock "$1" || fail "bad rendered deck $1"
}

check_log_values() { # numeric, not textual, resolved namtra_mle audit
  "$PY" - "$1" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
expected = {"rn_ce": .06, "nn_mle": 1., "rn_lf": 5000., "rn_time": 172800.,
            "rn_lat": 20., "nn_mld_uv": 0., "nn_conv": 0., "rn_rho_c_mle": .01}
for name, want in expected.items():
    got = re.findall(rf"(?im)\b{re.escape(name)}\s*=\s*({number})", text)
    vals = [float(x.replace('D','E').replace('d','e')) for x in got]
    if not vals or any(value != want for value in vals):
        raise SystemExit(f"{name} differs: {got}")
PY
}

check_log() {
  local out=$1/ocean.output
  grep -Eq 'ln_traadv_fct += +T$' "$out" \
    && grep -Eq 'ln_mle += +T$' "$out" \
    && grep -Fq 'Mixed Layer Eddy induced transport added to tracer advection' "$out" \
    || return 1
  check_log_values "$out"
}

probe_output_parent() {
  local dir=$1
  mkdir -p "$dir" 2>/dev/null || return 1
  : >"$dir/.l5-write-probe" 2>/dev/null || return 1
  rm -f "$dir/.l5-write-probe"
}

plant_audit_log() {
  printf '%s\n' ' ln_traadv_fct = T' ' ln_mle = T' \
    ' Mixed Layer Eddy induced transport added to tracer advection' \
    ' rn_ce = 6.0000000000000000E-002' ' nn_mle = 1' ' rn_lf = 5000.0' \
    ' rn_time = 172800.0' ' rn_lat = 20.0' ' nn_mld_uv = 0' ' nn_conv = 0' \
    ' rn_rho_c_mle = 1.0000000000000000E-002'
}

stage_metadata() { # binary dir itend stock (symlink inputs: preflight only)
  local binary=$1 dir=$2 f
  mkdir -p "$dir"
  for f in "${INPUT_FILES[@]}"; do ln -s "$INPUT_ROOT/$f" "$dir/$f"; done
  ln -s "$ICE_REF" "$dir/namelist_ice_ref"; ln -s "$ICE_DECK" "$dir/namelist_ice_cfg"
  ln -s "$binary" "$dir/nemo"
  render_deck "$dir/namelist_cfg" "$3" "$4"
  plant_audit_log >"$dir/ocean.output"
  printf '0\n' >"$dir/run.exit"; printf '%s\n' "$3" >"$dir/time.step"
}

audit_run() { # binary dir itend stock done|boundary
  local binary=$1 dir=$2 itend=$3 stock=$4 completion=$5 expected f
  expected=$(mktemp /tmp/orca2l5-admit-deck.XXXXXX)
  render_deck "$expected" "$itend" "$stock"
  cmp -s "$expected" "$dir/namelist_cfg" || return 1; rm -f "$expected"
  cmp -s "$ICE_REF" "$dir/namelist_ice_ref" || return 1
  cmp -s "$ICE_DECK" "$dir/namelist_ice_cfg" || return 1
  cmp -s "$binary" "$dir/nemo" || return 1
  for f in "${INPUT_FILES[@]}"; do cmp -s "$INPUT_ROOT/$f" "$dir/$f" || return 1; done
  check_log "$dir" || return 1
  if [[ "$completion" == done ]]; then
    [[ $(tr -d '[:space:]' <"$dir/run.exit") == 0 \
       && $(tr -d '[:space:]' <"$dir/time.step") == "$itend" ]]
  else
    completed_steps "$dir" >/dev/null
  fi
}

preflight() {
  local tmp f want rendered
  [[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] \
    || fail 'commit the exact acquisition tool before running it'
  for f in "$SOURCE" "$PP_SOURCE" "$INSTRUMENT" "$READER" "$BASE_DECK" \
           "$ICE_DECK" "$G5_TOOL" "$ICE_REF" "$FC"; do [[ -f "$f" ]] || fail "missing $f"; done
  [[ $(sha "$BASE_DECK") == "$BASE_DECK_SHA" ]] || fail 'G9 base deck changed'
  [[ $(sha "$ICE_DECK") == "$ICE_DECK_SHA" ]] || fail 'ice deck changed'
  for f in "${INPUT_FILES[@]}"; do
    want=$(input_sha "$f"); [[ ${#want} == 64 ]] || fail "no input SHA for $f"
    [[ -f "$INPUT_ROOT/$f" && $(sha "$INPUT_ROOT/$f") == "$want" ]] || fail "missing or changed input $f"
  done
  tmp=$(mktemp -d /tmp/orca2l5-preflight.XXXXXX); trap 'rm -rf -- "$tmp"' RETURN

  # 1. Exact transform, continuation refusal, compiler syntax.
  "$PY" "$INSTRUMENT" "$SOURCE" "$tmp/tramle.f90"
  "$FC" -cpp -ffree-line-length-none -fsyntax-only -I"$ORCA2/BLD/inc" -J"$tmp" "$tmp/tramle.f90" \
    || fail 'instrumented tramle syntax failed'
  if "$PY" "$INSTRUMENT" --plant-after-continuation "$SOURCE" "$tmp/continued.f90" >/dev/null 2>&1; then
    fail "continued-'&' insertion plant was accepted"
  fi
  if "$PY" "$INSTRUMENT" --plant-nested-before "$SOURCE" "$tmp/nested.f90" >/dev/null 2>&1; then
    fail 'nested before-copy plant was accepted'
  fi

  # 2. makenemo reference must be the first token in ref_cfgs.txt.
  ref_is_listed "$REF_CFG" || fail "$REF_CFG is not in ref_cfgs.txt"
  if ref_is_listed ORCA2_L5_NOT_A_REFERENCE; then fail 'bad -r plant was accepted'; fi

  # 3. Every rendered run obeys NEMO's nn_fsbc clock rule.
  render_deck "$tmp/smoke" 2 "$(pad10 2)"; rendered=$tmp/smoke
  render_deck "$tmp/ref" 100 '2,4,6,8,10,12,14,16,18,20'
  for f in 2 4 6 8 10; do
    render_deck "$tmp/a$f" "$f" "$(list_a "$f")"
    render_deck "$tmp/b$f" "$f" "$(list_b "$f")"
    render_deck "$tmp/c$f" "$f" "$(list_c "$f")"
  done
  sed -E 's/^( *nn_itend *= *).*/\1 3/' "$rendered" >"$tmp/bad-clock"
  if check_namelist_clock "$tmp/bad-clock"; then fail 'odd run-length plant was accepted'; fi

  # 4. Record layout/canonicalization and observer-literal plants.
  "$PY" "$READER" --selftest || fail 'record layout selftest failed'
  if "$PY" "$INSTRUMENT" --plant-short-literal "$SOURCE" "$tmp/short.f90" >/dev/null 2>&1; then
    fail 'short character-literal plant was accepted'
  fi
  if "$PY" "$INSTRUMENT" --plant-raw-observer "$SOURCE" "$tmp/raw.f90" >/dev/null 2>&1; then
    fail 'raw undefined-cell observer plant was accepted'
  fi

  # 5. Numeric log parsing accepts NEMO formatting and rejects a nearby float.
  plant_audit_log >"$tmp/float-log"; check_log_values "$tmp/float-log" \
    || fail 'numeric MLE log plant was refused'
  sed 's/6\.0000000000000000E-002/6.0000000000000008E-002/' "$tmp/float-log" >"$tmp/bad-float-log"
  if check_log_values "$tmp/bad-float-log"; then fail 'unequal rn_ce plant was accepted'; fi

  # 6. Absent output parents work; completed-run admission rechecks all data.
  probe_output_parent "$tmp/new/absent/evidence" || fail 'absent output parent unusable'
  : >"$tmp/blocked-parent"
  if probe_output_parent "$tmp/blocked-parent/child"; then fail 'file-parent plant was accepted'; fi
  stage_metadata "$FC" "$tmp/admit" 2 "$(pad10 2)"
  audit_run "$FC" "$tmp/admit" 2 "$(pad10 2)" done || fail 'valid admission plant refused'
  sed -i 's/ln_mle      = .true./ln_mle      = .false./' "$tmp/admit/namelist_cfg"
  if audit_run "$FC" "$tmp/admit" 2 "$(pad10 2)" done; then fail 'altered deck plant accepted'; fi
  render_deck "$tmp/admit/namelist_cfg" 2 "$(pad10 2)"; sed -i '/rn_ce/d' "$tmp/admit/ocean.output"
  if audit_run "$FC" "$tmp/admit" 2 "$(pad10 2)" done; then fail 'altered log plant accepted'; fi
  plant_audit_log >"$tmp/admit/ocean.output"; printf '1\n' >"$tmp/admit/run.exit"
  if audit_run "$FC" "$tmp/admit" 2 "$(pad10 2)" done; then fail 'incomplete smoke plant accepted'; fi

  # 7. Both configurations and evidence root must be absent unless admitting.
  if [[ "$mode" != --admit-existing ]]; then
    require_absent "$TARGET" || fail "target config exists: $TARGET"
    require_absent "$REF_TARGET" || fail "reference config exists: $REF_TARGET"
    require_absent "$EVIDENCE" || fail "run root exists: $EVIDENCE"
    mkdir "$tmp/existing-target"; if require_absent "$tmp/existing-target"; then fail 'existing-target plant accepted'; fi
  else
    [[ -x "$TARGET/BLD/bin/nemo.exe" && -x "$REF_TARGET/BLD/bin/nemo.exe" && -d "$EVIDENCE" ]] \
      || fail 'existing acquisition is incomplete'
  fi
  rm -rf -- "$tmp"; trap - RETURN
  printf 'L5_PREFLIGHT_OK checklist=7 plants=all_refused commit=%s\n' "$(git -C "$repo" rev-parse HEAD)"
}

stage_run() { # binary dir itend stock
  local f dir=$2; require_absent "$dir" || fail "run directory exists: $dir"; mkdir -p "$dir"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$dir/$f"; done
  cp "$ICE_REF" "$dir/namelist_ice_ref"; cp "$ICE_DECK" "$dir/namelist_ice_cfg"
  render_deck "$dir/namelist_cfg" "$3" "$4"; cp "$1" "$dir/nemo"
}

run_one() { # binary dir itend stock done|boundary
  local rc=0; stage_run "$1" "$2" "$3" "$4"
  (cd "$2"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$2/run.exit"
  [[ -f "$2/ocean.output" && -f "$2/time.step" ]] || fail "$2 produced no NEMO log"
  check_log "$2" || fail "$2 did not execute exact L5 selection"
  if [[ "$5" == done ]]; then
    [[ $rc -eq 0 && $(tr -d '[:space:]' <"$2/time.step") == "$3" ]] || fail "$2 did not complete $3 steps"
  fi
}

build_one() { # name inherited instrument=yes|no
  local name=$1 inherited=$2 target=$NEMO_ROOT/cfgs/$1
  (cd "$NEMO_ROOT" && ./makenemo -r "$REF_CFG" -n "$name" -m conda-scalarmath del_key key_xios)
  cp -a "$inherited/MY_SRC/." "$target/MY_SRC/"; cp "$ORCA2/cpp_ORCA2_OMIP_L4.fcm" "$target/cpp_$name.fcm"
  if [[ "$3" == yes ]]; then "$PY" "$INSTRUMENT" "$SOURCE" "$target/MY_SRC/tramle.F90"; fi
  touch "$target/MY_SRC/"*.F90
  (cd "$NEMO_ROOT" && ./makenemo -n "$name" -m conda-scalarmath -y --clean)
  (cd "$NEMO_ROOT" && ./makenemo -n "$name" -m conda-scalarmath del_key key_xios)
  [[ -x "$target/BLD/bin/nemo.exe" ]] || fail "$name produced no executable"
}

build_targets() {
  build_one "$REF_TARGET_NAME" "$REF_BASE" no; build_one "$TARGET_NAME" "$BASE" yes
  ! grep -Rqs 'oracle_l5_mle' "$REF_TARGET/BLD/ppsrc/nemo" || fail 'writer-free reference contains observer'
  grep -Fq 'oracle_l5_mle_kt00000001.bin' "$TARGET/BLD/ppsrc/nemo/tramle.f90" || fail 'built source lacks observer'
}

completed_steps() {
  local t rc; t=$(tr -d '[:space:]' <"$1/time.step"); rc=$(<"$1/run.exit")
  if [[ "$rc" -eq 0 && "$t" == 100 ]]; then printf 100; return; fi
  grep -Fq 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$1/ocean.output" \
    && [[ -f "$1/output.abort.nc" ]] || fail "reference stopped outside stp_ctl at kt=$t"
  printf '%s' $((t - 1))
}

admit() {
  local completed steps dir record=oracle_l5_mle_kt00000001.bin
  audit_run "$TARGET/BLD/bin/nemo.exe" "$SMOKE" 2 "$(pad10 2)" done || fail 'smoke audit failed'
  audit_run "$REF_TARGET/BLD/bin/nemo.exe" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary || fail 'boundary audit failed'
  completed=$(completed_steps "$REF"); steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
  ((steps >= 2)) || fail "NEMO completed only $completed steps"
  audit_run "$TARGET/BLD/bin/nemo.exe" "$TWIN_A" "$steps" "$(list_a "$steps")" done || fail 'twin A audit failed'
  audit_run "$TARGET/BLD/bin/nemo.exe" "$TWIN_B" "$steps" "$(list_b "$steps")" done || fail 'twin B audit failed'
  audit_run "$TARGET/BLD/bin/nemo.exe" "$TWIN_C" "$steps" "$(list_c "$steps")" done || fail 'twin C audit failed'
  for dir in "$SMOKE" "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
    "$PY" "$READER" --check "$dir/$record" --mesh "$dir/mesh_mask.nc" || fail "$dir has no acting MLE witness"
  done
  for dir in "$TWIN_B" "$TWIN_C"; do
    "$PY" "$READER" --compare-defined "$TWIN_A/$record" "$dir/$record" --mesh "$TWIN_A/mesh_mask.nc" \
      || fail 'twins differ on defined MLE cells'
  done
  mkdir -p "$EVIDENCE/manifest"; printf '%s\n' "$(git -C "$repo" rev-parse HEAD)" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$BASE_DECK" "$ICE_DECK" "$REF_TARGET/BLD/bin/nemo.exe" "$TARGET/BLD/bin/nemo.exe" \
    >"$EVIDENCE/manifest/producers.sha256"
  printf 'L5_ACQUISITION_PASS twins=3 steps=1..%s reference_completed=%s evidence=%s\n' "$steps" "$completed" "$EVIDENCE"
}

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac
preflight
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD_RUN build writer-free=%s record=%s; smoke, boundary, twins A/B/C into %s\n' \
    "$REF_TARGET_NAME" "$TARGET_NAME" "$EVIDENCE"; exit 0
fi
if [[ "$mode" == --admit-existing ]]; then admit; exit 0; fi
build_targets; mkdir -p "$EVIDENCE"
run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE" 2 "$(pad10 2)" done
run_one "$REF_TARGET/BLD/bin/nemo.exe" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps "$REF"); steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
((steps >= 2)) || fail "NEMO completed only $completed steps"
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_C" "$steps" "$(list_c "$steps")" done
admit
