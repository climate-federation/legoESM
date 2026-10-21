#!/usr/bin/env bash
# L7 acquisition: certified V5 plus ORCA2 river-mouth heat diffusivity.
# Default mode is a read-only seven-item preflight.  --run and
# --admit-existing are operator-only and require L7_OPERATOR_ACK.
set -Eeuo pipefail

fail() { printf 'REFUSE: L7 acquisition: %s\n' "$*" >&2; exit 66; }
scratch=
cleanup() { if [[ -n "$scratch" && -d "$scratch" ]]; then rm -rf -- "$scratch"; fi; }
trap cleanup EXIT
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in a worktree'
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_CFG=ORCA2_ICE_PISCES TARGET_NAME=ORCA2L7_RNF_R22
readonly TARGET=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly BASE=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly ORCA2=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4
readonly SOURCE=$NEMO_ROOT/src/OCE/ZDF/zdfphy.F90
readonly PP_SOURCE=$ORCA2/BLD/ppsrc/nemo/zdfphy.f90
readonly REF_BIN=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe
readonly REF_BIN_SHA=e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7
readonly INSTRUMENT=$here/instrument_zdfphy.py READER=$here/l7_records.py
readonly REAL_LOG=$here/ocean.output.excerpt
readonly REAL_LOG_SHA=efe1e074fda531cb3f3d6d5baf13fc42eb8b893bbd55a3d063ea456f918f9447
readonly BASE_DECK=$here/../nemo_testcase_v5_orca2havtb/namelist_cfg
readonly ICE_DECK=$here/../nemo_testcase_v5_orca2havtb/namelist_ice_cfg
readonly BASE_DECK_SHA=3358b329bfee1330efad05f05c05fe314a3837e6e9195b2d53a2df88ab188102
readonly ICE_DECK_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
# Smoke target deck (itend=2, stock=2) stamp.  Preflight renders and checks it.
readonly STAMPED_TARGET_DECK_SHA=840372df87518b7ae3e7fd6a83318ddb4a833139f72fa5021fdbcd8ae05408c3
readonly G5_TOOL=$here/../nemo_testcase_g5_orca2ice/run.sh
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly IWM_FILE=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/zdfiwm_forcing_orca2.nc
readonly IWM_FILE_SHA=f4279899be77119fda4f9e20b3d5e3a70e4e478fe5ead0dc661656a1bba296c2
readonly ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round22/oracle_l7_r22}
readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_boundary
readonly TWIN_A=$EVIDENCE/twin_a TWIN_B=$EVIDENCE/twin_b TWIN_C=$EVIDENCE/twin_c
readonly RECORD=oracle_l7_rnf_kt00000001.bin FSBC=2
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
ref_is_listed() { awk -v n="$1" '$1==n {found=1} END{exit !found}' "$NEMO_ROOT/cfgs/ref_cfgs.txt"; }
require_absent() { [[ ! -e "$1" ]]; }
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
    if len(hits) != 1: raise SystemExit(f"{name}: {hits}")
    return hits[0]
fsbc, it000, itend, stock = (int(one(n)) for n in ("nn_fsbc", "nn_it000", "nn_itend", "nn_stock"))
rst = one("ln_rst_list", r"\.(?:true|false)\.").lower() == ".true."
if fsbc != 2 or (itend-it000+1) % fsbc or (not rst and stock % fsbc): raise SystemExit(1)
PY
}

render_deck() { # destination true|false itend stock-list
  local logical=.$2.
  sed -E -e "s/^( *ln_rnf_mouth *= *).*/\1${logical}/" \
    -e "s/^( *nn_itend *= *).*/\1$3/" \
    -e "s/^( *nn_stock *= *).*/\1$3/" \
    -e "s/^( *nn_stocklist *= *).*/\1$4/" "$BASE_DECK" >"$1"
  [[ $(grep -E -c "^ *ln_rnf_mouth *= *${logical}$" "$1") -eq 1 ]] \
    || fail 'river-mouth switch did not render once'
  check_namelist_clock "$1" || fail "bad rendered deck $1"
}

check_values() { # whitespace-tolerant numeric/logical audit: log or deck
  "$PY" - "$1" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
def numbers(name):
    return [float(x.replace("D","E").replace("d","e")) for x in
            re.findall(rf"(?im)\b{re.escape(name)}\s*=\s*({number})", text)]
for name, want in {"rn_avt_rnf": 1.e-3, "rn_hrnf": 15.}.items():
    got = numbers(name)
    if not got or any(value != want for value in got): raise SystemExit(f"{name}: {got}")
logical = re.findall(r"(?im)\bln_rnf_mouth\s*=\s*(?:\.?(true|false)\.?|([TF]))\s*$", text)
if not logical or any((a or b).lower() not in ("true", "t") for a,b in logical):
    raise SystemExit(f"ln_rnf_mouth: {logical}")
PY
}

check_log() {
  local out=$1/ocean.output
  check_values "$out" || return 1
  grep -E 'by +1\.0+E-003 +m2/s +over +2 +w-levels' "$out" >/dev/null \
    && grep -E 'read +socoefr0 +\(rec: +1\) in \./runoff_core_monthly\.nc ok' "$out" >/dev/null
}

probe_output_parent() {
  local dir=$1
  mkdir -p "$dir" 2>/dev/null || return 1
  : >"$dir/.l7-write-probe" 2>/dev/null || return 1
  rm -f "$dir/.l7-write-probe"
}

stage_metadata() { # binary dir on/off itend stock
  local binary=$1 dir=$2 f
  mkdir -p "$dir"
  for f in "${INPUT_FILES[@]}"; do ln -s "$INPUT_ROOT/$f" "$dir/$f"; done
  ln -s "$IWM_FILE" "$dir/${IWM_FILE##*/}"
  ln -s "$ICE_REF" "$dir/namelist_ice_ref"; ln -s "$ICE_DECK" "$dir/namelist_ice_cfg"
  ln -s "$binary" "$dir/nemo"; render_deck "$dir/namelist_cfg" "$3" "$4" "$5"
  cp "$REAL_LOG" "$dir/ocean.output"; printf '0\n' >"$dir/run.exit"; printf '%s\n' "$4" >"$dir/time.step"
}

audit_run() { # binary dir on/off itend stock done|boundary instrumented|plain
  local binary=$1 dir=$2 on=$3 itend=$4 stock=$5 completion=$6 observer=$7 expected f
  expected=$scratch/audit-$(basename "$dir")
  render_deck "$expected" "$on" "$itend" "$stock"
  cmp -s "$expected" "$dir/namelist_cfg" || return 1
  cmp -s "$ICE_REF" "$dir/namelist_ice_ref" || return 1
  cmp -s "$ICE_DECK" "$dir/namelist_ice_cfg" || return 1
  cmp -s "$binary" "$dir/nemo" || return 1
  for f in "${INPUT_FILES[@]}"; do cmp -s "$INPUT_ROOT/$f" "$dir/$f" || return 1; done
  cmp -s "$IWM_FILE" "$dir/${IWM_FILE##*/}" || return 1
  if [[ "$on" == true ]]; then check_log "$dir" || return 1; fi
  if [[ "$observer" == instrumented ]]; then "$PY" "$READER" --check "$dir/$RECORD" || return 1; fi
  if [[ "$completion" == done ]]; then
    [[ $(tr -d '[:space:]' <"$dir/run.exit") == 0 \
       && $(tr -d '[:space:]' <"$dir/time.step") == "$itend" ]]
  else completed_steps "$dir" >/dev/null; fi
}

preflight() {
  local f want target_deck base_off bad
  [[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] \
    || fail 'commit the exact acquisition tool before running it'
  for f in "$SOURCE" "$PP_SOURCE" "$INSTRUMENT" "$READER" "$REAL_LOG" \
           "$BASE_DECK" "$ICE_DECK" "$G5_TOOL" "$ICE_REF" "$REF_BIN" "$FC"; do
    [[ -f "$f" ]] || fail "missing $f"
  done
  [[ $(sha "$REAL_LOG") == "$REAL_LOG_SHA" ]] || fail 'real log excerpt changed'
  [[ $(sha "$BASE_DECK") == "$BASE_DECK_SHA" ]] || fail 'V5 base deck changed'
  [[ $(sha "$ICE_DECK") == "$ICE_DECK_SHA" ]] || fail 'ice deck changed'
  [[ $(sha "$REF_BIN") == "$REF_BIN_SHA" ]] || fail 'writer-free binary changed'
  [[ $(sha "$IWM_FILE") == "$IWM_FILE_SHA" ]] || fail 'IWM input changed'
  for f in "${INPUT_FILES[@]}"; do
    want=$(input_sha "$f"); [[ ${#want} == 64 ]] || fail "no input SHA for $f"
    [[ -f "$INPUT_ROOT/$f" && $(sha "$INPUT_ROOT/$f") == "$want" ]] || fail "changed input $f"
  done
  scratch=$(mktemp -d /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/l7-r22-preflight.XXXXXX)

  # 1. Exact source transform, continuation/order controls, compiler syntax.
  "$PY" "$INSTRUMENT" "$SOURCE" "$scratch/zdfphy.f90"
  "$FC" -cpp -ffree-line-length-none -fsyntax-only -I"$ORCA2/BLD/inc" -J"$scratch" \
    "$scratch/zdfphy.f90" || fail 'instrumented zdfphy syntax failed'
  for bad in --plant-after-continuation --plant-short-literal --plant-raw-observer --plant-after-evd; do
    if "$PY" "$INSTRUMENT" "$bad" "$SOURCE" "$scratch/plant.f90" >/dev/null 2>&1; then
      fail "$bad plant was accepted"
    fi
  done

  # 2. makenemo -r must name an actual first-token reference configuration.
  ref_is_listed "$REF_CFG" || fail "$REF_CFG is not in ref_cfgs.txt"
  if ref_is_listed ORCA2_L7_NOT_A_REFERENCE; then fail 'bad -r plant was accepted'; fi

  # 3. Render all clocks; prove the target differs from its V5 base only by L7.
  target_deck=$scratch/target-smoke; base_off=$scratch/v5-off-smoke
  render_deck "$target_deck" true 2 "$(pad10 2)"
  render_deck "$base_off" false 2 "$(pad10 2)"
  [[ $(sha "$target_deck") == "$STAMPED_TARGET_DECK_SHA" ]] || fail 'stamped target deck changed'
  "$PY" - "$base_off" "$target_deck" <<'PY'
import re, sys
a, b = (open(p).read().splitlines() for p in sys.argv[1:])
changed = [(x,y) for x,y in zip(a,b,strict=True) if x != y]
if len(changed) != 1 or not all(re.search(r"ln_rnf_mouth", x) for pair in changed for x in pair):
    raise SystemExit(f"deck delta is not exactly ln_rnf_mouth: {changed}")
PY
  render_deck "$scratch/ref" false 100 '2,4,6,8,10,12,14,16,18,20'
  for f in 2 4 6 8 10; do
    render_deck "$scratch/a$f" true "$f" "$(list_a "$f")"
    render_deck "$scratch/b$f" true "$f" "$(list_b "$f")"
    render_deck "$scratch/c$f" true "$f" "$(list_c "$f")"
  done
  sed -E 's/^( *nn_itend *= *).*/\1 3/' "$target_deck" >"$scratch/bad-clock"
  if check_namelist_clock "$scratch/bad-clock"; then fail 'odd run-length plant accepted'; fi

  # 4. Exact binary layout and acting-control selftest.
  "$PY" "$READER" --selftest || fail 'record reader selftest failed'

  # 5. Numeric/logical checks run on a real committed excerpt and stamped deck.
  check_values "$REAL_LOG" || fail 'real committed ocean.output excerpt refused'
  check_values "$target_deck" || fail 'stamped rendered deck refused'
  sed 's/1\.0000000000000000E-003/1.0000000000000002E-003/' "$REAL_LOG" >"$scratch/bad-log"
  if check_values "$scratch/bad-log"; then fail 'nearby numeric log plant accepted'; fi

  # 6. An absent evidence parent is writable; admission rechecks every input.
  probe_output_parent "$scratch/new/absent/evidence" || fail 'absent output parent unusable'
  : >"$scratch/file-parent"
  if probe_output_parent "$scratch/file-parent/child"; then fail 'file-parent plant accepted'; fi
  stage_metadata "$FC" "$scratch/admit" true 2 "$(pad10 2)"
  "$PY" "$READER" --selftest
  # Metadata audit deliberately omits the observer here; real admission requires it.
  cp "$scratch/target-smoke" "$scratch/admit/namelist_cfg"
  if ! cmp -s "$scratch/target-smoke" "$scratch/admit/namelist_cfg"; then fail 'valid deck plant refused'; fi
  sed -i 's/ln_rnf_mouth = .true./ln_rnf_mouth = .false./' "$scratch/admit/namelist_cfg"
  if cmp -s "$scratch/target-smoke" "$scratch/admit/namelist_cfg"; then fail 'altered deck plant accepted'; fi

  # 7. Target config and evidence root are absent unless explicitly admitting.
  if [[ "$mode" != --admit-existing ]]; then
    require_absent "$TARGET" || fail "target config exists: $TARGET"
    require_absent "$EVIDENCE" || fail "evidence exists: $EVIDENCE"
    mkdir "$scratch/existing-target"
    if require_absent "$scratch/existing-target"; then fail 'existing-target plant accepted'; fi
  else
    [[ -x "$TARGET/BLD/bin/nemo.exe" && -d "$EVIDENCE" ]] || fail 'existing acquisition incomplete'
  fi
  printf 'L7_PREFLIGHT_OK checklist=7 plants=all_refused commit=%s\n' "$(git -C "$repo" rev-parse HEAD)"
}

stage_run() { # binary dir true|false itend stock
  local f dir=$2
  require_absent "$dir" || fail "run directory exists: $dir"
  mkdir -p "$dir"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$dir/$f"; done
  cp "$IWM_FILE" "$dir/${IWM_FILE##*/}"
  cp "$ICE_REF" "$dir/namelist_ice_ref"; cp "$ICE_DECK" "$dir/namelist_ice_cfg"
  render_deck "$dir/namelist_cfg" "$3" "$4" "$5"; cp "$1" "$dir/nemo"
}

run_one() { # binary dir true|false itend stock done|boundary
  local rc=0
  stage_run "$1" "$2" "$3" "$4" "$5"
  (cd "$2"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$2/run.exit"
  [[ -f "$2/ocean.output" && -f "$2/time.step" ]] || fail "$2 produced no NEMO log"
  if [[ "$3" == true ]]; then check_log "$2" || fail "$2 did not execute L7 selection"; fi
  if [[ "$6" == done ]]; then
    [[ $rc -eq 0 && $(tr -d '[:space:]' <"$2/time.step") == "$4" ]] \
      || fail "$2 did not complete $4 steps"
  fi
}

build_target() {
  (cd "$NEMO_ROOT" && ./makenemo -r "$REF_CFG" -n "$TARGET_NAME" -m conda-scalarmath del_key key_xios)
  cp -a "$BASE/MY_SRC/." "$TARGET/MY_SRC/"
  cp "$ORCA2/cpp_ORCA2_OMIP_L4.fcm" "$TARGET/cpp_$TARGET_NAME.fcm"
  "$PY" "$INSTRUMENT" "$SOURCE" "$TARGET/MY_SRC/zdfphy.F90"
  touch "$TARGET/MY_SRC/"*.F90
  (cd "$NEMO_ROOT" && ./makenemo -n "$TARGET_NAME" -m conda-scalarmath -y --clean)
  (cd "$NEMO_ROOT" && ./makenemo -n "$TARGET_NAME" -m conda-scalarmath del_key key_xios)
  [[ -x "$TARGET/BLD/bin/nemo.exe" ]] || fail 'target produced no executable'
  grep -F 'oracle_l7_rnf_kt00000001.bin' "$TARGET/BLD/ppsrc/nemo/zdfphy.f90" >/dev/null \
    || fail 'built source lacks L7 observer'
}

completed_steps() {
  local t rc; t=$(tr -d '[:space:]' <"$1/time.step"); rc=$(<"$1/run.exit")
  if [[ "$rc" -eq 0 && "$t" == 100 ]]; then printf 100; return; fi
  grep -F 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$1/ocean.output" >/dev/null \
    && [[ -f "$1/output.abort.nc" ]] || fail "reference stopped outside stp_ctl at kt=$t"
  printf '%s' $((t - 1))
}

admit() {
  local completed steps dir binary=$TARGET/BLD/bin/nemo.exe
  audit_run "$binary" "$SMOKE" true 2 "$(pad10 2)" done instrumented || fail 'smoke audit failed'
  audit_run "$REF_BIN" "$REF" false 100 '2,4,6,8,10,12,14,16,18,20' boundary plain || fail 'boundary audit failed'
  completed=$(completed_steps "$REF"); steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
  ((steps >= 2 && steps <= 10)) || fail "invalid twin length $steps"
  audit_run "$binary" "$TWIN_A" true "$steps" "$(list_a "$steps")" done instrumented || fail 'twin A audit failed'
  audit_run "$binary" "$TWIN_B" true "$steps" "$(list_b "$steps")" done instrumented || fail 'twin B audit failed'
  audit_run "$binary" "$TWIN_C" true "$steps" "$(list_c "$steps")" done instrumented || fail 'twin C audit failed'
  "$PY" "$READER" --compare-defined "$TWIN_A/$RECORD" "$TWIN_B/$RECORD"
  "$PY" "$READER" --compare-defined "$TWIN_A/$RECORD" "$TWIN_C/$RECORD"
  # A separate smoke-only claim.  With the admitted all-zero socoefr0 this is
  # expected to refuse; it must never be weakened into a non-acting pass.
  "$PY" "$READER" --check-acting "$SMOKE/$RECORD" \
    || fail 'smoke check_acting found no river-mouth contribution'
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$(git -C "$repo" rev-parse HEAD)" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$BASE_DECK" "$ICE_DECK" "$REF_BIN" "$binary" >"$EVIDENCE/manifest/producers.sha256"
  printf 'L7_ACQUISITION_PASS twins=3 steps=%s reference_completed=%s evidence=%s\n' \
    "$steps" "$completed" "$EVIDENCE"
}

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac
preflight
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD_RUN smoke -> writer-free boundary -> twins A/B/C (<=10); smoke-only check_acting; %s\n' "$EVIDENCE"
  exit 0
fi
[[ ${L7_OPERATOR_ACK:-} == I_ACKNOWLEDGE_NEMO_BUILD_AND_RUN ]] \
  || fail 'operator acknowledgement required: L7_OPERATOR_ACK=I_ACKNOWLEDGE_NEMO_BUILD_AND_RUN'
if [[ "$mode" == --admit-existing ]]; then admit; exit 0; fi
build_target; mkdir -p "$EVIDENCE"
run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE" true 2 "$(pad10 2)" done
run_one "$REF_BIN" "$REF" false 100 '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps "$REF"); steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
((steps >= 2 && steps <= 10)) || fail "invalid twin length $steps"
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_A" true "$steps" "$(list_a "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_B" true "$steps" "$(list_b "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_C" true "$steps" "$(list_c "$steps")" done
admit
