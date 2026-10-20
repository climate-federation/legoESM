#!/usr/bin/env bash
# Grid ladder L4 round-1 acquisition (operator executed only).
# Default is a read-only seven-item preflight.  --run performs smoke ->
# writer-free boundary -> three <=10-step instrumented twins.  --admit-existing
# validates an operator-produced acquisition without executing NEMO.
set -Eeuo pipefail

fail() { printf 'REFUSE: L4 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in worktree'
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly G5_TOOL=$here/../nemo_testcase_g5_orca2ice/run.sh
readonly G5_DECK=$here/../nemo_testcase_g5_orca2ice/namelist_cfg
readonly DECK=$here/namelist_cfg ICE_DECK=$here/namelist_ice_cfg
readonly PATCH=$here/trabbc_round53.patch CHECKER=$here/l4_records.py
readonly BASE_TRABBC=$NEMO_ROOT/src/OCE/TRA/trabbc.F90
readonly REC_SOURCE_NAME=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly REF_NAME=ORCA2G5_OMIP_L4_R31REF_FIX1
readonly TARGET_NAME=ORCA2L4_BBC_R53
readonly REC_SOURCE=$NEMO_ROOT/cfgs/$REC_SOURCE_NAME
readonly REF_BIN=$NEMO_ROOT/cfgs/$REF_NAME/BLD/bin/nemo.exe
readonly TARGET=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly QGH_ROOT=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0
readonly QGH_SOURCE=$QGH_ROOT/geothermal_heating.nc
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly ROUNDS=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds
readonly EVIDENCE=${EVIDENCE:-$ROUNDS/round53/oracle_l4_r53}
readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly TWIN_A=$EVIDENCE/twin_a TWIN_B=$EVIDENCE/twin_b TWIN_C=$EVIDENCE/twin_c
readonly FSBC=2
readonly G5_DECK_SHA=c57eeb83d8f59c5b1d44df1295a10f5f30d28537358c8d327fcecc3d538f569e
readonly L4_DECK_SHA=00c849085493cf0b8fcdb9ec2b0c3099096a8515a6741f8565b035d6aa4c879e
readonly ICE_DECK_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly PATCH_SHA=dd98dfca4051cec8dd4814edd71c3bc2717ad599af3a4a9b7e086495f2f68453
readonly CHECKER_SHA=2435c5a31ebdd2db250f3ff104c9e0834dd08a03bd83be1d0541e6cdb02f9d2e
readonly BASE_TRABBC_SHA=28fd28c51c0616d7429600318148bfccebefb458d0db81c2f48f688714b9c2ab
readonly REF_BIN_SHA=e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7
readonly QGH_SHA=e369762f068a7fbc1561be310fb8ded844233da37e6a0c6ffc10805e4996fffb
readonly QGH_BYTES=116499
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

check_deck_delta() {
  python -I - "$G5_DECK" "$DECK" <<'PY'
import re, sys
from pathlib import Path
def values(path):
    out = {}; group = None
    for raw in Path(path).read_text().splitlines():
        line = raw.split('!', 1)[0].strip()
        if line.startswith('&'): group = line[1:].split()[0].lower()
        elif line == '/': group = None
        elif group and '=' in line:
            key, value = line.split('=', 1)
            out[f'{group}.{key.strip().lower()}'] = ''.join(value.rstrip(',').split()).lower()
    return out
g5, l4 = values(sys.argv[1]), values(sys.argv[2])
changed = {k: (g5.get(k), l4.get(k)) for k in g5.keys() | l4.keys()
           if g5.get(k) != l4.get(k)}
want = {
 'nambbc.ln_trabbc': ('.false.', '.true.'),
 'nambbc.nn_geoflx': (None, '2'),
 'nambbc.cn_dir': (None, "'./'"),
 'nambbc.sn_qgh': (None, "'geothermal_heating.nc',-12.,'heatflow',.false.,.true.,'yearly','','',''"),
}
if changed != want: raise SystemExit(f'L4 deck delta is not BBC-only: {changed!r}')
PY
}

syntax_check() { # patched source, module-output directory
  cpp -Dkey_nosignedzero -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 \
    -P -traditional -I"$REC_SOURCE/WORK" "$1" | \
    mpif90 -c -cpp -fsyntax-only -x f95 -fdefault-real-8 -fcray-pointer \
      -ffree-line-length-none -fallow-argument-mismatch -I"$REC_SOURCE/BLD/inc" \
      -J"$2" - \
      >/dev/null 2>&1
}

ref_cfg_check() { grep -Eq "^$1([[:space:]]|$)" "$NEMO_ROOT/cfgs/ref_cfgs.txt"; }
schedule_check() { # itend stock fsbc
  (( $1 > 0 && $2 > 0 && $1 % $3 == 0 && $2 % $3 == 0 && $2 <= $1 ));
}
absent_check() { [[ ! -e "$1" ]]; }
path_check() { # a path whose parent may not exist; nearest ancestor must be writable
  local p=$1
  while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done
  [[ -d "$p" && -w "$p" ]] && (( $(df -Pk "$p" | awk 'NR==2 {print $4}') > 1048576 ))
}

write_synthetic() { # directory steps
  python -I - "$1" "$2" <<'PY'
import pathlib, struct, sys
import numpy as np
root, steps = pathlib.Path(sys.argv[1]), int(sys.argv[2]); root.mkdir(parents=True)
magic = b'NEMO_L4_BBC__1  '; jpi, jpj = 3, 2
qgh = np.array([1., 2., 0., 3., 0., 4.], dtype=np.float64)
e3t = np.array([10., 20., 0., 30., 0., 40.], dtype=np.float64)
trend = np.zeros_like(qgh); wet = e3t > 0.; trend[wet] = qgh[wet] / e3t[wet]
for kt in range(1, steps + 1):
    (root / f'oracle_geothermal_kt{kt:08d}.bin').write_bytes(
        magic + struct.pack('=8i', 1, kt, 2, 3, jpi, jpj, 64, 3)
        + qgh.tobytes() + e3t.tobytes() + trend.tobytes())
PY
}

seven_item_preflight() {
  local scratch=$repo/.l4-preflight-$$ bad
  absent_check "$scratch" || fail "preflight scratch exists: $scratch"
  mkdir "$scratch"
  trap 'rm -rf -- "$scratch"' RETURN

  # 1. The exact patched Fortran parses after NEMO preprocessing.  The patch
  # also cannot attach a declaration after a continued ampersand line.
  cp "$BASE_TRABBC" "$scratch/trabbc.F90"
  patch -s -d "$scratch" -p0 <"$PATCH"
  awk '
    /^[[:space:]]*(INTEGER|REAL|CHARACTER)/ && previous ~ /&[[:space:]]*$/ { bad=1 }
    NF { previous=$0 }
    END { exit bad }
  ' "$scratch/trabbc.F90" || fail 'writer declaration inserted after a continued statement'
  syntax_check "$scratch/trabbc.F90" "$scratch" || fail 'patched trabbc fails syntax proof'
  cp "$scratch/trabbc.F90" "$scratch/bad.F90"
  printf '\nTHIS IS NOT FORTRAN\n' >>"$scratch/bad.F90"
  ! syntax_check "$scratch/bad.F90" "$scratch" || fail 'syntax plant was not rejected'

  # 2. makenemo reference name is registered.
  ref_cfg_check ORCA2_ICE_PISCES || fail 'ORCA2_ICE_PISCES absent from ref_cfgs.txt'
  ! ref_cfg_check ROUND53_NOT_A_CONFIG || fail 'reference-name plant was not rejected'

  # 3. Every rendered stop/stock value respects nn_fsbc=2.
  schedule_check 2 2 "$FSBC" && schedule_check 100 20 "$FSBC" \
    && schedule_check 10 10 "$FSBC" || fail 'planned schedule violates nn_fsbc'
  ! schedule_check 9 8 "$FSBC" || fail 'schedule plant was not rejected'

  # 4. Reader and writer share the 16-byte magic, eight-int header, three
  # binary64 full fields and exact EOF.  A truncated real must fail.
  grep -Fq 'CHARACTER(len=16)' "$scratch/trabbc.F90" \
    && grep -Fq 'WRITE(l4_unit) 1, kt, Kmm, Krhs, jpi, jpj, STORAGE_SIZE(0._wp), 3' "$scratch/trabbc.F90" \
    || fail 'writer schema changed'
  write_synthetic "$scratch/layout" 1
  PYTHONPATH=$PYPATH python "$CHECKER" check-records "$scratch/layout" 1
  bad=$scratch/layout/oracle_geothermal_kt00000001.bin
  head -c $(( $(stat -c %s "$bad") - 8 )) "$bad" >"$scratch/truncated.bin"
  mv "$scratch/truncated.bin" "$bad"
  ! PYTHONPATH=$PYPATH python "$CHECKER" check-records "$scratch/layout" 1 \
    >/dev/null 2>&1 || fail 'layout plant was not rejected'

  # 5. Inventory is exactly one frame per expected step.
  rm -rf -- "$scratch/layout"; write_synthetic "$scratch/layout" 2
  PYTHONPATH=$PYPATH python "$CHECKER" check-records "$scratch/layout" 2
  rm -- "$scratch/layout/oracle_geothermal_kt00000002.bin"
  ! PYTHONPATH=$PYPATH python "$CHECKER" check-records "$scratch/layout" 2 \
    >/dev/null 2>&1 || fail 'missing-frame plant was not rejected'

  # 6. The output path is viable even though its evidence parent is absent.
  path_check "$EVIDENCE" || fail 'evidence path has no writable, spacious ancestor'
  ! path_check /proc/round53/not-writable || fail 'output-path plant was not rejected'

  # 7. A fresh run cannot collide with a target config or run directory.
  if [[ "$mode" != --admit-existing ]]; then
    absent_check "$TARGET" || fail "target config already exists: $TARGET"
    for bad in "$SMOKE" "$REF" "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
      absent_check "$bad" || fail "run directory already exists: $bad"
    done
  fi
  mkdir "$scratch/occupied"
  ! absent_check "$scratch/occupied" || fail 'existing-target plant was not rejected'
  trap - RETURN
  rm -rf -- "$scratch"
  printf 'L4_PREFLIGHT_7_OF_7_PASS plants=7/7 evidence_parent_absent=%s\n' \
    "$([[ ! -e "$EVIDENCE" ]] && printf true || printf false)"
}

preflight() {
  local f want
  [[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] \
    || fail 'commit the exact acquisition tool before running it'
  for f in "$G5_TOOL" "$G5_DECK" "$DECK" "$ICE_DECK" "$PATCH" "$CHECKER" \
           "$BASE_TRABBC" "$REF_BIN" "$REC_SOURCE/BLD/bin/nemo.exe" "$QGH_SOURCE"; do
    [[ -f "$f" ]] || fail "missing $f"
  done
  [[ "$(sha "$G5_DECK")" == "$G5_DECK_SHA" && "$(sha "$DECK")" == "$L4_DECK_SHA" \
     && "$(sha "$ICE_DECK")" == "$ICE_DECK_SHA" && "$(sha "$PATCH")" == "$PATCH_SHA" \
     && "$(sha "$CHECKER")" == "$CHECKER_SHA" && "$(sha "$BASE_TRABBC")" == "$BASE_TRABBC_SHA" \
     && "$(sha "$REF_BIN")" == "$REF_BIN_SHA" ]] || fail 'pinned producer changed'
  [[ "$(sha "$QGH_SOURCE")" == "$QGH_SHA" && "$(stat -c %s "$QGH_SOURCE")" == "$QGH_BYTES" ]] \
    || fail 'official geothermal input changed'
  for f in "${INPUT_FILES[@]}"; do
    want=$(input_sha "$f"); [[ ${#want} == 64 ]] || fail "G5 tool pins no SHA for $f"
    [[ -f "$INPUT_ROOT/$f" && "$(sha "$INPUT_ROOT/$f")" == "$want" ]] \
      || fail "missing or changed input $f"
  done
  check_deck_delta
  nm "$REF_BIN" | grep '__trabbc_MOD_tra_bbc$' >/dev/null || fail 'G5 reference binary lacks tra_bbc'
  seven_item_preflight
  printf 'L4_PREFLIGHT_OK commit=%s reference=%s record_target=%s\n' \
    "$(git -C "$repo" rev-parse HEAD)" "$REF_NAME" "$TARGET_NAME"
}

render_deck() { # target itend stocklist
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
         -e "s/^(   nn_stock    = ).*/\1    $2/" \
         -e "s/^(   nn_stocklist = ).*/\1$3/" "$DECK" >"$1"
}
pad10() { local a=("$@"); while ((${#a[@]} < 10)); do a+=(0); done; local IFS=,; printf '%s' "${a[*]}"; }
list_a() { pad10 "$1"; }
list_b() { pad10 $(seq 2 2 "$1"); }
list_c() { (($1 >= 3)) && pad10 $(seq 3 2 "$1") || pad10 0; }

build_target() {
  (cd "$NEMO_ROOT" && ./makenemo -r ORCA2_ICE_PISCES -n "$TARGET_NAME" \
    -m conda-scalarmath del_key key_xios)
  cp "$REC_SOURCE"/MY_SRC/* "$TARGET/MY_SRC/"
  cp "$REC_SOURCE/cpp_$REC_SOURCE_NAME.fcm" "$TARGET/cpp_$TARGET_NAME.fcm"
  cp "$BASE_TRABBC" "$TARGET/MY_SRC/trabbc.F90"
  patch -s -d "$TARGET/MY_SRC" -p0 <"$PATCH"
  (cd "$NEMO_ROOT" && ./makenemo -n "$TARGET_NAME" -m conda-scalarmath -j 8)
  nm "$TARGET/BLD/bin/nemo.exe" | grep '__trabbc_MOD_tra_bbc$' >/dev/null \
    || fail 'record binary lacks tra_bbc'
}

stage_run() { # binary directory itend stocklist
  local f
  absent_check "$2" || fail "run directory exists: $2"
  mkdir -p "$2"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$2/$f"; done
  cp "$QGH_SOURCE" "$2/geothermal_heating.nc"
  [[ "$(sha "$2/geothermal_heating.nc")" == "$QGH_SHA" ]] || fail 'copied qgh changed'
  cp "$ICE_DECK" "$2/namelist_ice_cfg"
  cp "$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref" "$2/namelist_ice_ref"
  render_deck "$2/namelist_cfg" "$3" "$4"
  cp "$1" "$2/nemo"
}

run_one() { # binary directory itend stocklist expected
  local rc=0
  stage_run "$1" "$2" "$3" "$4"
  (cd "$2"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$2/run.exit"
  [[ -f "$2/ocean.output" && -f "$2/time.step" ]] || fail "$2 produced no NEMO log"
  PYTHONPATH=$PYPATH python "$CHECKER" check-log "$2/ocean.output" "$2/namelist_cfg"
  if [[ "$5" == done ]]; then
    [[ "$rc" -eq 0 && "$(tr -d '[:space:]' <"$2/time.step")" == "$3" ]] \
      || fail "$2 did not complete $3 steps"
  fi
}

completed_steps() {
  local t rc
  t=$(tr -d '[:space:]' <"$REF/time.step"); rc=$(<"$REF/run.exit")
  if [[ "$rc" -eq 0 && "$t" == 100 ]]; then printf 100; return; fi
  grep -q 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$REF/ocean.output" \
    && [[ -f "$REF/output.abort.nc" ]] || fail "reference stopped outside stp_ctl at kt=$t"
  printf '%s' $((t - 1))
}

admit() { # steps
  local steps=$1 dir name rel
  for dir in "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
    PYTHONPATH=$PYPATH python "$CHECKER" check-log "$dir/ocean.output" "$dir/namelist_cfg"
    "$G5_TOOL" --check-surface-payload "$dir" "$steps" >/dev/null
  done
  PYTHONPATH=$PYPATH python "$CHECKER" admit "$steps" "$TWIN_A" "$TWIN_B" "$TWIN_C"
  while IFS= read -r rel; do
    [[ -f "$TWIN_B/$rel" && -f "$TWIN_C/$rel" ]] || fail "twin inventory lacks $rel"
    cmp -s "$TWIN_A/$rel" "$TWIN_B/$rel" && cmp -s "$TWIN_A/$rel" "$TWIN_C/$rel" \
      || fail "twins differ in $rel"
  done < <(cd "$TWIN_A" && find . -maxdepth 1 -type f -name 'oracle_*.bin' -printf '%P\n' | LC_ALL=C sort)
  mkdir -p "$EVIDENCE/manifest"
  sha256sum "$DECK" "$ICE_DECK" "$PATCH" "$CHECKER" "$QGH_SOURCE" \
    "$REF_BIN" "$TARGET/BLD/bin/nemo.exe" >"$EVIDENCE/manifest/producers.sha256"
  printf '%s\n' "$(git -C "$repo" rev-parse HEAD)" >"$EVIDENCE/manifest/legoesm_commit"
  printf 'L4_ACQUISITION_PASS steps=1..%s twins=3 evidence=%s\n' "$steps" "$EVIDENCE"
}

preflight
if [[ "$mode" == --admit-existing ]]; then
  [[ -d "$TWIN_A" && -d "$TWIN_B" && -d "$TWIN_C" ]] || fail 'existing twins missing'
  steps=$(tr -d '[:space:]' <"$TWIN_A/time.step")
  ((steps >= 2 && steps <= 10 && steps % FSBC == 0)) || fail "bad existing twin length $steps"
  admit "$steps"
  exit 0
fi
[[ "$mode" == --run ]] || {
  printf 'WOULD_RUN smoke, writer-free boundary, twins A/B/C into %s\n' "$EVIDENCE"
  exit 0
}

build_target
mkdir -p "$EVIDENCE"
run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE" 2 "$(pad10 2)" done
PYTHONPATH=$PYPATH python "$CHECKER" check-records "$SMOKE" 2
run_one "$REF_BIN" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps)
steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
((steps >= 2)) || fail "NEMO completed only $completed steps"
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_C" "$steps" "$(list_c "$steps")" done
admit "$steps"
