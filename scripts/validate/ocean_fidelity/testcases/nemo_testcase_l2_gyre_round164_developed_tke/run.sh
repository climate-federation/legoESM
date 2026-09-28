#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: unexpected Round-164 failure at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

refuse() {
  local status=$1
  shift
  printf 'REFUSE: %s\n' "$*" >&2
  exit "$status"
}

# USER-EXECUTED ACQUISITION.  The sandbox cannot provide mpirun's PMIx
# socket.  --preflight-only performs the complete additive-patch and
# gfortran syntax proof without invoking makenemo or mpirun.
readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--plant-layout|--plant-admission|--admit-existing) ;;
  *) refuse 64 "usage: $0 [--run|--preflight-only|--plant-layout|--plant-admission|--admit-existing]" ;;
esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R132DAILY
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R164TKEDEV
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round132/oracle_daily_restarts
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round164/oracle_developed_tke
readonly REFERENCE_RESTART=$SOURCE_RUN/GYRE_OMIP_L2_P3_00001080_restart.nc
readonly CANDIDATE_RESTART=$TARGET_RUN/GYRE_OMIP_L2_P3_00001080_restart.nc
readonly OPERANDS=oracle_tke_operands_kt00001081.bin
readonly STATEMENTS=oracle_tke_statement_walk_kt00001081.bin
readonly OPERANDS_SIZE=4546636
readonly STATEMENTS_SIZE=873028

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SHIPPED_TKE=$NEMO_ROOT/src/OCE/ZDF/zdftke.F90
readonly SHIPPED_PHY=$NEMO_ROOT/src/OCE/ZDF/zdfphy.F90
readonly R54_DIR=$here/../nemo_testcase_l2_gyre_round54_tke_operands
readonly R101_DIR=$here/../nemo_testcase_l2_gyre_round101_tke_walk
readonly R54_WRITER=$R54_DIR/l2_r54_tke.F90
readonly R101_WRITER=$R101_DIR/l2_r101_tke_walk.F90
readonly R54_SOURCE_PATCH=$R54_DIR/zdftke_round54.patch
readonly R101_SOURCE_PATCH=$R101_DIR/zdftke_round101.patch
readonly PHY_PATCH=$R54_DIR/zdfphy_round55.patch
readonly R54_DEVELOPED_PATCH=$here/l2_r54_tke_developed.patch
readonly R101_DEVELOPED_PATCH=$here/l2_r101_tke_walk_developed.patch
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round164.md
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  refuse 63 "acquisition requires a clean committed tree"
readonly COMMIT=$(git rev-parse HEAD)

export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$SHIPPED_TKE" "$SHIPPED_PHY" "$R54_WRITER" "$R101_WRITER" \
    "$R54_SOURCE_PATCH" "$R101_SOURCE_PATCH" "$PHY_PATCH" \
    "$R54_DEVELOPED_PATCH" "$R101_DEVELOPED_PATCH" "$PREREG" \
    "$REFERENCE_RESTART"; do
  [[ -f "$path" ]] || refuse 64 "missing required input $path"
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]] || \
  refuse 64 "source card is incomplete: $SOURCE_ROOT"
[[ "$(stat -c %s "$REFERENCE_RESTART")" -eq 1466328 ]] || \
  refuse 65 "day-180 reference restart size moved"

readonly DRY=$(mktemp -d /tmp/gyre-r164-tke-source.XXXXXXXX)
cp "$SHIPPED_TKE" "$DRY/zdftke.F90"
cp "$SHIPPED_PHY" "$DRY/zdfphy.F90"
cp "$R54_WRITER" "$DRY/l2_r54_tke.F90"
cp "$R101_WRITER" "$DRY/l2_r101_tke_walk.F90"
patch -s "$DRY/zdftke.F90" <"$R54_SOURCE_PATCH"
patch -s "$DRY/zdftke.F90" <"$R101_SOURCE_PATCH"
patch -s "$DRY/zdfphy.F90" <"$PHY_PATCH"
patch -s "$DRY/l2_r54_tke.F90" <"$R54_DEVELOPED_PATCH"
patch -s "$DRY/l2_r101_tke_walk.F90" <"$R101_DEVELOPED_PATCH"

check_layout() {
  local tke=$1
  [[ "$(grep -c 'CALL r54_tke_begin' "$tke")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r54_tke_matrix_row' "$tke")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r54_tke_avn_row' "$tke")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r54_tke_finish' "$tke")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_tke_begin' "$tke")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_entry' "$tke")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_after_boundaries_row' "$tke")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_after_langmuir_row' "$tke")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_rhs_row' "$tke")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_post_sweep_row' "$tke")" -eq 1 ]] &&
  [[ "$(grep -c 'CALL r101_tke_finish' "$tke")" -eq 1 ]]
}
check_layout "$DRY/zdftke.F90" || refuse 66 "dry source lacks one registered writer call"
[[ "$(grep -c 'CALL r54_zdfphy_finish' "$DRY/zdfphy.F90")" -eq 1 ]] || \
  refuse 66 "dry zdfphy lacks the registered exit writer"
grep -Fq 'kt == 1081' "$DRY/l2_r54_tke.F90" || \
  refuse 66 "operand writer does not select developed step 1081"
grep -Fq 'kt == 1081' "$DRY/l2_r101_tke_walk.F90" || \
  refuse 66 "statement writer does not select developed step 1081"

if [[ "$MODE" == --plant-layout ]]; then
  sed -i '/CALL r101_after_langmuir_row/d' "$DRY/zdftke.F90"
  if check_layout "$DRY/zdftke.F90"; then
    refuse 2 "source-layout plant stayed green"
  fi
  refuse 69 "source-layout plant removed the post-Langmuir boundary"
fi

readonly SYNTAX=$(mktemp -d /tmp/gyre-r164-tke-syntax.XXXXXXXX)
preprocess() {
  cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P \
    -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$1" -o "$2"
}
preprocess "$DRY/l2_r54_tke.F90" "$SYNTAX/l2_r54_tke.f90"
preprocess "$DRY/l2_r101_tke_walk.F90" "$SYNTAX/l2_r101_tke_walk.f90"
preprocess "$DRY/zdftke.F90" "$SYNTAX/zdftke.f90"
preprocess "$DRY/zdfphy.F90" "$SYNTAX/zdfphy.f90"
for source in l2_r54_tke l2_r101_tke_walk zdftke zdfphy; do
  "$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
    -I "$SYNTAX" -J "$SYNTAX" "$SYNTAX/$source.f90" || \
    refuse 66 "gfortran syntax proof failed for $source.f90"
  printf 'SYNTAX_PROOF_PASS %s.f90\n' "$source"
done

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND164_DEVELOPED_TKE_PREFLIGHT_READY %s\n' "$DRY"
  exit 0
fi

validate_existing() {
  for path in "$TARGET_ROOT" "$TARGET_RUN" "$BINARY" "$TARGET_RUN/nemo" \
      "$TARGET_RUN/$OPERANDS" "$TARGET_RUN/$STATEMENTS" \
      "$TARGET_RUN/$OPERANDS.stamp" "$TARGET_RUN/$STATEMENTS.stamp" \
      "$TARGET_RUN/producer_commit.txt" "$TARGET_RUN/binary.sha256" \
      "$TARGET_RUN/run.user.stdout.log" "$TARGET_RUN/run.user.time.log" \
      "$CANDIDATE_RESTART"; do
    [[ -e "$path" ]] || refuse 64 "existing acquisition lacks $path"
  done
  [[ "$(stat -c %s "$TARGET_RUN/$OPERANDS")" -eq "$OPERANDS_SIZE" ]] || \
    refuse 66 "operand record size differs from $OPERANDS_SIZE"
  [[ "$(stat -c %s "$TARGET_RUN/$STATEMENTS")" -eq "$STATEMENTS_SIZE" ]] || \
    refuse 66 "statement record size differs from $STATEMENTS_SIZE"
  grep -Fxq 'STOP 0' "$TARGET_RUN/run.user.stdout.log" || \
    refuse 66 "NEMO stdout lacks STOP 0"
  grep -Fxq 'RUN_DONE' "$TARGET_RUN/run.user.time.log" || \
    refuse 66 "acquisition lacks RUN_DONE"
  cmp -s "$REFERENCE_RESTART" "$CANDIDATE_RESTART" || \
    refuse 71 "instrumented day-180 restart differs from the uninstrumented Round-132 restart"
  printf 'TWIN_ADMISSION_PASS restart_kt1080_byte_identical=1\n'
  "$PY" - "$TARGET_RUN/$OPERANDS" "$TARGET_RUN/$STATEMENTS" "$MODE" <<'PYRECORD'
import struct
import sys
from pathlib import Path

for raw, magic, expected, expected_header in (
    (sys.argv[1], b"NEMO_L2_R56TKE2 ", 4546636,
     (2, 1081, 1, 1, 32, 22, 31, 30, 1, 32, 1, 22, 64)),
    (sys.argv[2], b"NEMO_L2_R101TKE ", 873028,
     (1, 1081, 1, 1, 36, 26, 31, 30, 3, 34, 3, 24, 64)),
):
    path = Path(raw)
    blob = path.read_bytes()
    if len(blob) != expected:
        raise SystemExit(f"REFUSE: {path.name} has {len(blob)} bytes, expected {expected}")
    expected_magic = (b"X" + magic[1:]
                      if sys.argv[3] == "--plant-admission" else magic)
    if blob[:16] != expected_magic:
        raise SystemExit(f"REFUSE: {path.name} magic is {blob[:16]!r}")
    ints = struct.unpack_from("=13i", blob, 16)
    if ints != expected_header:
        raise SystemExit(
            f"REFUSE: {path.name} header is {ints}, expected {expected_header}")
    print(f"RECORD_LAYOUT_PASS {path.name} bytes={len(blob)} kt={ints[1]}")
PYRECORD
  local name digest stamped_digest stamped_commit stamped_name producer_commit
  producer_commit=$(<"$TARGET_RUN/producer_commit.txt")
  [[ "$producer_commit" =~ ^[0-9a-f]{40}$ ]] || \
    refuse 66 "producer_commit.txt is not a full commit"
  for name in "$OPERANDS" "$STATEMENTS"; do
    digest=$(sha256sum "$TARGET_RUN/$name" | awk '{print $1}')
    read -r stamped_digest stamped_commit stamped_name <"$TARGET_RUN/$name.stamp"
    [[ "$digest" == "$stamped_digest" && "$stamped_commit" == "$producer_commit" \
       && "$stamped_name" == "$name" ]] || \
      refuse 66 "commit stamp disagrees for $name"
  done
  [[ "$MODE" != --plant-admission ]] || \
    refuse 2 "record-magic plant stayed green"
  printf 'ROUND164_DEVELOPED_TKE_READY %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing || "$MODE" == --plant-admission ]]; then
  validate_existing
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || \
  refuse 64 "target exists; use --admit-existing: $TARGET_ROOT or $TARGET_RUN"
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || refuse 67 "$mount has under 4 GB free"
done

readonly MANIFEST=$(mktemp -d /tmp/gyre-r164-tke-manifest.XXXXXXXX)
printf '%s\n' "$COMMIT" >"$MANIFEST/producer_commit.txt"
sha256sum "$SHIPPED_TKE" "$SHIPPED_PHY" "$R54_WRITER" "$R101_WRITER" \
  "$R54_SOURCE_PATCH" "$R101_SOURCE_PATCH" "$PHY_PATCH" \
  "$R54_DEVELOPED_PATCH" "$R101_DEVELOPED_PATCH" "$PREREG" \
  "$REFERENCE_RESTART" >"$MANIFEST/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$DRY/zdftke.F90" "$TARGET_ROOT/MY_SRC/zdftke.F90"
cp "$DRY/zdfphy.F90" "$TARGET_ROOT/MY_SRC/zdfphy.F90"
cp "$DRY/l2_r54_tke.F90" "$TARGET_ROOT/MY_SRC/l2_r54_tke.F90"
cp "$DRY/l2_r101_tke_walk.F90" "$TARGET_ROOT/MY_SRC/l2_r101_tke_walk.F90"

"$PY" - "$SOURCE_ROOT/EXP00/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg" <<'PYNAMELIST'
import re
import sys

source, target = sys.argv[1:3]
text = open(source, encoding="utf-8").read()
for key, value in (("nn_itend", "1081"), ("nn_stock", "1080"), ("nn_write", "0")):
    text, count = re.subn(
        rf"^(\s*{key}\s*=\s*)(\S+)", rf"\g<1>{value}", text,
        count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"REFUSE: {key} was not found exactly once")
open(target, "w", encoding="utf-8").write(text)
print("NAMELIST_DELTA_PASS nn_itend=1081 nn_stock=1080 nn_write=0")
PYNAMELIST
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || refuse 68 "target build produced no executable"
check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/zdftke.f90" || \
  refuse 68 "compiled zdftke lacks a registered call"
if nm -D "$BINARY" | grep -q '_ZGV'; then
  refuse 68 "vector-math symbol present in new binary"
fi
sha256sum "$BINARY" >"$MANIFEST/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do
  cp -L "$TARGET_ROOT/EXP00/$name" "$TARGET_RUN/$name"
done
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$MANIFEST"/* "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  start_seconds=$SECONDS
  set +e
  mpirun -np 1 --oversubscribe ./nemo > >(tee run.user.stdout.log) 2>&1
  run_status=$?
  set -e
  [[ "$run_status" -eq 0 ]] || refuse "$run_status" "NEMO acquisition exited $run_status"
  elapsed=$((SECONDS - start_seconds))
  printf 'elapsed_seconds=%s\nRUN_DONE\n' "$elapsed" >>run.user.time.log
  for name in "$OPERANDS" "$STATEMENTS"; do
    [[ -s "$name" ]] || refuse 66 "NEMO produced no $name"
    digest=$(sha256sum "$name" | awk '{print $1}')
    printf '%s %s %s\n' "$digest" "$COMMIT" "$name" >"$name.stamp"
  done
)

validate_existing
