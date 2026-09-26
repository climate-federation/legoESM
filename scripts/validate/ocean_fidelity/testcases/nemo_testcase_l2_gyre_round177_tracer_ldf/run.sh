#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: unexpected Round-177 failure at line %s (exit %s)\n' \
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

# USER-EXECUTED ACQUISITION.  mpirun's PMIx socket is unavailable inside the
# sandbox.  --preflight-only performs the exact patch and syntax proof.
readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--plant-layout|--plant-admission|--admit-existing) ;;
  *) refuse 64 "usage: $0 [--run|--preflight-only|--plant-layout|--plant-admission|--admit-existing]" ;;
esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R132DAILY
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R177TRALDF
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round132/oracle_daily_restarts
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round177/oracle_tracer_ldf_walk
readonly PROCESS_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round123/oracle_process_budget
readonly RECORD=oracle_tracer_ldf_walk_kt00001081.bin
readonly PROCESS_RECORD=oracle_process_budget_kt00001081.bin
readonly RESTART_1080=GYRE_OMIP_L2_P3_00001080_restart.nc
readonly RESTART_1440=GYRE_OMIP_L2_P3_00001440_restart.nc
readonly EXPECTED_SIZE=8903548
readonly SOURCE_BINARY_SHA=24aefbfb9f4b596c4c0c8002c17811577f3622b4c4bf34add13d770caf988eff
readonly SOURCE_TRA_SHA=e0ff55594dfd34eefe0d39aa9d4544567d7e6e2c18433ec77a5432124db9b3b3
readonly SOURCE_SCHEME_SHA=65d3c21f96f395ab71ff513a15c671ae75c8a87a44d7d8e98152e1a409624598
readonly SHA_1080=6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976
readonly SHA_1440=96529a98da0e0d89b328632a826a9d41593f81f0d1a917350f28f184d49b163a
readonly PROCESS_SHA=526d1fc73faeda990c661f2363a5bb328168bae17d4aea315daf05c35b4cd7b0

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly CANONICAL_TRA=$NEMO_ROOT/src/OCE/TRA/traldf_iso.F90
readonly CANONICAL_SCHEME=$NEMO_ROOT/src/OCE/TRA/traldf_iso_scheme.h90
readonly TRA_PATCH=$here/traldf_iso_round177.patch
readonly SCHEME_PATCH=$here/traldf_iso_scheme_round177.patch
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round177.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  refuse 63 "acquisition requires a clean committed tree"
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$CANONICAL_TRA" "$CANONICAL_SCHEME" "$TRA_PATCH" \
    "$SCHEME_PATCH" "$PREREG" "$SOURCE_ROOT/BLD/bin/nemo.exe" \
    "$SOURCE_RUN/nemo" "$SOURCE_RUN/namelist_cfg" \
    "$SOURCE_RUN/$RESTART_1080" "$SOURCE_RUN/$RESTART_1440" \
    "$PROCESS_ROOT/$PROCESS_RECORD"; do
  [[ -f "$path" ]] || refuse 64 "missing required input $path"
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]] || \
  refuse 64 "source card is incomplete: $SOURCE_ROOT"

check_sha() {
  local expected=$1 path=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || \
    refuse 65 "$path hash is $actual, expected $expected"
}
check_sha "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe"
check_sha "$SOURCE_TRA_SHA" "$CANONICAL_TRA"
check_sha "$SOURCE_SCHEME_SHA" "$CANONICAL_SCHEME"
check_sha "$SHA_1080" "$SOURCE_RUN/$RESTART_1080"
check_sha "$SHA_1440" "$SOURCE_RUN/$RESTART_1440"
check_sha "$PROCESS_SHA" "$PROCESS_ROOT/$PROCESS_RECORD"
cmp -s "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" || \
  refuse 65 "Round-132 run binary differs from its compiled card"
[[ "$(stat -c %s "$SOURCE_RUN/$RESTART_1080")" -eq 1466328 ]] || \
  refuse 65 "Round-132 step-1080 restart size moved"
[[ "$(stat -c %s "$SOURCE_RUN/$RESTART_1440")" -eq 1466328 ]] || \
  refuse 65 "Round-132 step-1440 restart size moved"
for row in 'nn_itend *= *2160' 'nn_stock *= *6' 'nn_write *= *2160'; do
  grep -Eq "^[[:space:]]*$row" "$SOURCE_RUN/namelist_cfg" || \
    refuse 65 "source namelist lacks row $row"
done

for patch_path in "$TRA_PATCH" "$SCHEME_PATCH"; do
  removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$patch_path")
  [[ "$removed" -eq 0 ]] || refuse 66 "$patch_path removes $removed source lines"
done

readonly DRY=$(mktemp -d /tmp/gyre-r177-tracer-ldf-source.XXXXXXXX)
cp "$CANONICAL_TRA" "$DRY/traldf_iso.F90"
cp "$CANONICAL_SCHEME" "$DRY/traldf_iso_scheme.h90"
patch -s -d "$DRY" -p0 --fuzz=0 <"$TRA_PATCH"
patch -s -d "$DRY" -p0 --fuzz=0 <"$SCHEME_PATCH"

check_layout() {
  local tra=$1 scheme=$2
  [[ "$(grep -Fc "r177_magic = 'NEMO_L2_R177LDF'" "$tra")" -eq 1 ]] &&
  [[ "$(grep -Fc 'lr177_write = lwp .AND. kt == 1081' "$tra")" -eq 1 ]] &&
  [[ "$(grep -Fc 'ROUND177_TRACER_LDF_DUMP' "$tra")" -eq 1 ]] &&
  [[ "$(grep -Fc 'WRITE(r177_unit)' "$tra")" -eq 5 ]] &&
  [[ "$(grep -Fc 'lr177_write .AND. jn == jp_tem' "$scheme")" -eq 4 ]] &&
  [[ "$(grep -Fc 'r177_fu(ji,jj,jk) = zfu(ji,jj)' "$scheme")" -eq 1 ]] &&
  [[ "$(grep -Fc 'r177_fw_upper(ji,jj,jk) = zfw_kp1' "$scheme")" -eq 1 ]]
}
check_layout "$DRY/traldf_iso.F90" "$DRY/traldf_iso_scheme.h90" || \
  refuse 66 "dry source lacks the registered Round-177 writer layout"

if [[ "$MODE" == --plant-layout ]]; then
  sed -i '/r177_fw_upper(ji,jj,jk) = zfw_kp1/d' "$DRY/traldf_iso_scheme.h90"
  if check_layout "$DRY/traldf_iso.F90" "$DRY/traldf_iso_scheme.h90"; then
    refuse 2 "source-layout plant stayed green"
  fi
  printf 'STATUS PLANT-FIRED: source-layout\n'
  refuse 69 "intentional source-layout plant exit"
fi

readonly SYNTAX=$(mktemp -d /tmp/gyre-r177-tracer-ldf-syntax.XXXXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$DRY" -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$DRY/traldf_iso.F90" -o "$SYNTAX/traldf_iso.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -I "$SYNTAX" -J "$SYNTAX" "$SYNTAX/traldf_iso.f90" || \
  refuse 66 "gfortran syntax proof failed for traldf_iso.f90"
printf 'SYNTAX_PROOF_PASS traldf_iso.f90\n'

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND177_TRACER_LDF_PREFLIGHT_READY %s\n' "$DRY"
  exit 0
fi

validate_record() {
  local record_path=$1
  "$PY" - "$record_path" "$PROCESS_ROOT/$PROCESS_RECORD" "$EXPECTED_SIZE" <<'PYRECORD'
import struct
import sys
from pathlib import Path

import numpy as np

record_path = Path(sys.argv[1])
process_path = Path(sys.argv[2])
expected_size = int(sys.argv[3])
blob = record_path.read_bytes()
if len(blob) != expected_size:
    raise SystemExit(
        f"REFUSE: {record_path.name} has {len(blob)} bytes, expected {expected_size}")
if blob[:16] != b"NEMO_L2_R177LDF ":
    raise SystemExit(f"REFUSE: {record_path.name} magic is {blob[:16]!r}")
header = struct.unpack_from("=13i", blob, 16)
expected_header = (1, 1081, 1, 1, 3, 36, 26, 31, 30, 1, 64, 38, 11)
if header != expected_header:
    raise SystemExit(f"REFUSE: header is {header}, expected {expected_header}")

names3 = (
    "T_Kbb", "Krhs_before", "Krhs_after", "Krhs_increment",
    "e3t_3d", "e3u_3d", "e3v_3d", "tmask", "umask", "vmask", "wmask",
    "ahtu", "ahtv", "uslp", "vslp", "wslpi", "wslpj", "ah_wslp2", "akz",
    "dit", "djt", "dkt", "A11", "A22", "A13", "A23", "hmsku", "hmskv",
    "fu", "fv", "vmsku", "vmskv", "ahu_w", "ahv_w", "A31", "A32",
    "fw_lower", "fw_upper")
names2 = ("r3t_Kmm", "r3u_Kmm", "r3v_Kmm", "e2_e1u", "e1_e2v",
          "e2u", "e1v", "e1t", "e2t", "e1e2t", "r1_e1e2t")
offset = 16 + 13 * 4
shape3 = (36, 26, 31)
shape2 = (36, 26)
count3 = int(np.prod(shape3))
count2 = int(np.prod(shape2))
arrays = {}
for name in names3:
    arrays[name] = np.frombuffer(blob, dtype="=f8", count=count3, offset=offset).reshape(shape3, order="F")
    offset += count3 * 8
for name in names2:
    arrays[name] = np.frombuffer(blob, dtype="=f8", count=count2, offset=offset).reshape(shape2, order="F")
    offset += count2 * 8
arrays["e3w_1d"] = np.frombuffer(blob, dtype="=f8", count=31, offset=offset)
offset += 31 * 8
if offset != len(blob):
    raise SystemExit(f"REFUSE: parser stopped at {offset} of {len(blob)} bytes")
for name, value in arrays.items():
    if not np.isfinite(value).all():
        raise SystemExit(f"REFUSE: {name} contains non-finite values")

wet = arrays["tmask"] != 0.0
if int(np.count_nonzero(wet)) != 18000:
    raise SystemExit(f"REFUSE: wet-cell count is {np.count_nonzero(wet)}, expected 18000")
rebuilt = arrays["Krhs_after"] - arrays["Krhs_before"]
if not np.array_equal(rebuilt[wet], arrays["Krhs_increment"][wet]):
    raise SystemExit("REFUSE: stored LDF increment is not the exact before/after difference")

process = process_path.read_bytes()
if len(process) != 1415300 or process[:16] != b"NEMO_L2_R123PROC":
    raise SystemExit("REFUSE: Round-123 process record layout changed")
process_header = struct.unpack_from("=11i", process, 16)
if process_header != (1, 1081, 3, 1, 1, 3, 2, 36, 26, 31, 64):
    raise SystemExit(f"REFUSE: Round-123 header changed: {process_header}")
process_offset = 16 + 11 * 4 + 8 + count3 * 8 + 3 * count2 * 8
process_rows = []
for _ in range(5):
    process_rows.append(np.frombuffer(
        process, dtype="=f8", count=count3, offset=process_offset).reshape(shape3, order="F"))
    process_offset += count3 * 8
process_increment = process_rows[3] - process_rows[2]
if not np.array_equal(process_increment[wet], arrays["Krhs_increment"][wet]):
    cells = int(np.count_nonzero(process_increment[wet] != arrays["Krhs_increment"][wet]))
    maximum = float(np.max(np.abs(process_increment[wet] - arrays["Krhs_increment"][wet])))
    raise SystemExit(
        f"REFUSE: Round-177 versus Round-123 active LDF increment moved {cells} cells, max {maximum:.17e}")

for name in ("T_Kbb", "Krhs_increment", "ahtu", "ahtv", "uslp", "vslp",
             "wslpi", "wslpj", "ah_wslp2", "akz", "dit", "djt", "dkt",
             "A11", "A22", "A13", "A23", "hmsku", "hmskv", "fu", "fv",
             "vmsku", "vmskv", "ahu_w", "ahv_w", "A31", "A32",
             "fw_lower", "fw_upper"):
    if int(np.count_nonzero(arrays[name])) == 0:
        raise SystemExit(f"REFUSE: registered row {name} is unexpectedly all zero")
print(f"RECORD_LAYOUT_PASS {record_path.name} bytes={len(blob)} wet=18000")
print("ROUND123_INCREMENT_CALIBRATION_PASS cells_unequal=0")
PYRECORD
}

validate_existing() {
  for path in "$TARGET_ROOT" "$TARGET_RUN" "$BINARY" "$TARGET_RUN/nemo" \
      "$TARGET_RUN/$RECORD" "$TARGET_RUN/$RECORD.stamp" \
      "$TARGET_RUN/producer_commit.txt" "$TARGET_RUN/binary.sha256" \
      "$TARGET_RUN/run.user.stdout.log" "$TARGET_RUN/run.user.time.log" \
      "$TARGET_RUN/$RESTART_1080" "$TARGET_RUN/$RESTART_1440"; do
    [[ -e "$path" ]] || refuse 70 "existing acquisition lacks $path"
  done
  grep -Fxq 'STOP 0' "$TARGET_RUN/run.user.stdout.log" || \
    refuse 70 "NEMO stdout lacks STOP 0"
  grep -Fxq 'RUN_DONE' "$TARGET_RUN/run.user.time.log" || \
    refuse 70 "acquisition lacks RUN_DONE"
  cmp -s "$SOURCE_RUN/$RESTART_1080" "$TARGET_RUN/$RESTART_1080" || \
    refuse 71 "instrument moved the step-1080 restart"
  cmp -s "$SOURCE_RUN/$RESTART_1440" "$TARGET_RUN/$RESTART_1440" || \
    refuse 71 "instrument moved the step-1440 restart"
  printf 'TWIN_ADMISSION_PASS restart_1080_byte_identical=1 restart_1440_byte_identical=1\n'
  validate_record "$TARGET_RUN/$RECORD"
  local producer_commit digest stamped_digest stamped_commit stamped_name
  producer_commit=$(<"$TARGET_RUN/producer_commit.txt")
  [[ "$producer_commit" =~ ^[0-9a-f]{40}$ ]] || \
    refuse 70 "producer_commit.txt is not a full commit"
  digest=$(sha256sum "$TARGET_RUN/$RECORD" | awk '{print $1}')
  read -r stamped_digest stamped_commit stamped_name <"$TARGET_RUN/$RECORD.stamp"
  [[ "$digest" == "$stamped_digest" && "$stamped_commit" == "$producer_commit" \
     && "$stamped_name" == "$RECORD" ]] || \
    refuse 70 "commit stamp disagrees for $RECORD"
  printf 'ROUND177_TRACER_LDF_READY %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --plant-admission ]]; then
  [[ -f "$TARGET_RUN/$RECORD" ]] || refuse 70 "admission plant needs an existing record"
  planted=$(mktemp /tmp/r177-record-plant.XXXXXXXX)
  cp "$TARGET_RUN/$RECORD" "$planted"
  printf 'X' | dd of="$planted" bs=1 seek=0 count=1 conv=notrunc status=none
  if validate_record "$planted" >/tmp/r177-record-plant.log 2>&1; then
    refuse 2 "record-magic plant stayed green"
  fi
  grep -Fq 'REFUSE:' /tmp/r177-record-plant.log || \
    refuse 72 "record-magic plant failed without a named refusal"
  printf 'STATUS PLANT-FIRED: record-magic\n'
  refuse 69 "intentional record-magic plant exit"
fi

if [[ "$MODE" == --admit-existing ]]; then
  validate_existing
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || \
  refuse 67 "target exists; use --admit-existing: $TARGET_ROOT or $TARGET_RUN"
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || refuse 67 "$mount has under 4 GB free"
done

readonly MANIFEST=$(mktemp -d /tmp/gyre-r177-tracer-ldf-manifest.XXXXXXXX)
printf '%s\n' "$COMMIT" >"$MANIFEST/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$MANIFEST/source_cfg.sha256"
sha256sum "$CANONICAL_TRA" "$CANONICAL_SCHEME" "$TRA_PATCH" \
  "$SCHEME_PATCH" "$PREREG" "$SOURCE_ROOT/BLD/ppsrc/nemo/traldf.f90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/traldf_iso.f90" \
  "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/ocean.output" \
  >"$MANIFEST/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$DRY/traldf_iso.F90" "$TARGET_ROOT/MY_SRC/traldf_iso.F90"
cp "$DRY/traldf_iso_scheme.h90" "$TARGET_ROOT/MY_SRC/traldf_iso_scheme.h90"

"$PY" - "$SOURCE_RUN/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg" <<'PYNAMELIST'
import re
import sys

source, target = sys.argv[1:3]
text = open(source, encoding="utf-8").read()
for key, value in (("nn_itend", "1440"), ("nn_stock", "360")):
    text, count = re.subn(
        rf"^(\s*{key}\s*=\s*)(\S+)", rf"\g<1>{value}", text,
        count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"REFUSE: {key} was not found exactly once")
open(target, "w", encoding="utf-8").write(text)
print("NAMELIST_DELTA_PASS nn_itend=1440 nn_stock=360")
PYNAMELIST

touch "$TARGET_ROOT/MY_SRC/"*.F90 "$TARGET_ROOT/MY_SRC/"*.h90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || refuse 68 "target build produced no executable"
check_layout "$TARGET_ROOT/MY_SRC/traldf_iso.F90" \
  "$TARGET_ROOT/MY_SRC/traldf_iso_scheme.h90" || \
  refuse 68 "target source lacks the registered writer layout"
readonly COMPILED_TRA=$TARGET_ROOT/BLD/ppsrc/nemo/traldf_iso.f90
for needle in 'CALL traldf_iso_a33' \
  'zA11 = e2_e1u(ji,jj) * (e3u_3d' \
  'zfw_kp1 = zA31' \
  'pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +' \
  "r177_magic = 'NEMO_L2_R177LDF'"; do
  grep -Fq "$needle" "$COMPILED_TRA" || \
    refuse 68 "compiled tracer LDF source lacks $needle"
done
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
  [[ -s "$RECORD" ]] || refuse 69 "NEMO produced no $RECORD"
  digest=$(sha256sum "$RECORD" | awk '{print $1}')
  printf '%s %s %s\n' "$digest" "$COMMIT" "$RECORD" >"$RECORD.stamp"
)

validate_existing
