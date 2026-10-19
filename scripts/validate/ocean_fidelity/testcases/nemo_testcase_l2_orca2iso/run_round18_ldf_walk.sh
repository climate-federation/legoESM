#!/usr/bin/env bash
set -Eeuo pipefail

unexpected() {
  local status=$?
  printf 'REFUSE: unexpected L2 round-18 failure at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap unexpected ERR

refuse() {
  local status=$1
  shift
  printf 'REFUSE: %s\n' "$*" >&2
  exit "$status"
}

# Operator-executed acquisition. The default is read-only preflight; --run is
# the only mode that builds or runs NEMO.
readonly MODE=${1:---preflight-only}
case "$MODE" in
  --preflight-only|--run|--admit-existing|--plant-layout|--plant-admission) ;;
  *) refuse 64 "usage: $0 [--preflight-only|--run|--admit-existing|--plant-layout|--plant-admission]" ;;
esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly TARGET_CFG=ORCA2G5_OMIP_L4_VMIX_R18LDFWALK
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round17/oracle_l2_r17/twin_a
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round18/oracle_l2_r18_ldf_walk
readonly RECORD=oracle_l2_ldf_walk_kt00000001.bin
readonly SOURCE_BINARY_SHA=41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275
readonly SOURCE_TRA_SHA=e0ff55594dfd34eefe0d39aa9d4544567d7e6e2c18433ec77a5432124db9b3b3
readonly SOURCE_SCHEME_SHA=65d3c21f96f395ab71ff513a15c671ae75c8a87a44d7d8e98152e1a409624598
readonly SOURCE_DECK_SHA=e1ed6a7e1dfa670a65701329afcfa88fb7b66b5dd7699dfedd4e809e075fc39e
readonly ENTRY_SHA=27c98aeb61498f63f79d8e4c22caac1f4b5d73670b6daae38a32bfaf3645b3bd
readonly STAGE3_SHA=37fc7d2e294e8c4f42e398675fb99c13a8294c39bca037199637afe6d525168d
readonly EXPECTED_SIZE=279904060

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly CANONICAL_TRA=$NEMO_ROOT/src/OCE/TRA/traldf_iso.F90
readonly CANONICAL_SCHEME=$NEMO_ROOT/src/OCE/TRA/traldf_iso_scheme.h90
readonly TRA_PATCH=$here/traldf_iso_round18.patch
readonly SCHEME_PATCH=$here/traldf_iso_scheme_round18.patch
readonly PREREG=$REPO/docs/ocean/fidelity/testcases/PREREG_nemo_testcases_gridladder_vmix_l2_round18.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly INPUT_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc
  ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc q_10.15JUNE2009_fill.nc
  ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc
  sss_data.nc runoff_core_monthly.nc namelist_ref namelist_ice_ref namelist_ice_cfg)

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  refuse 63 "acquisition requires a clean committed tree"
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

check_sha() {
  local expected=$1 path=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || \
    refuse 65 "$path hash is $actual, expected $expected"
}

for path in "$CANONICAL_TRA" "$CANONICAL_SCHEME" "$TRA_PATCH" \
    "$SCHEME_PATCH" "$PREREG" "$SOURCE_ROOT/BLD/bin/nemo.exe" \
    "$SOURCE_RUN/nemo" "$SOURCE_RUN/namelist_cfg"; do
  [[ -f "$path" ]] || refuse 64 "missing required input $path"
done
for name in "${INPUT_FILES[@]}"; do
  [[ -f "$SOURCE_RUN/$name" ]] || refuse 64 "source record lacks $name"
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]] || \
  refuse 64 "source card is incomplete: $SOURCE_ROOT"
check_sha "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe"
check_sha "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo"
check_sha "$SOURCE_TRA_SHA" "$CANONICAL_TRA"
check_sha "$SOURCE_SCHEME_SHA" "$CANONICAL_SCHEME"
check_sha "$SOURCE_DECK_SHA" "$SOURCE_RUN/namelist_cfg"
check_sha "$ENTRY_SHA" "$SOURCE_RUN/oracle_step_entry_kt00000001.bin"
check_sha "$STAGE3_SHA" "$SOURCE_RUN/oracle_rktracer_stage3_kt00000001.bin"

for patch_path in "$TRA_PATCH" "$SCHEME_PATCH"; do
  removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$patch_path")
  [[ "$removed" -eq 0 ]] || refuse 66 "$patch_path removes $removed source lines"
done

readonly DRY=$(mktemp -d /tmp/orca2-l2-r18-ldf-source.XXXXXXXX)
cp "$CANONICAL_TRA" "$DRY/traldf_iso.F90"
cp "$CANONICAL_SCHEME" "$DRY/traldf_iso_scheme.h90"
patch -s -d "$DRY" -p0 --fuzz=0 <"$TRA_PATCH"
patch -s -d "$DRY" -p0 --fuzz=0 <"$SCHEME_PATCH"

check_layout() {
  local tra=$1 scheme=$2
  [[ "$(grep -Fc "r18_magic = 'NEMO_L2_R18LDF1X'" "$tra")" -eq 1 ]] &&
  [[ "$(grep -Fc 'lr18_write = lwp .AND. kt == nit000' "$tra")" -eq 1 ]] &&
  [[ "$(grep -Fc 'ROUND18_L2_LDF_DUMP' "$tra")" -eq 1 ]] &&
  [[ "$(grep -Fc 'WRITE(r18_unit)' "$tra")" -eq 5 ]] &&
  [[ "$(grep -Fc 'lr18_write .AND. jn == jp_tem' "$scheme")" -eq 4 ]] &&
  [[ "$(grep -Fc 'r18_e3u_live(ji,jj,jk) =' "$scheme")" -eq 1 ]] &&
  [[ "$(grep -Fc 'r18_fw_upper(ji,jj,jk) = zfw_kp1' "$scheme")" -eq 1 ]]
}
check_layout "$DRY/traldf_iso.F90" "$DRY/traldf_iso_scheme.h90" || \
  refuse 66 "dry source lacks the registered Round-18 writer layout"

if [[ "$MODE" == --plant-layout ]]; then
  sed -i '/r18_e3u_live(ji,jj,jk) =/d' "$DRY/traldf_iso_scheme.h90"
  if check_layout "$DRY/traldf_iso.F90" "$DRY/traldf_iso_scheme.h90"; then
    refuse 2 "source-layout plant stayed green"
  fi
  printf 'STATUS PLANT-FIRED: source-layout\n'
  refuse 69 "intentional source-layout plant exit"
fi

readonly SYNTAX=$(mktemp -d /tmp/orca2-l2-r18-ldf-syntax.XXXXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$DRY" -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$DRY/traldf_iso.F90" -o "$SYNTAX/traldf_iso.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -I "$SYNTAX" -J "$SYNTAX" "$SYNTAX/traldf_iso.f90" || \
  refuse 66 "gfortran syntax proof failed for traldf_iso.f90"
printf 'SYNTAX_PROOF_PASS traldf_iso.f90\n'

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND18_L2_LDF_PREFLIGHT_READY target=%s evidence=%s\n' \
    "$TARGET_CFG" "$TARGET_RUN"
  exit 0
fi

validate_record() {
  local record_path=$1
  "$PY" - "$record_path" "$EXPECTED_SIZE" <<'PYRECORD'
import struct
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
expected_size = int(sys.argv[2])
blob = path.read_bytes()
if len(blob) != expected_size:
    raise SystemExit(
        f"REFUSE: {path.name} has {len(blob)} bytes, expected {expected_size}")
if blob[:16] != b"NEMO_L2_R18LDF1X":
    raise SystemExit(f"REFUSE: {path.name} magic is {blob[:16]!r}")
header = struct.unpack_from("=13i", blob, 16)
expected_header = (1, 1, 1, 2, 3, 184, 152, 31, 30, 1, 64, 40, 11)
if header != expected_header:
    raise SystemExit(f"REFUSE: header is {header}, expected {expected_header}")

names3 = (
    "T_Kbb", "Krhs_before", "Krhs_after", "Krhs_increment",
    "e3t_3d", "e3u_3d", "e3v_3d", "e3u_live", "e3v_live",
    "tmask", "umask", "vmask", "wmask", "ahtu", "ahtv", "uslp",
    "vslp", "wslpi", "wslpj", "ah_wslp2", "akz", "dit", "djt",
    "dkt", "A11", "A22", "A13", "A23", "hmsku", "hmskv", "fu",
    "fv", "vmsku", "vmskv", "ahu_w", "ahv_w", "A31", "A32",
    "fw_lower", "fw_upper")
names2 = ("r3t_Kmm", "r3u_Kmm", "r3v_Kmm", "e2_e1u", "e1_e2v",
          "e2u", "e1v", "e1t", "e2t", "e1e2t", "r1_e1e2t")
shape3, shape2 = (184, 152, 31), (184, 152)
count3, count2 = int(np.prod(shape3)), int(np.prod(shape2))
offset = 16 + 13 * 4
arrays = {}
for name in names3:
    arrays[name] = np.frombuffer(
        blob, dtype="=f8", count=count3, offset=offset).reshape(
            shape3, order="F")
    offset += count3 * 8
for name in names2:
    arrays[name] = np.frombuffer(
        blob, dtype="=f8", count=count2, offset=offset).reshape(
            shape2, order="F")
    offset += count2 * 8
arrays["e3w_1d"] = np.frombuffer(blob, dtype="=f8", count=31, offset=offset)
offset += 31 * 8
if offset != len(blob):
    raise SystemExit(f"REFUSE: parser stopped at {offset} of {len(blob)} bytes")
for name, value in arrays.items():
    if not np.isfinite(value).all():
        raise SystemExit(f"REFUSE: {name} contains non-finite values")

rebuilt_u = arrays["e3u_3d"] * (
    np.float64(1.0) + arrays["r3u_Kmm"][..., None] * arrays["umask"])
rebuilt_v = arrays["e3v_3d"] * (
    np.float64(1.0) + arrays["r3v_Kmm"][..., None] * arrays["vmask"])
if not np.array_equal(rebuilt_u, arrays["e3u_live"]):
    raise SystemExit("REFUSE: dumped e3u_live does not replay its NEMO statement")
if not np.array_equal(rebuilt_v, arrays["e3v_live"]):
    raise SystemExit("REFUSE: dumped e3v_live does not replay its NEMO statement")
if np.count_nonzero(arrays["dkt"][:, :, 0]) != 0:
    raise SystemExit("REFUSE: NEMO surface dkt is not zero")
if np.count_nonzero(arrays["fw_lower"][:, :, 0]) != 0:
    raise SystemExit("REFUSE: NEMO surface vertical flux is not zero")
if np.count_nonzero(arrays["fw_upper"][:, :, 0]) == 0:
    raise SystemExit("REFUSE: first below-surface vertical flux is inert")
rebuilt_increment = arrays["Krhs_after"] - arrays["Krhs_before"]
if not np.array_equal(rebuilt_increment, arrays["Krhs_increment"]):
    raise SystemExit("REFUSE: stored LDF increment is not exact before/after")
for name in ("e3u_live", "e3v_live", "A11", "A22", "fu", "fv",
             "A31", "A32", "fw_upper", "Krhs_increment"):
    if np.count_nonzero(arrays[name]) == 0:
        raise SystemExit(f"REFUSE: registered row {name} is unexpectedly zero")
print(f"RECORD_LAYOUT_PASS {path.name} bytes={len(blob)}")
print("FACE_STATEMENT_CALIBRATION_PASS e3u_unequal=0 e3v_unequal=0")
print("SURFACE_BOUNDARY_CALIBRATION_PASS dkt_surface=0 fw_surface=0")
PYRECORD
}

validate_existing() {
  for path in "$TARGET_ROOT" "$TARGET_RUN" "$BINARY" "$TARGET_RUN/nemo" \
      "$TARGET_RUN/$RECORD" "$TARGET_RUN/$RECORD.stamp" \
      "$TARGET_RUN/producer_commit.txt" "$TARGET_RUN/binary.sha256" \
      "$TARGET_RUN/run.user.stdout.log" "$TARGET_RUN/run.user.time.log"; do
    [[ -e "$path" ]] || refuse 70 "existing acquisition lacks $path"
  done
  grep -Fxq 'STOP 0' "$TARGET_RUN/run.user.stdout.log" || \
    refuse 70 "NEMO stdout lacks STOP 0"
  grep -Fxq 'RUN_DONE' "$TARGET_RUN/run.user.time.log" || \
    refuse 70 "acquisition lacks RUN_DONE"
  for name in oracle_step_entry_kt00000001.bin \
      oracle_rktracer_stage3_kt00000001.bin oracle_stage_kt00000001_s3.bin; do
    cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name" || \
      refuse 71 "write-only instrument moved $name"
  done
  printf 'TWIN_ADMISSION_PASS entry_stage3_byte_identical=1\n'
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
  printf 'ROUND18_L2_LDF_READY %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --plant-admission ]]; then
  [[ -f "$TARGET_RUN/$RECORD" ]] || \
    refuse 70 "admission plant needs an existing record"
  planted=$(mktemp /tmp/orca2-l2-r18-record-plant.XXXXXXXX)
  cp "$TARGET_RUN/$RECORD" "$planted"
  printf 'X' | dd of="$planted" bs=1 seek=0 count=1 conv=notrunc status=none
  if validate_record "$planted" >/tmp/orca2-l2-r18-record-plant.log 2>&1; then
    refuse 2 "record-magic plant stayed green"
  fi
  grep -Fq 'REFUSE:' /tmp/orca2-l2-r18-record-plant.log || \
    refuse 72 "record-magic plant failed without a named refusal"
  printf 'STATUS PLANT-FIRED: record-magic\n'
  refuse 69 "intentional record-magic plant exit"
fi

if [[ "$MODE" == --admit-existing ]]; then
  validate_existing
  exit 0
fi

[[ "$MODE" == --run ]] || refuse 64 "unexpected mode $MODE"
[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || \
  refuse 67 "target exists; use --admit-existing: $TARGET_ROOT or $TARGET_RUN"
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || refuse 67 "$mount has under 4 GB free"
done

readonly MANIFEST=$(mktemp -d /tmp/orca2-l2-r18-ldf-manifest.XXXXXXXX)
printf '%s\n' "$COMMIT" >"$MANIFEST/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$MANIFEST/source_cfg.sha256"
sha256sum "$CANONICAL_TRA" "$CANONICAL_SCHEME" "$TRA_PATCH" \
  "$SCHEME_PATCH" "$PREREG" "$SOURCE_ROOT/BLD/ppsrc/nemo/traldf_iso.f90" \
  "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/ocean.output" \
  >"$MANIFEST/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r ORCA2_OMIP_L4 -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
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
for key, value in (("nn_itend", "1"), ("nn_stock", "1")):
    text, count = re.subn(
        rf"^(\s*{key}\s*=\s*)(\S+)", rf"\g<1>{value}", text,
        count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"REFUSE: {key} was not found exactly once")
text, count = re.subn(
    r"^(\s*nn_stocklist\s*=\s*).*$", r"\g<1>1,0,0,0,0,0,0,0,0,0",
    text, count=1, flags=re.MULTILINE)
if count != 1:
    raise SystemExit("REFUSE: nn_stocklist was not found exactly once")
open(target, "w", encoding="utf-8").write(text)
print("NAMELIST_DELTA_PASS nn_itend=1 nn_stock=1 nn_stocklist=1")
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
  "r18_magic = 'NEMO_L2_R18LDF1X'"; do
  grep -Fq "$needle" "$COMPILED_TRA" || \
    refuse 68 "compiled tracer LDF source lacks $needle"
done
if nm -D "$BINARY" | grep -q '_ZGV'; then
  refuse 68 "vector-math symbol present in new binary"
fi
sha256sum "$BINARY" >"$MANIFEST/binary.sha256"

mkdir -p "$TARGET_RUN"
for name in "${INPUT_FILES[@]}"; do cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done
cp "$TARGET_ROOT/EXP00/namelist_cfg" "$TARGET_RUN/namelist_cfg"
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
