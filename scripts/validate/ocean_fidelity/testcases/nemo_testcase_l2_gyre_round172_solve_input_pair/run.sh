#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: unexpected Round-172 failure at line %s (exit %s)\n' \
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

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--plant-input-truncation|--admit-existing) ;;
  *) refuse 64 "usage: $0 [--run|--preflight-only|--plant-input-truncation|--admit-existing]" ;;
esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R125ZDFMAG
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
readonly SOURCE_RUN=$L2/round125/oracle_vertical_decomposition
readonly TARGET_RUN=$L2/round172/oracle_solve_input_pair
readonly INPUT_ROOT=$L2/round172/lego_solve_inputs
readonly RAW_BYTES=167132160
readonly RESTART_1080=GYRE_OMIP_L2_P3_00001080_restart.nc
readonly RESTART_1440=GYRE_OMIP_L2_P3_00001440_restart.nc

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/trazdf_round172.patch
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round172.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  refuse 63 "acquisition requires a clean committed tree"
readonly COMMIT=$(git rev-parse HEAD)

for path in "$PATCH" "$PREREG" "$SOURCE_ROOT/MY_SRC/trazdf.F90" \
    "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$SOURCE_RUN/nemo" \
    "$SOURCE_RUN/$RESTART_1080" "$SOURCE_RUN/$RESTART_1440" \
    "$INPUT_ROOT/e3t_Kaa.npy" "$INPUT_ROOT/content_T.npy" \
    "$INPUT_ROOT/manifest.json" "$INPUT_ROOT/trace_files.sha256" \
    "$INPUT_ROOT/trace_files.stamp"; do
  [[ -f "$path" ]] || refuse 64 "missing required input $path"
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]] || \
  refuse 64 "source card is incomplete: $SOURCE_ROOT"
cmp -s "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" || \
  refuse 65 "Round-125 source run binary differs from its source card"

readonly DRY=$(mktemp -d /tmp/gyre-r172-solve-source.XXXXXXXX)
cp "$SOURCE_ROOT/MY_SRC/trazdf.F90" "$DRY/trazdf.F90"
patch -s --fuzz=0 "$DRY/trazdf.F90" <"$PATCH"
[[ "$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")" -eq 0 ]] || \
  refuse 66 "Round-172 source patch is not additive"
for needle in 'round172_e3t.arm' 'round172_content.arm' \
    'round172_solve_inputs.raw' 'zr172_e3t(ji,jj,jk)' \
    'zr172_content(ji,jj,jk)'; do
  grep -Fq "$needle" "$DRY/trazdf.F90" || \
    refuse 66 "dry source lacks $needle"
done

readonly SYNTAX=$(mktemp -d /tmp/gyre-r172-solve-syntax.XXXXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P \
  -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$DRY/trazdf.F90" -o "$SYNTAX/trazdf.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$SYNTAX" "$SYNTAX/trazdf.f90" || \
  refuse 66 "gfortran syntax proof failed for trazdf.f90"
printf 'SYNTAX_PROOF_PASS trazdf.f90\n'

readonly RAW=$DRY/round172_solve_inputs.raw
"$PY" - "$INPUT_ROOT" "$RAW" "$MODE" <<'PYINPUT'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

root, raw, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
manifest = json.loads((root / "manifest.json").read_text())
if manifest.get("format") != "gyre-round172-developed-solve-input-record-v1":
    raise SystemExit("REFUSE: solve-input manifest has the wrong format")
if manifest.get("shape") != [360, 22, 32, 30]:
    raise SystemExit(f"REFUSE: solve-input shape is {manifest.get('shape')}")
stamp = (root / "trace_files.stamp").read_text().split()
manifest_path = root / "trace_files.sha256"
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if len(stamp) != 3 or stamp[0] != digest or stamp[2] != manifest_path.name:
    raise SystemExit("REFUSE: solve-input commit stamp disagrees")
rows = {}
for line in manifest_path.read_text().splitlines():
    sha, name = line.split()
    rows[Path(name).name] = sha
for name in ("e3t_Kaa.npy", "content_T.npy", "manifest.json"):
    observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if rows.get(name) != observed:
        raise SystemExit(f"REFUSE: solve-input hash disagrees for {name}")
e3t = np.load(root / "e3t_Kaa.npy", mmap_mode="r")
content = np.load(root / "content_T.npy", mmap_mode="r")
with raw.open("wb") as handle:
    for frame in range(360):
        for source in (e3t[frame], content[frame]):
            full = np.zeros((36, 26, 31), dtype="<f8", order="F")
            full[2:34, 2:24, :30] = np.transpose(source, (1, 0, 2))
            handle.write(full.tobytes(order="F"))
if mode == "--plant-input-truncation":
    with raw.open("r+b") as handle:
        handle.truncate(raw.stat().st_size - 8)
expected = 167132160
if raw.stat().st_size != expected:
    raise SystemExit(
        f"REFUSE: solve-input raw size is {raw.stat().st_size}, expected {expected}")
print(f"SOLVE_INPUT_LAYOUT_PASS bytes={expected} producer={stamp[1]}")
PYINPUT
[[ "$MODE" != --plant-input-truncation ]] || \
  refuse 2 "input-truncation plant stayed green"
[[ "$(stat -c %s "$RAW")" -eq "$RAW_BYTES" ]] || \
  refuse 66 "converted solve-input byte count moved"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND172_SOLVE_INPUT_PAIR_PREFLIGHT_READY %s\n' "$DRY"
  exit 0
fi

validate_existing() {
  for arm in baseline e3t content; do
    local run=$TARGET_RUN/$arm
    [[ -f "$run/$RESTART_1080" && -f "$run/$RESTART_1440" ]] || \
      refuse 70 "existing $arm arm lacks required restarts"
    grep -Fxq 'STOP 0' "$run/run.user.stdout.log" || \
      refuse 70 "existing $arm arm lacks STOP 0"
  done
  cmp -s "$SOURCE_RUN/$RESTART_1080" "$TARGET_RUN/baseline/$RESTART_1080" || \
    refuse 71 "baseline day-180 restart moved"
  cmp -s "$SOURCE_RUN/$RESTART_1440" "$TARGET_RUN/baseline/$RESTART_1440" || \
    refuse 71 "baseline day-240 restart moved"
  cmp -s "$TARGET_RUN/baseline/$RESTART_1440" "$TARGET_RUN/e3t/$RESTART_1440" && \
    refuse 72 "e3t arm is non-discriminating at day 240"
  cmp -s "$TARGET_RUN/baseline/$RESTART_1440" "$TARGET_RUN/content/$RESTART_1440" && \
    refuse 72 "content arm is non-discriminating at day 240"
  printf 'ROUND172_SOLVE_INPUT_PAIR_READY %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  validate_existing
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || \
  refuse 64 "new target exists: $TARGET_ROOT or $TARGET_RUN"
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || refuse 67 "$mount has under 4 GB free"
done

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$DRY/trazdf.F90" "$TARGET_ROOT/MY_SRC/trazdf.F90"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || refuse 68 "target build produced no executable"
grep -Fq 'round172_solve_inputs.raw' \
  "$TARGET_ROOT/BLD/ppsrc/nemo/trazdf.f90" || \
  refuse 68 "compiled source lacks the Round-172 input reader"
if nm -D "$BINARY" | grep -q '_ZGV'; then
  refuse 68 "vector-math symbol present in new binary"
fi

mkdir "$TARGET_RUN"
readonly PREPARED='namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml'
for arm in baseline e3t content; do
  run=$TARGET_RUN/$arm
  mkdir "$run"
  for name in $PREPARED; do
    [[ -f "$SOURCE_RUN/$name" ]] || refuse 68 "source run lacks $name"
    cp -L "$SOURCE_RUN/$name" "$run/$name"
  done
  cp "$BINARY" "$run/nemo"
  cp "$RAW" "$run/round172_solve_inputs.raw"
  [[ "$arm" == e3t ]] && touch "$run/round172_e3t.arm"
  [[ "$arm" == content ]] && touch "$run/round172_content.arm"
  (
    cd "$run"
    export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    started=$SECONDS
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
    mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
    [[ "${PIPESTATUS[0]}" -eq 0 ]] || refuse 69 "$arm NEMO process failed"
    grep -Fxq 'STOP 0' run.user.stdout.log || refuse 69 "$arm lacks STOP 0"
    printf 'wall_seconds=%s\nRUN_DONE\n' "$((SECONDS-started))" >>run.user.time.log
  )
done

printf '%s\n' "$COMMIT" >"$TARGET_RUN/producer_commit.txt"
sha256sum "$BINARY" >"$TARGET_RUN/binary.sha256"
(
  cd "$TARGET_RUN"
  sha256sum baseline/$RESTART_1080 baseline/$RESTART_1440 \
    e3t/$RESTART_1440 content/$RESTART_1440 >round172_outputs.sha256
)
validate_existing
