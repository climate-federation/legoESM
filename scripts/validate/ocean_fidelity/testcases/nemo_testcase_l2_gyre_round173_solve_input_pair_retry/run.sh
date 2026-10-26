#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: unexpected Round-173 failure at line %s (exit %s)\n' \
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
  --run|--preflight-only|--plant-wet-mask|--admit-existing) ;;
  *) refuse 64 "usage: $0 [--run|--preflight-only|--plant-wet-mask|--admit-existing]" ;;
esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
export PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
readonly SOURCE_RUN=$L2/round125/oracle_vertical_decomposition
readonly BASELINE_RUN=$L2/round172/oracle_solve_input_pair/baseline
readonly FAILED_RUN=$L2/round172/oracle_solve_input_pair/e3t
readonly TARGET_RUN=$L2/round173/oracle_solve_input_pair_wet_retry
readonly INPUT_ROOT=$L2/round172/lego_solve_inputs
readonly RAW_BYTES=167132160
readonly RESTART_1080=GYRE_OMIP_L2_P3_00001080_restart.nc
readonly RESTART_1440=GYRE_OMIP_L2_P3_00001440_restart.nc
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round173.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || \
  refuse 63 "acquisition requires a clean committed tree"
readonly COMMIT=$(git rev-parse HEAD)

for path in "$PREREG" "$BINARY" "$BASELINE_RUN/nemo" \
    "$FAILED_RUN/nemo" "$SOURCE_RUN/$RESTART_1080" \
    "$SOURCE_RUN/$RESTART_1440" "$BASELINE_RUN/$RESTART_1080" \
    "$BASELINE_RUN/$RESTART_1440" "$INPUT_ROOT/e3t_Kaa.npy" \
    "$INPUT_ROOT/content_T.npy" "$INPUT_ROOT/manifest.json" \
    "$INPUT_ROOT/trace_files.sha256" "$INPUT_ROOT/trace_files.stamp" \
    "$SOURCE_RUN/vertical_records.sha256" \
    "$SOURCE_RUN/vertical_records.stamp" \
    "$FAILED_RUN/round172_solve_inputs.raw" "$FAILED_RUN/ocean.output"; do
  [[ -f "$path" ]] || refuse 64 "missing required input $path"
done

cmp -s "$BINARY" "$BASELINE_RUN/nemo" || \
  refuse 65 "installed Round-172 binary differs from baseline copy"
cmp -s "$BINARY" "$FAILED_RUN/nemo" || \
  refuse 65 "installed Round-172 binary differs from failed-arm copy"
grep -Fxq 'STOP 0' "$BASELINE_RUN/run.user.stdout.log" || \
  refuse 65 "existing baseline lacks STOP 0"
cmp -s "$SOURCE_RUN/$RESTART_1080" "$BASELINE_RUN/$RESTART_1080" || \
  refuse 65 "existing baseline day-180 restart moved"
cmp -s "$SOURCE_RUN/$RESTART_1440" "$BASELINE_RUN/$RESTART_1440" || \
  refuse 65 "existing baseline day-240 restart moved"
grep -Fq 'kt 1082 |ssh| max' "$FAILED_RUN/ocean.output" || \
  refuse 65 "failed e3t arm does not report its step-1082 NaN"

readonly DRY=$(mktemp -d /tmp/gyre-r173-wet-input.XXXXXXXX)
readonly RAW=$DRY/round172_solve_inputs.raw
if "$PY" - "$REPO" "$SOURCE_RUN" "$INPUT_ROOT" "$FAILED_RUN/round172_solve_inputs.raw" "$RAW" "$MODE" <<'PYINPUT'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

repo, source_root, input_root, failed_raw, raw, mode = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
    Path(sys.argv[4]), Path(sys.argv[5]), sys.argv[6])
sys.path.insert(0, str(repo))
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_l2_gyre_round35_trazdf_matrix import (  # noqa: E402
    read_trazdf_matrix,
)

manifest = json.loads((input_root / "manifest.json").read_text())
if manifest.get("format") != "gyre-round172-developed-solve-input-record-v1":
    raise SystemExit("REFUSE: solve-input manifest has the wrong format")
if manifest.get("shape") != [360, 22, 32, 30]:
    raise SystemExit(f"REFUSE: solve-input shape is {manifest.get('shape')}")

input_manifest = input_root / "trace_files.sha256"
input_stamp = (input_root / "trace_files.stamp").read_text().split()
input_digest = hashlib.sha256(input_manifest.read_bytes()).hexdigest()
if (len(input_stamp) != 3 or input_stamp[0] != input_digest
        or input_stamp[2] != input_manifest.name):
    raise SystemExit("REFUSE: solve-input commit stamp disagrees")
input_rows = {}
for line in input_manifest.read_text().splitlines():
    sha, name = line.split()
    input_rows[Path(name).name] = sha
for name in ("e3t_Kaa.npy", "content_T.npy", "manifest.json"):
    observed = hashlib.sha256((input_root / name).read_bytes()).hexdigest()
    if input_rows.get(name) != observed:
        raise SystemExit(f"REFUSE: solve-input hash disagrees for {name}")

vertical_manifest = source_root / "vertical_records.sha256"
vertical_stamp = (source_root / "vertical_records.stamp").read_text().split()
vertical_digest = hashlib.sha256(vertical_manifest.read_bytes()).hexdigest()
if (len(vertical_stamp) != 3 or vertical_stamp[0] != vertical_digest
        or vertical_stamp[2] != vertical_manifest.name):
    raise SystemExit("REFUSE: Round-125 vertical-record stamp disagrees")
vertical_rows = {}
for line in vertical_manifest.read_text().splitlines():
    sha, name = line.split()
    vertical_rows[Path(name).name] = sha

e3t = np.load(input_root / "e3t_Kaa.npy", mmap_mode="r")
content = np.load(input_root / "content_T.npy", mmap_mode="r")
factor = np.float64(1.0 + 2.0**-20)
selected_total = 0
retained_total = 0
first_selected_changed = {"e3t": 0, "content": 0}

with raw.open("wb") as handle:
    for frame, step in enumerate(range(1081, 1441)):
        name = f"oracle_trazdf_matrix_kt{step:08d}.bin"
        path = source_root / name
        if not path.is_file():
            raise SystemExit(f"REFUSE: Round-125 record lacks {name}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if vertical_rows.get(name) != observed:
            raise SystemExit(f"REFUSE: Round-125 record hash disagrees for {name}")
        record = read_trazdf_matrix(path, expect_kt=step)
        header = record["header"]
        if ((header["ntsi"], header["ntei"], header["ntsj"], header["ntej"])
                != (3, 34, 3, 24)):
            raise SystemExit(f"REFUSE: {name} has unexpected interior bounds")
        arrays = record["arrays"]
        mask = np.asarray(arrays["tmask"][:, :, :30], dtype=bool)
        if int(mask.sum()) != 18000:
            raise SystemExit(
                f"REFUSE: {name} has {int(mask.sum())} wet cells, expected 18000")
        active = np.zeros((36, 26, 30), dtype=bool)
        active[2:34, 2:24, :] = True
        retained = active & ~mask
        if int(retained.sum()) != 3120:
            raise SystemExit(
                f"REFUSE: {name} has {int(retained.sum())} retained dry cells, expected 3120")

        lego_e3t = np.zeros((36, 26, 31), dtype="<f8", order="F")
        lego_content = np.zeros((36, 26, 31), dtype="<f8", order="F")
        lego_e3t[2:34, 2:24, :30] = np.transpose(e3t[frame], (1, 0, 2))
        lego_content[2:34, 2:24, :30] = np.transpose(content[frame], (1, 0, 2))
        wet = np.zeros((36, 26, 31), dtype=bool)
        wet[:, :, :30] = mask
        if frame == 0:
            failed_e3t = np.fromfile(
                failed_raw, dtype="<f8", count=36 * 26 * 31,
            ).reshape((36, 26, 31), order="F")
            dry_active = np.zeros((36, 26, 31), dtype=bool)
            dry_active[:, :, :30] = retained
            oracle_e3t = np.asarray(arrays["e3t_Kaa"], dtype="<f8")
            dry_zero = int(np.count_nonzero(failed_e3t[dry_active] == 0.0))
            dry_positive = int(np.count_nonzero(oracle_e3t[dry_active] > 0.0))
            if (dry_zero, dry_positive) != (3120, 3120):
                raise SystemExit(
                    "REFUSE: failed-input dry-cell diagnosis does not close: "
                    f"zero={dry_zero} oracle_positive={dry_positive}")
            print(
                "FAILED_INPUT_DIAGNOSIS step=1081 dry_zero_replacements=3120 "
                "dry_oracle_positive=3120 failure_step=1082")

        outputs = []
        for label, oracle_name, lego in (
                ("e3t", "e3t_Kaa", lego_e3t),
                ("content", "rhs_T", lego_content)):
            oracle = np.asarray(arrays[oracle_name], dtype="<f8")
            corrected = np.array(oracle, dtype="<f8", order="F", copy=True)
            source = np.array(lego, copy=True)
            if mode == "--plant-wet-mask" and frame == 0:
                source[wet] *= factor
            corrected[wet] = source[wet]
            if not np.array_equal(corrected[~wet], oracle[~wet]):
                raise SystemExit(
                    f"REFUSE: {name} {label} moved an unselected cell")
            if mode == "--plant-wet-mask" and frame == 0:
                ordinary = lego[wet]
                first_selected_changed[label] = int(
                    np.count_nonzero(corrected[wet] != ordinary))
            outputs.append(corrected)
        for output in outputs:
            handle.write(output.tobytes(order="F"))
        selected_total += int(mask.sum())
        retained_total += int(retained.sum())

expected = 167132160
if raw.stat().st_size != expected:
    raise SystemExit(
        f"REFUSE: corrected raw size is {raw.stat().st_size}, expected {expected}")
if selected_total != 360 * 18000 or retained_total != 360 * 3120:
    raise SystemExit(
        "REFUSE: corrected raw aggregate wet/dry census moved")
if mode == "--plant-wet-mask":
    if first_selected_changed != {"e3t": 18000, "content": 18000}:
        raise SystemExit(
            f"REFUSE: wet-mask plant changed {first_selected_changed}")
    print("STATUS PLANT-FIRED wet-mask selected=18000 retained=3120")
    raise SystemExit(2)
print(
    "WET_INPUT_LAYOUT_PASS bytes=167132160 selected_per_frame=18000 "
    "retained_per_frame=3120")
PYINPUT
then
  python_status=0
else
  python_status=$?
fi

if [[ "$MODE" == --plant-wet-mask ]]; then
  [[ "$python_status" -eq 2 ]] || \
    refuse 66 "wet-mask plant returned $python_status instead of 2"
  refuse 2 "wet-mask plant fired"
fi
[[ "$python_status" -eq 0 ]] || \
  refuse 66 "corrected solve-input converter exited $python_status"
[[ "$(stat -c %s "$RAW")" -eq "$RAW_BYTES" ]] || \
  refuse 66 "corrected solve-input byte count moved"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND173_WET_RETRY_PREFLIGHT_READY binary=%s raw=%s\n' \
    "$(sha256sum "$BINARY" | awk '{print $1}')" "$RAW"
  exit 0
fi

validate_existing() {
  for arm in e3t_wet content_wet; do
    local run=$TARGET_RUN/$arm
    [[ -f "$run/$RESTART_1080" && -f "$run/$RESTART_1440" ]] || \
      refuse 70 "existing $arm arm lacks required restarts"
    grep -Fxq 'STOP 0' "$run/run.user.stdout.log" || \
      refuse 70 "existing $arm arm lacks STOP 0"
    cmp -s "$BINARY" "$run/nemo" || \
      refuse 70 "existing $arm executable moved"
    cmp -s "$RAW" "$run/round172_solve_inputs.raw" || \
      refuse 70 "existing $arm corrected input moved"
  done
  cmp -s "$SOURCE_RUN/$RESTART_1080" "$BASELINE_RUN/$RESTART_1080" || \
    refuse 71 "reused baseline day-180 restart moved"
  cmp -s "$SOURCE_RUN/$RESTART_1440" "$BASELINE_RUN/$RESTART_1440" || \
    refuse 71 "reused baseline day-240 restart moved"
  cmp -s "$BASELINE_RUN/$RESTART_1440" "$TARGET_RUN/e3t_wet/$RESTART_1440" && \
    refuse 72 "e3t wet-only arm is non-discriminating at day 240"
  cmp -s "$BASELINE_RUN/$RESTART_1440" "$TARGET_RUN/content_wet/$RESTART_1440" && \
    refuse 72 "content wet-only arm is non-discriminating at day 240"
  printf 'ROUND173_SOLVE_INPUT_PAIR_READY %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  validate_existing
  exit 0
fi

[[ ! -e "$TARGET_RUN" ]] || refuse 64 "new retry run exists: $TARGET_RUN"
mkdir -p "$TARGET_RUN"
readonly PREPARED='namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml'
for arm in e3t_wet content_wet; do
  run=$TARGET_RUN/$arm
  mkdir "$run"
  for name in $PREPARED; do
    [[ -f "$BASELINE_RUN/$name" ]] || refuse 68 "baseline run lacks $name"
    cp -L "$BASELINE_RUN/$name" "$run/$name"
  done
  cp "$BINARY" "$run/nemo"
  cp "$RAW" "$run/round172_solve_inputs.raw"
  if [[ "$arm" == e3t_wet ]]; then
    touch "$run/round172_e3t.arm"
  else
    touch "$run/round172_content.arm"
  fi
  (
    cd "$run"
    export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    started=$SECONDS
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
    set +e
    mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
    nemo_status=${PIPESTATUS[0]}
    set -e
    [[ "$nemo_status" -eq 0 ]] || refuse 69 "$arm NEMO process failed"
    grep -Fxq 'STOP 0' run.user.stdout.log || refuse 69 "$arm lacks STOP 0"
    printf 'wall_seconds=%s\nRUN_DONE\n' "$((SECONDS-started))" >>run.user.time.log
  )
done

printf '%s\n' "$COMMIT" >"$TARGET_RUN/producer_commit.txt"
sha256sum "$BINARY" >"$TARGET_RUN/binary.sha256"
raw_digest=$(sha256sum "$RAW" | awk '{print $1}')
printf '%s  corrected_round172_solve_inputs.raw\n' "$raw_digest" \
  >"$TARGET_RUN/corrected_input.sha256"
(
  cd "$TARGET_RUN"
  sha256sum e3t_wet/$RESTART_1080 e3t_wet/$RESTART_1440 \
    content_wet/$RESTART_1080 content_wet/$RESTART_1440 \
    >round173_outputs.sha256
)
validate_existing
