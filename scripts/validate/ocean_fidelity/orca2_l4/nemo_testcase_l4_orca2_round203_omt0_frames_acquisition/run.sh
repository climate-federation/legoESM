#!/usr/bin/env bash
# ORCA2 round-203 acquisition: additions-only OMT-0 entry/stage frames.
set -Eeuo pipefail

refuse_unexpected() {
  status=$?
  printf 'REFUSE: round-203 OMT-0 frame acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

mode=${1:---run}
case "$mode" in
  --run|--preflight-only|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(CDPATH= cd -- "$here/../../../../../" && pwd -P)
py=/home/dbalwada/legoESM/.venv/bin/python
nemo_root=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
cpp=$nemo_root/cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm
source_deck=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/acquisition/rung0_namelist_cfg
base=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/acquisition/orca2_rung0_restart_list_repair_240step_np2
instrument_source=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round90/acquisition/orca2_rung0_entry_stage_runoff_guarded_10step_np2
month=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round200/acquisition/orca2_omt0_month_boundary_96step_np2
failed_frequency=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round202/acquisition/orca2_omt0_frequency_10step_a_np2
evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round203/acquisition
canonical=$evidence/omt0_namelist_cfg
calibration=$evidence/orca2_omt0_uninstrumented_10step_np2
twin_a=$evidence/orca2_omt0_frames_10step_a_np2
twin_b=$evidence/orca2_omt0_frames_10step_b_np2
deck_gate=$here/../nemo_testcase_l4_orca2_round199_omt0_record_gate.py
record_gate=$here/../nemo_testcase_l4_orca2_round203_omt0_frame_record_gate.py
prereg=$repo/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round203.md

gate_sha=d2bede62b04623866e21bdacdbedc174de1c14878c58486b20281300955ae4e8
prereg_sha=13ac40b178346cb3d27b81281393892fec2d57d34f3c229e11ad6c06f29a5ad9
base_binary_sha=c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343
instrument_binary_sha=b54b37788058697e6f649d16bf9f3843bc9bb672c05bdf06404c946c659ac6a9

cd "$repo"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63;
}
for path in "$0" "$deck_gate" "$record_gate" "$prereg"; do
  git ls-files --error-unmatch "${path#"$repo"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 65;
  }
  git diff --quiet HEAD -- "$path" || {
    printf 'REFUSE: acquisition artifact differs from HEAD: %s\n' "$path" >&2; exit 65;
  }
done
[[ "$(sha256sum "$record_gate" | awk '{print $1}')" == "$gate_sha" ]] || {
  printf 'REFUSE: round-203 frame gate content changed\n' >&2; exit 66;
}
[[ "$(sha256sum "$prereg" | awk '{print $1}')" == "$prereg_sha" ]] || {
  printf 'REFUSE: round-203 preregistration content changed\n' >&2; exit 66;
}
[[ "$(sha256sum "$source_deck" | awk '{print $1}')" == \
  b627f4e2d94e4619dbfa27f39b811732497f032b73be86adcc57ddca2dee0e91 ]] || {
  printf 'REFUSE: admitted rung-0 source deck changed\n' >&2; exit 66;
}
[[ "$(sha256sum "$base/nemo" | awk '{print $1}')" == "$base_binary_sha" ]] || {
  printf 'REFUSE: admitted uninstrumented binary changed\n' >&2; exit 66;
}
[[ "$(sha256sum "$instrument_source/nemo" | awk '{print $1}')" == "$instrument_binary_sha" ]] || {
  printf 'REFUSE: admitted frame binary changed\n' >&2; exit 66;
}
(cd "$base" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)

mkdir -p "$evidence"
[[ -d "$evidence" && ! -L "$evidence" ]] || {
  printf 'REFUSE: evidence root is not a real directory\n' >&2; exit 65;
}
export PYTHONPATH=$repo:$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

bash -n "$0"
"$py" -m py_compile "$deck_gate" "$record_gate"
"$py" "$deck_gate" --source "$source_deck" --render-deck "$canonical"
"$py" "$deck_gate" --source "$source_deck" --candidate "$canonical" --cpp "$cpp" \
  --output "$evidence/omt0_deck_preflight.json"
"$py" "$record_gate" --preflight-only --output "$evidence/omt0_frame_preflight.json"

if "$py" "$deck_gate" --candidate "$canonical" \
  --render-frequency-deck "$evidence/retracted_frequency_namelist" --itend 10 \
  >"$evidence/retracted_frequency_plant.log" 2>&1; then
  printf 'REFUSE: retracted nn_stock=1 protocol stayed green\n' >&2; exit 72
fi
grep -q 'nn_stock=1 is rejected' "$evidence/retracted_frequency_plant.log"
printf 'STATUS PLANT-FIRED: compiled cadence rejects nn_stock=1\n' \
  >>"$evidence/retracted_frequency_plant.log"

"$py" - "$failed_frequency" <<'PYFAILED'
from pathlib import Path
from netCDF4 import Dataset
import sys
root = Path(sys.argv[1])
ocean = (root / "ocean.output").read_text()
assert "nn_stock (           1 ) is NOT a multiple of nn_fsbc (           2 )" in ocean
files = sorted(root.glob("ORCA2_00000001_restart_*.nc"))
assert len(files) == 2
for path in files:
    with Dataset(path) as ds:
        assert int(ds["kt"][:]) == 1
        assert all(name not in ds.variables for name in ("sshn", "un", "vn", "tn", "sn"))
print("STATUS PASS_R203_R202_HEADER_ONLY_REFUSAL")
PYFAILED

if [[ "$mode" == --preflight-only ]]; then
  printf 'ORCA2_ROUND203_OMT0_FRAMES_PREFLIGHT_READY %s\n' "$twin_a"
  exit 0
fi

admit() {
  local plant
  for plant in cadence header field-name truncation nonfinite missing-frame \
    twin-ulp terminal-byte changed-binary; do
    if "$py" "$record_gate" --candidate "$canonical" --calibration "$calibration" \
      --twin-a "$twin_a" --twin-b "$twin_b" --month "$month" --plant "$plant" \
      >"$evidence/record_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: record %s plant stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$evidence/record_${plant}_plant.log"
  done
  "$py" "$record_gate" --candidate "$canonical" --calibration "$calibration" \
    --twin-a "$twin_a" --twin-b "$twin_b" --month "$month" \
    --output "$evidence/omt0_frame_record_admission.json"
  grep -q 'PASS_R203_OMT0_ENTRY_STAGE_RECORD__STOP_MONTH_AT_KT11' \
    "$evidence/omt0_frame_record_admission.json" || {
      printf 'REFUSE: OMT-0 entry/stage record is incomplete\n' >&2; exit 73;
    }
  (cd "$evidence" && sha256sum omt0_*json *_plant.log \
    "$calibration"/ORCA2_00000010_restart_*.nc \
    "$twin_a"/ORCA2_00000010_restart_*.nc "$twin_b"/ORCA2_00000010_restart_*.nc \
    "$twin_a"/oracle_r84_frame_*.bin "$twin_b"/oracle_r84_frame_*.bin \
    >ROUND203_SHA256SUMS)
  printf 'ORCA2_ROUND203_OMT0_ENTRY_STAGE_RECORD_PASS %s\n' "$twin_a"
}

if [[ "$mode" == --admit-existing ]]; then
  [[ -d "$calibration" && -d "$twin_a" && -d "$twin_b" ]] || {
    printf 'REFUSE: one or more round-203 targets are absent\n' >&2; exit 68;
  }
  admit
  exit 0
fi

for target in "$calibration" "$twin_a" "$twin_b"; do
  [[ ! -e "$target" ]] || {
    printf 'REFUSE: target exists; use --admit-existing: %s\n' "$target" >&2; exit 68;
  }
done
free_kb=$(df -Pk "$evidence" | awk 'NR==2 {print $4}')
[[ "$free_kb" -ge 6291456 ]] || {
  printf 'REFUSE: evidence volume has under 6 GiB free\n' >&2; exit 67;
}

stage_run() {
  target=$1
  binary=$2
  mkdir "$target"
  while read -r digest name; do cp -a "$base/$name" "$target/$name"; done <"$base/deck_files.sha256"
  while read -r digest name; do cp -a "$base/$name" "$target/$name"; done <"$base/input_files.sha256"
  cp "$binary" "$target/nemo"
  "$py" "$deck_gate" --candidate "$canonical" --render-run-deck "$target/namelist_cfg" \
    --itend 10 --stock 10 --restart-steps 10
  while read -r digest name; do (cd "$target" && sha256sum "$name"); done \
    <"$base/deck_files.sha256" >"$target/deck_files.sha256"
  while read -r digest name; do (cd "$target" && sha256sum "$name"); done \
    <"$base/input_files.sha256" >"$target/input_files.sha256"
  (cd "$target" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
  (cd "$target" && sha256sum nemo namelist_cfg deck_files.sha256 input_files.sha256 >producer_inputs.sha256)
}

run_clean() {
  target=$1
  (
    cd "$target"
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
    started=$SECONDS
    set +e
    mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
    pipe_rc=("${PIPESTATUS[@]}")
    set -e
    [[ "${pipe_rc[0]}" -eq 0 && "${pipe_rc[1]:-0}" -eq 0 ]] || {
      printf 'REFUSE: NEMO clean pipeline failed (%s,%s)\n' "${pipe_rc[0]}" "${pipe_rc[1]:-0}" >&2; exit 70;
    }
    grep -q 'STOP 0' run.user.stdout.log
    printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  )
}

stage_run "$calibration" "$base/nemo"
run_clean "$calibration"
stage_run "$twin_a" "$instrument_source/nemo"
run_clean "$twin_a"
stage_run "$twin_b" "$instrument_source/nemo"
run_clean "$twin_b"
admit
