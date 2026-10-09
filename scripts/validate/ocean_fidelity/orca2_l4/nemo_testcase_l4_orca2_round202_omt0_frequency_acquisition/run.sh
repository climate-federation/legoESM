#!/usr/bin/env bash
# ORCA2 round-202 acquisition: write OMT-0 restarts every step in NEMO frequency mode.
set -Eeuo pipefail

refuse_unexpected() {
  status=$?
  printf 'REFUSE: round-202 OMT-0 frequency acquisition failed at line %s (exit %s)\n' \
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
smoke=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round199/acquisition/orca2_omt0_smoke_2step_np2
prior=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round200/acquisition
month=$prior/orca2_omt0_month_boundary_96step_np2
evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round202/acquisition
canonical=$evidence/omt0_namelist_cfg
twin_a=$evidence/orca2_omt0_frequency_10step_a_np2
twin_b=$evidence/orca2_omt0_frequency_10step_b_np2
gate=$here/../nemo_testcase_l4_orca2_round199_omt0_record_gate.py
prereg=$repo/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round202.md

gate_sha=ef3f79331016ee890cc6de2469777051e1f1b71ec2ae58d82fb6ee2355787b75
prereg_sha=54271e44004e69be9f35f1b12596932504abf0487860615362ead0e252a9d70e

cd "$repo"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63;
}
for path in "$0" "$gate" "$prereg"; do
  git ls-files --error-unmatch "${path#"$repo"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 65;
  }
  git diff --quiet HEAD -- "$path" || {
    printf 'REFUSE: acquisition artifact differs from HEAD: %s\n' "$path" >&2; exit 65;
  }
done
[[ "$(sha256sum "$gate" | awk '{print $1}')" == "$gate_sha" ]] || {
  printf 'REFUSE: OMT-0 frequency gate content changed\n' >&2; exit 66;
}
[[ "$(sha256sum "$prereg" | awk '{print $1}')" == "$prereg_sha" ]] || {
  printf 'REFUSE: round-202 preregistration content changed\n' >&2; exit 66;
}
[[ "$(sha256sum "$source_deck" | awk '{print $1}')" == \
  b627f4e2d94e4619dbfa27f39b811732497f032b73be86adcc57ddca2dee0e91 ]] || {
  printf 'REFUSE: admitted rung-0 source deck changed\n' >&2; exit 66;
}
[[ "$(sha256sum "$base/nemo" | awk '{print $1}')" == \
  c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343 ]] || {
  printf 'REFUSE: admitted scalar-math binary changed\n' >&2; exit 66;
}
(cd "$base" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)

mkdir -p "$evidence"
[[ -d "$evidence" && ! -L "$evidence" ]] || {
  printf 'REFUSE: evidence root is not a real directory\n' >&2; exit 65;
}
export PYTHONPATH=$repo:$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

bash -n "$0"
"$py" -m py_compile "$gate"
"$py" "$gate" --source "$source_deck" --render-deck "$canonical"
"$py" "$gate" --source "$source_deck" --candidate "$canonical" --cpp "$cpp" \
  --output "$evidence/omt0_deck_preflight.json"
"$py" "$gate" --preflight-only --output "$evidence/omt0_frequency_preflight.json"
"$py" "$gate" --candidate "$canonical" \
  --render-frequency-deck "$evidence/omt0_frequency_namelist_preflight" --itend 10
for plant in extra-delta live-module; do
  if "$py" "$gate" --source "$source_deck" --candidate "$canonical" --cpp "$cpp" \
    --plant "$plant" >"$evidence/deck_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: deck %s plant stayed green\n' "$plant" >&2; exit 72
  fi
  grep -q 'STATUS PLANT-FIRED' "$evidence/deck_${plant}_plant.log"
done
if "$py" - "$gate" "$canonical" <<'PYADJ' >"$evidence/deck_adjacent-list_plant.log" 2>&1
import importlib.util
import pathlib
import sys
spec = importlib.util.spec_from_file_location("omt0_gate", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.render_run_deck(
    pathlib.Path(sys.argv[2]).read_text(), itend=10, stock=10,
    restart_steps=tuple(range(1, 11)),
)
PYADJ
then
  printf 'REFUSE: adjacent restart-list plant stayed green\n' >&2; exit 72
fi
grep -q 'restart-list first step must be later than nit000' "$evidence/deck_adjacent-list_plant.log"
printf 'STATUS PLANT-FIRED: adjacent restart list cannot open its first file\n' >>"$evidence/deck_adjacent-list_plant.log"

if [[ "$mode" == --preflight-only ]]; then
  printf 'ORCA2_ROUND202_OMT0_FREQUENCY_PREFLIGHT_READY %s\n' "$twin_a"
  exit 0
fi

admit() {
  for plant in oversized-list twin-list-mode missing-smoke missing-rank wrong-step \
    nonfinite twin-ulp calibration-ulp missing-month wrong-oracle-stop changed-binary; do
    if "$py" "$gate" --candidate "$canonical" --smoke "$smoke" \
      --twin-a "$twin_a" --twin-b "$twin_b" --month "$month" --plant "$plant" \
      >"$evidence/record_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: record %s plant stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$evidence/record_${plant}_plant.log"
  done
  "$py" "$gate" --candidate "$canonical" --smoke "$smoke" \
    --twin-a "$twin_a" --twin-b "$twin_b" --month "$month" \
    --output "$evidence/omt0_record_admission.json"
  grep -q 'PASS_R202_OMT0_TEN_STEP_RECORD__STOP_MONTH_AT_KT11' \
    "$evidence/omt0_record_admission.json" || {
      printf 'REFUSE: OMT-0 ten-step record or oracle boundary is incomplete\n' >&2; exit 73;
    }
  (cd "$evidence" && sha256sum omt0_*json deck_*_plant.log record_*_plant.log \
    "$twin_a"/ORCA2_000000{01,02,03,04,05,06,07,08,09,10}_restart_*.nc \
    "$twin_b"/ORCA2_000000{01,02,03,04,05,06,07,08,09,10}_restart_*.nc \
    "$month"/ORCA2_00000010_restart_*.nc "$month"/output.abort_0000.nc \
    "$month"/ocean.output "$month"/run.user.stdout.log "$month"/run.user.time.log \
    >ROUND202_SHA256SUMS)
  printf 'ORCA2_ROUND202_OMT0_TEN_STEP_PASS_MONTH_STOP_KT11 %s\n' "$month"
}

if [[ "$mode" == --admit-existing ]]; then
  [[ -d "$twin_a" && -d "$twin_b" ]] || {
    printf 'REFUSE: one or more OMT-0 frequency targets are absent\n' >&2; exit 68;
  }
  admit
  exit 0
fi

for target in "$twin_a" "$twin_b"; do
  [[ ! -e "$target" ]] || {
    printf 'REFUSE: OMT-0 frequency target exists; use --admit-existing: %s\n' "$target" >&2; exit 68;
  }
done
free_kb=$(df -Pk "$evidence" | awk 'NR==2 {print $4}')
[[ "$free_kb" -ge 6291456 ]] || {
  printf 'REFUSE: evidence volume has under 6 GiB free\n' >&2; exit 67;
}

stage_frequency() {
  target=$1
  mkdir "$target"
  while read -r digest name; do cp -a "$base/$name" "$target/$name"; done <"$base/deck_files.sha256"
  while read -r digest name; do cp -a "$base/$name" "$target/$name"; done <"$base/input_files.sha256"
  cp "$base/nemo" "$target/nemo"
  "$py" "$gate" --candidate "$canonical" --render-frequency-deck "$target/namelist_cfg" --itend 10
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

stage_frequency "$twin_a"
run_clean "$twin_a"
stage_frequency "$twin_b"
run_clean "$twin_b"
admit
