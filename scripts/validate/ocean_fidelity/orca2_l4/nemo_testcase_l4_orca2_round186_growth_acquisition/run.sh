#!/usr/bin/env bash
# ORCA2 round-186 acquisition: rung-0 restart checkpoints for growth scoring.
set -Eeuo pipefail

refuse_unexpected() {
  status=$?
  printf 'REFUSE: round-186 growth acquisition failed at line %s (exit %s)\n' \
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
evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round186/acquisition
source=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/acquisition/rung0_namelist_cfg
base=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/acquisition/orca2_rung0_restart_list_repair_240step_np2
calibration=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2
target_a=$evidence/orca2_rung0_growth_96step_a_np2
target_b=$evidence/orca2_rung0_growth_96step_b_np2
gate=$here/../nemo_testcase_l4_orca2_round186_growth_record_gate.py
prereg=$repo/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round186.md
gate_sha=9fa9a1f9fff8d0c2cae7499df30c47a9417c42e0588a8ab14f0dc39d98207f17
prereg_sha=d60cee6b31bcaef09eefbea2b91909b5439e7cfac66aeee006a36f134024a209

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
  printf 'REFUSE: growth gate content changed\n' >&2; exit 66;
}
[[ "$(sha256sum "$prereg" | awk '{print $1}')" == "$prereg_sha" ]] || {
  printf 'REFUSE: preregistration content changed\n' >&2; exit 66;
}
[[ -f "$source" && -f "$base/nemo" ]] || {
  printf 'REFUSE: admitted rung-0 source record is missing\n' >&2; exit 65;
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
"$py" "$gate" --render-source "$source" --render-output "$evidence/namelist_growth_preflight" \
  >"$evidence/deck_preflight.log"

if [[ "$mode" == --preflight-only ]]; then
  printf 'ORCA2_ROUND186_GROWTH_PREFLIGHT_READY %s\n' "$target_a"
  exit 0
fi

admit() {
  for plant in missing-rank twin-ulp step10-calibration hidden-deck; do
    if "$py" "$gate" --source "$source" \
      --deck-a "$target_a/namelist_cfg" --deck-b "$target_b/namelist_cfg" \
      --twin-a "$target_a" --twin-b "$target_b" --calibration "$calibration" \
      --plant "$plant" >"$evidence/record_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$evidence/record_${plant}_plant.log"
  done
  "$py" "$gate" --source "$source" \
    --deck-a "$target_a/namelist_cfg" --deck-b "$target_b/namelist_cfg" \
    --twin-a "$target_a" --twin-b "$target_b" --calibration "$calibration" \
    --output "$evidence/growth_record_admission.json"
  (cd "$evidence" && sha256sum growth_record_admission.json record_*_plant.log \
    "$target_a"/ORCA2_*_restart_*.nc "$target_b"/ORCA2_*_restart_*.nc \
    >ROUND186_SHA256SUMS)
  printf 'ORCA2_ROUND186_GROWTH_RECORD_PASS %s\n' "$target_a"
}

if [[ "$mode" == --admit-existing ]]; then
  [[ -d "$target_a" && -d "$target_b" ]] || {
    printf 'REFUSE: growth targets do not both exist\n' >&2; exit 68;
  }
  admit
  exit 0
fi

[[ ! -e "$target_a" && ! -e "$target_b" ]] || {
  printf 'REFUSE: growth target exists; use --admit-existing\n' >&2; exit 68;
}
free_kb=$(df -Pk "$evidence" | awk 'NR==2 {print $4}')
[[ "$free_kb" -ge 6291456 ]] || {
  printf 'REFUSE: evidence volume has under 6 GiB free\n' >&2; exit 67;
}

stage() {
  target=$1
  mkdir "$target"
  while read -r digest name; do cp -a "$base/$name" "$target/$name"; done <"$base/deck_files.sha256"
  while read -r digest name; do cp -a "$base/$name" "$target/$name"; done <"$base/input_files.sha256"
  cp "$base/nemo" "$target/nemo"
  "$py" "$gate" --render-source "$source" --render-output "$target/namelist_cfg" >/dev/null
  printf '%s\n' "$(git rev-parse HEAD)" >"$target/producer_commit.txt"
  (cd "$target" && sha256sum nemo namelist_cfg >growth_inputs.sha256)
}

run_target() {
  target=$1
  (
    cd "$target"
    for output in ocean.output time.step run.user.stdout.log run.user.time.log; do
      [[ ! -e "$output" ]] || { printf 'REFUSE: output exists: %s/%s\n' "$target" "$output" >&2; exit 69; }
    done
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
    started=$SECONDS
    set +e
    mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
    pipe_rc=("${PIPESTATUS[@]}")
    set -e
    [[ "${pipe_rc[0]}" -eq 0 && "${pipe_rc[1]:-0}" -eq 0 ]] || {
      printf 'REFUSE: NEMO pipeline failed (%s,%s)\n' "${pipe_rc[0]}" "${pipe_rc[1]:-0}" >&2; exit 70;
    }
    printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  )
  grep -q 'STOP 0' "$target/run.user.stdout.log" || {
    printf 'REFUSE: NEMO did not report STOP 0: %s\n' "$target" >&2; exit 70;
  }
}

stage "$target_a"
run_target "$target_a"
stage "$target_b"
run_target "$target_b"
admit
