#!/usr/bin/env bash
# ORCA2 round-199 acquisition: Decision-103 mini-ladder rung OMT-0.
set -Eeuo pipefail

refuse_unexpected() {
  status=$?
  printf 'REFUSE: round-199 OMT-0 acquisition failed at line %s (exit %s)\n' \
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
evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round199/acquisition
canonical=$evidence/omt0_namelist_cfg
smoke=$evidence/orca2_omt0_smoke_2step_np2
twin_a=$evidence/orca2_omt0_record_12step_a_np2
twin_b=$evidence/orca2_omt0_record_12step_b_np2
month=$evidence/orca2_omt0_month_96step_np2
gate=$here/../nemo_testcase_l4_orca2_round199_omt0_record_gate.py
prereg=$repo/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round199.md

# Updated only after review; pins dependencies by content, never by moving commit SHA.
gate_sha=83fa2e579ee6acdcfdc5b69342dd9ceae766ffadd10a945ee19e5b6f16213f31
prereg_sha=3acdaca72aa05b604228dd644a0ac2a04dec636042a29178c5526076347ba530

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
  printf 'REFUSE: OMT-0 gate content changed\n' >&2; exit 66;
}
[[ "$(sha256sum "$prereg" | awk '{print $1}')" == "$prereg_sha" ]] || {
  printf 'REFUSE: round-199 preregistration content changed\n' >&2; exit 66;
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
"$py" "$gate" --preflight-only --output "$evidence/omt0_record_preflight.json"
"$py" "$gate" --candidate "$canonical" --render-run-deck "$evidence/omt0_smoke_namelist_preflight" \
  --itend 2 --stock 2 --restart-steps 1
"$py" "$gate" --candidate "$canonical" --render-run-deck "$evidence/omt0_twin_namelist_preflight" \
  --itend 12 --stock 12 --restart-steps 1,2,3,4,5,6,7,8,9,10
"$py" "$gate" --candidate "$canonical" --render-run-deck "$evidence/omt0_month_namelist_preflight" \
  --itend 96 --stock 96 --restart-steps 10,20,30,40,50,60,70,80,90,95
for plant in extra-delta live-module; do
  if "$py" "$gate" --source "$source_deck" --candidate "$canonical" --cpp "$cpp" \
    --plant "$plant" >"$evidence/deck_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: deck %s plant stayed green\n' "$plant" >&2; exit 72
  fi
  grep -q 'STATUS PLANT-FIRED' "$evidence/deck_${plant}_plant.log"
done
if "$py" - "$gate" <<'PYLIST' >"$evidence/deck_oversized-list_plant.log" 2>&1
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("omt0_gate", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.render_run_deck("", itend=12, stock=12, restart_steps=tuple(range(1, 12)))
PYLIST
then
  printf 'REFUSE: oversized-list plant stayed green\n' >&2; exit 72
fi
printf 'STATUS PLANT-FIRED: restart list exceeds compiled capacity 10\n' >>"$evidence/deck_oversized-list_plant.log"

if [[ "$mode" == --preflight-only ]]; then
  printf 'ORCA2_ROUND199_OMT0_PREFLIGHT_READY %s\n' "$smoke"
  exit 0
fi

admit() {
  for plant in oversized-list missing-smoke missing-rank wrong-step nonfinite \
    twin-ulp calibration-ulp missing-month changed-binary; do
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
  grep -q 'PASS_R199_OMT0_RECORD' "$evidence/omt0_record_admission.json" || {
    printf 'REFUSE: OMT-0 record is incomplete\n' >&2; exit 73;
  }
  (cd "$evidence" && sha256sum omt0_*json deck_*_plant.log record_*_plant.log \
    "$smoke"/ORCA2_00000002_restart_*.nc \
    "$twin_a"/ORCA2_000000{01,02,03,04,05,06,07,08,09,10,12}_restart_*.nc \
    "$twin_b"/ORCA2_000000{01,02,03,04,05,06,07,08,09,10,12}_restart_*.nc \
    "$month"/ORCA2_000000{10,20,30,40,50,60,70,80,90,95,96}_restart_*.nc \
    >ROUND199_SHA256SUMS)
  printf 'ORCA2_ROUND199_OMT0_RECORD_PASS %s\n' "$month"
}

if [[ "$mode" == --admit-existing ]]; then
  [[ -d "$smoke" && -d "$twin_a" && -d "$twin_b" && -d "$month" ]] || {
    printf 'REFUSE: one or more OMT-0 targets are absent\n' >&2; exit 68;
  }
  admit
  exit 0
fi

for target in "$smoke" "$twin_a" "$twin_b" "$month"; do
  [[ ! -e "$target" ]] || {
    printf 'REFUSE: OMT-0 target exists; use --admit-existing: %s\n' "$target" >&2; exit 68;
  }
done
free_kb=$(df -Pk "$evidence" | awk 'NR==2 {print $4}')
[[ "$free_kb" -ge 6291456 ]] || {
  printf 'REFUSE: evidence volume has under 6 GiB free\n' >&2; exit 67;
}

stage() {
  target=$1 itend=$2 stock=$3 restart_steps=$4
  mkdir "$target"
  while read -r digest name; do cp -a "$base/$name" "$target/$name"; done <"$base/deck_files.sha256"
  while read -r digest name; do cp -a "$base/$name" "$target/$name"; done <"$base/input_files.sha256"
  cp "$base/nemo" "$target/nemo"
  "$py" "$gate" --candidate "$canonical" --render-run-deck "$target/namelist_cfg" \
    --itend "$itend" --stock "$stock" --restart-steps "$restart_steps"
  while read -r digest name; do (cd "$target" && sha256sum "$name"); done \
    <"$base/deck_files.sha256" >"$target/deck_files.sha256"
  while read -r digest name; do (cd "$target" && sha256sum "$name"); done \
    <"$base/input_files.sha256" >"$target/input_files.sha256"
  (cd "$target" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
  (cd "$target" && sha256sum nemo namelist_cfg deck_files.sha256 input_files.sha256 >producer_inputs.sha256)
}

run_target() {
  target=$1
  (
    cd "$target"
    for output in ocean.output time.step run.user.stdout.log run.user.time.log; do
      [[ ! -e "$output" ]] || {
        printf 'REFUSE: output exists: %s/%s\n' "$target" "$output" >&2; exit 69;
      }
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

stage "$smoke" 2 2 1
run_target "$smoke"
stage "$twin_a" 12 12 1,2,3,4,5,6,7,8,9,10
run_target "$twin_a"
stage "$twin_b" 12 12 1,2,3,4,5,6,7,8,9,10
run_target "$twin_b"
stage "$month" 96 96 10,20,30,40,50,60,70,80,90,95
run_target "$month"
admit
