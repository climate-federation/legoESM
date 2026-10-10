#!/usr/bin/env bash
# Operator-executed Decision-109 OMT-2 acquisition; binary reuse only.
set -Eeuo pipefail

refuse_unexpected() {
  status=$?
  printf 'REFUSE: round-218 OMT-2 acquisition failed at line %s (exit %s)\n' \
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
source_deck=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round211/acquisition/omt1_namelist_cfg
base=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round203/acquisition/orca2_omt0_uninstrumented_10step_np2
instrument_binary=$nemo_root/cfgs/ORCA2_OMIP_L4_R210OMT1_P3/BLD/bin/nemo.exe
compiled_spg=$nemo_root/cfgs/ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90
compiled_zdf=$nemo_root/cfgs/ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90
evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round218/acquisition
canonical=$evidence/omt2_namelist_cfg
smoke=$evidence/orca2_omt2_smoke_2step_np2
calibration=$evidence/orca2_omt2_uninstrumented_10step_np2
twin_a=$evidence/orca2_omt2_frames_10step_a_np2
twin_b=$evidence/orca2_omt2_frames_10step_b_np2
month=$evidence/orca2_omt2_month_boundary_96step_np2
deck_gate=$here/../nemo_testcase_l4_orca2_round218_omt2_deck_gate.py
record_gate=$here/../nemo_testcase_l4_orca2_round218_omt2_frame_record_gate.py
prereg=$repo/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round218.md

base_binary_sha=c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343
instrument_binary_sha=5b82a3254c40f71186af159b93cba419709ccf49cf4172ad3d44440b8fb1d895

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
[[ "$(sha256sum "$source_deck" | awk '{print $1}')" == \
  d0cccebd76c9c4631c91e51a1af3a9b552d97f307c7c523020160c883fb5e73f ]] || {
  printf 'REFUSE: admitted OMT-1 source deck changed\n' >&2; exit 66;
}
[[ "$(sha256sum "$base/nemo" | awk '{print $1}')" == "$base_binary_sha" ]] || {
  printf 'REFUSE: admitted uninstrumented binary changed\n' >&2; exit 66;
}
[[ "$(sha256sum "$instrument_binary" | awk '{print $1}')" == "$instrument_binary_sha" ]] || {
  printf 'REFUSE: admitted frame binary changed\n' >&2; exit 66;
}
(cd "$base" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
grep -q 'CALL r84_dump_frame' \
  "$nemo_root/cfgs/ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3.f90" || {
  printf 'REFUSE: compiled P3 frame call is absent\n' >&2; exit 66;
}
grep -q 'IF( ln_dynadv_vec .OR. lk_linssh )' "$compiled_spg" || {
  printf 'REFUSE: compiled vector-form branch is absent\n' >&2; exit 66;
}
grep -q 'IF( ln_drgimp .AND. ln_dynspg_ts )' "$compiled_zdf" || {
  printf 'REFUSE: compiled implicit-drag composition is absent\n' >&2; exit 66;
}

mkdir -p "$evidence"
[[ -d "$evidence" && ! -L "$evidence" ]] || {
  printf 'REFUSE: evidence root is not a real directory\n' >&2; exit 65;
}
export PYTHONPATH=$repo:$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

bash -n "$0"
"$py" -m py_compile "$deck_gate" "$record_gate"
"$py" "$deck_gate" --source "$source_deck" --render-deck "$canonical"
"$py" "$deck_gate" --source "$source_deck" --candidate "$canonical" \
  --output "$evidence/omt2_deck_preflight.json"
"$py" "$record_gate" --preflight-only \
  --output "$evidence/omt2_frame_preflight.json"
for plant in extra-delta wrong-selector; do
  if "$py" "$deck_gate" --source "$source_deck" --candidate "$canonical" \
    --plant "$plant" >"$evidence/deck_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: deck %s plant stayed green\n' "$plant" >&2; exit 72
  fi
  grep -q 'STATUS PLANT-FIRED' "$evidence/deck_${plant}_plant.log"
done

if [[ "$mode" == --preflight-only ]]; then
  printf 'ORCA2_ROUND218_OMT2_PREFLIGHT_READY %s\n' "$twin_a"
  exit 0
fi

admit() {
  local plant
  for plant in cadence header field-name truncation nonfinite missing-frame \
    twin-ulp terminal-byte changed-binary early-month; do
    if "$py" "$record_gate" --candidate "$canonical" \
      --calibration "$calibration" --twin-a "$twin_a" --twin-b "$twin_b" \
      --month "$month" --plant "$plant" \
      >"$evidence/record_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: record %s plant stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$evidence/record_${plant}_plant.log"
  done
  "$py" "$record_gate" --candidate "$canonical" \
    --calibration "$calibration" --twin-a "$twin_a" --twin-b "$twin_b" \
    --month "$month" --output "$evidence/omt2_frame_record_admission.json"
  grep -q 'PASS_R218_OMT2_ENTRY_STAGE_AND_MONTH_RECORD' \
    "$evidence/omt2_frame_record_admission.json" || {
    printf 'REFUSE: OMT-2 record is incomplete\n' >&2; exit 73;
  }
  (cd "$evidence" && sha256sum omt2_*json *_plant.log \
    "$calibration"/ORCA2_00000010_restart_*.nc \
    "$twin_a"/ORCA2_00000010_restart_*.nc \
    "$twin_b"/ORCA2_00000010_restart_*.nc \
    "$twin_a"/oracle_r84_frame_*.bin "$twin_b"/oracle_r84_frame_*.bin \
    >ROUND218_SHA256SUMS)
  printf 'ORCA2_ROUND218_OMT2_RECORD_PASS %s\n' "$twin_a"
}

if [[ "$mode" == --admit-existing ]]; then
  [[ -d "$calibration" && -d "$twin_a" && -d "$twin_b" && -d "$month" ]] || {
    printf 'REFUSE: one or more round-218 targets are absent\n' >&2; exit 68;
  }
  admit
  exit 0
fi

for target in "$smoke" "$calibration" "$twin_a" "$twin_b" "$month"; do
  [[ ! -e "$target" ]] || {
    printf 'REFUSE: target exists; use --admit-existing: %s\n' "$target" >&2; exit 68;
  }
done
free_kb=$(df -Pk "$evidence" | awk 'NR==2 {print $4}')
[[ "$free_kb" -ge 6291456 ]] || {
  printf 'REFUSE: evidence volume has under 6 GiB free\n' >&2; exit 67;
}

stage_run() {
  target=$1; binary=$2; itend=$3; stock=$4; steps=$5
  mkdir "$target"
  while read -r _ name; do cp -a "$base/$name" "$target/$name"; done <"$base/deck_files.sha256"
  while read -r _ name; do cp -a "$base/$name" "$target/$name"; done <"$base/input_files.sha256"
  cp "$binary" "$target/nemo"
  "$py" "$deck_gate" --candidate "$canonical" \
    --render-run-deck "$target/namelist_cfg" --itend "$itend" \
    --stock "$stock" --restart-steps "$steps"
  while read -r _ name; do (cd "$target" && sha256sum "$name"); done \
    <"$base/deck_files.sha256" >"$target/deck_files.sha256"
  while read -r _ name; do (cd "$target" && sha256sum "$name"); done \
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
      printf 'REFUSE: NEMO clean pipeline failed (%s,%s)\n' \
        "${pipe_rc[0]}" "${pipe_rc[1]:-0}" >&2; exit 70;
    }
    grep -q 'STOP 0' run.user.stdout.log
    printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' \
      "$((SECONDS-started))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >>run.user.time.log
  )
}

run_month() {
  (
    cd "$month"
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
    started=$SECONDS
    set +e
    mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
    pipe_rc=("${PIPESTATUS[@]}")
    set -e
    if [[ "${pipe_rc[0]}" -eq 0 && "${pipe_rc[1]:-0}" -eq 0 ]]; then
      grep -q 'STOP 0' run.user.stdout.log
      printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' \
        "$((SECONDS-started))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >>run.user.time.log
      exit 0
    fi
    [[ "${pipe_rc[0]}" -eq 123 && "${pipe_rc[1]:-0}" -eq 0 ]] || {
      printf 'REFUSE: unexpected OMT-2 month exit (%s,%s)\n' \
        "${pipe_rc[0]}" "${pipe_rc[1]:-0}" >&2; exit 70;
    }
    grep -q 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' ocean.output
    last_step=$(awk '/^[[:space:]]*kt[[:space:]]+[0-9]+/{step=$2} END{print step+0}' ocean.output)
    [[ "$last_step" -gt 10 ]] || {
      printf 'REFUSE: OMT-2 did not complete its ten-step ladder (kt=%s)\n' "$last_step" >&2; exit 70;
    }
    printf 'wall_seconds %s\nRUN_EXPECTED_STP_CTL kt=%s\n' \
      "$((SECONDS-started))" "$last_step" >>run.user.time.log
  )
}

# Mandatory smoke first: identical OMT-2 physics, legal two-step run control.
stage_run "$smoke" "$base/nemo" 2 2 2
run_clean "$smoke"
stage_run "$calibration" "$base/nemo" 10 10 10
run_clean "$calibration"
stage_run "$twin_a" "$instrument_binary" 10 10 10
run_clean "$twin_a"
stage_run "$twin_b" "$instrument_binary" 10 10 10
run_clean "$twin_b"
stage_run "$month" "$base/nemo" 96 96 10,20,30,40,50,60,70,80,90,95
run_month
admit
