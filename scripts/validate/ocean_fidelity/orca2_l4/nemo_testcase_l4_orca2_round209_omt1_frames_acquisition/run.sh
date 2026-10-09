#!/usr/bin/env bash
# Operator-executed Decision-109 OMT-1 acquisition, repaired in round 210.
set -Eeuo pipefail
refuse_unexpected() { status=$?; printf 'REFUSE: round-210 OMT-1 acquisition failed at line %s (exit %s)\n' "${BASH_LINENO[0]:-unknown}" "$status" >&2; exit "$status"; }
trap refuse_unexpected ERR
mode=${1:---run}
case "$mode" in --run|--preflight-only|--admit-existing) ;; *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;; esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(CDPATH= cd -- "$here/../../../../../" && pwd -P)
py=/home/dbalwada/legoESM/.venv/bin/python
nemo_root=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
reference_cfg=ORCA2_OMIP_L4
source_cfg=ORCA2_OMIP_L4_R90FRAMES
target_cfg=ORCA2_OMIP_L4_R210OMT1_P3
target_root=$nemo_root/cfgs/$target_cfg
source_root=$nemo_root/cfgs/$source_cfg
reference_root=$nemo_root/cfgs/$reference_cfg
source_deck=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round203/acquisition/omt0_namelist_cfg
base=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round203/acquisition/orca2_omt0_uninstrumented_10step_np2
evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round210/acquisition
canonical=$evidence/omt1_namelist_cfg
smoke=$evidence/orca2_omt1_smoke_2step_np2
calibration=$evidence/orca2_omt1_uninstrumented_10step_np2
twin_a=$evidence/orca2_omt1_frames_10step_a_np2
twin_b=$evidence/orca2_omt1_frames_10step_b_np2
month=$evidence/orca2_omt1_month_boundary_96step_np2
deck_gate=$here/../nemo_testcase_l4_orca2_round209_omt1_deck_gate.py
record_gate=$here/../nemo_testcase_l4_orca2_round209_omt1_frame_record_gate.py
prereg=$repo/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round210.md
source_manifest=$here/source_files.sha256

cd "$repo"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63; }
for path in "$0" "$deck_gate" "$record_gate" "$prereg" "$source_manifest"; do
  git ls-files --error-unmatch "${path#"$repo"/}" >/dev/null || { printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 65; }
  git diff --quiet HEAD -- "$path" || { printf 'REFUSE: acquisition artifact differs from HEAD: %s\n' "$path" >&2; exit 65; }
done
[[ "$(sha256sum "$source_deck" | awk '{print $1}')" == 9279c638a9fe50b551fcca8dc62dc1c19c0fc2c55bdcae7f3501a8afeb5ca8ac ]] || { printf 'REFUSE: admitted OMT-0 source deck changed\n' >&2; exit 66; }
[[ "$(sha256sum "$base/nemo" | awk '{print $1}')" == c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343 ]] || { printf 'REFUSE: admitted uninstrumented binary changed\n' >&2; exit 66; }
(cd "$base" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
[[ "$(sha256sum "$source_root/cpp_$source_cfg.fcm" | awk '{print $1}')" == \
  2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67 ]] || {
  printf 'REFUSE: admitted record CPP card changed\n' >&2; exit 66;
}
(cd "$source_root/MY_SRC" && sha256sum -c "$source_manifest" >/dev/null)
validate_bootstrap() {
  local candidate=$1
  grep -q "^${candidate} " "$nemo_root/cfgs/ref_cfgs.txt" || {
    printf 'REFUSE: makenemo reference is not registered in ref_cfgs.txt: %s\n' "$candidate" >&2
    return 1
  }
}
validate_source_inventory() {
  local expected=$1 actual
  actual=$(find "$source_root/MY_SRC" -maxdepth 1 -type f -printf '%f\n' | sort)
  [[ "$actual" == "$expected" ]] || {
    printf 'REFUSE: admitted record source inventory changed\n' >&2
    return 1
  }
}
expected_sources=$(awk '{print $2}' "$source_manifest" | sort)
validate_bootstrap "$reference_cfg"
grep -q "^${source_cfg} " "$nemo_root/cfgs/work_cfgs.txt" || {
  printf 'REFUSE: admitted record source is not registered in work_cfgs.txt\n' >&2; exit 66;
}
validate_source_inventory "$expected_sources"
mkdir -p "$evidence"
[[ -d "$evidence" && ! -L "$evidence" ]] || { printf 'REFUSE: evidence root is not a real directory\n' >&2; exit 65; }
export PYTHONPATH=$repo:$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

bash -n "$0"
"$py" -m py_compile "$deck_gate" "$record_gate"
"$py" "$deck_gate" --source "$source_deck" --render-deck "$canonical"
"$py" "$deck_gate" --source "$source_deck" --candidate "$canonical" --output "$evidence/omt1_deck_preflight.json"
"$py" "$record_gate" --preflight-only --output "$evidence/omt1_frame_preflight.json"
for plant in extra-delta wrong-selector; do
  if "$py" "$deck_gate" --source "$source_deck" --candidate "$canonical" --plant "$plant" >"$evidence/deck_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: deck %s plant stayed green\n' "$plant" >&2; exit 72
  fi
  grep -q 'STATUS PLANT-FIRED' "$evidence/deck_${plant}_plant.log"
done
if validate_bootstrap "$source_cfg" >"$evidence/bootstrap_work_cfg_plant.log" 2>&1; then
  printf 'REFUSE: work-configuration bootstrap plant stayed green\n' >&2; exit 72
fi
grep -q 'REFUSE: makenemo reference is not registered' "$evidence/bootstrap_work_cfg_plant.log"
printf 'STATUS PLANT-FIRED: work configuration cannot be a makenemo reference\n' >>"$evidence/bootstrap_work_cfg_plant.log"
if validate_source_inventory "${expected_sources}"$'\n__missing_source_plant__.F90' >"$evidence/source_inventory_plant.log" 2>&1; then
  printf 'REFUSE: source-inventory plant stayed green\n' >&2; exit 72
fi
grep -q 'REFUSE: admitted record source inventory changed' "$evidence/source_inventory_plant.log"
printf 'STATUS PLANT-FIRED: source inventory is closed\n' >>"$evidence/source_inventory_plant.log"
if [[ "$mode" == --preflight-only ]]; then printf 'ORCA2_ROUND210_OMT1_PREFLIGHT_READY %s\n' "$twin_a"; exit 0; fi

admit() {
  local plant
  for plant in cadence header field-name truncation nonfinite missing-frame twin-ulp terminal-byte changed-binary early-month; do
    if "$py" "$record_gate" --candidate "$canonical" --calibration "$calibration" --twin-a "$twin_a" --twin-b "$twin_b" --month "$month" --plant "$plant" >"$evidence/record_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: record %s plant stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$evidence/record_${plant}_plant.log"
  done
  "$py" "$record_gate" --candidate "$canonical" --calibration "$calibration" --twin-a "$twin_a" --twin-b "$twin_b" --month "$month" --output "$evidence/omt1_frame_record_admission.json"
  grep -q 'PASS_R209_OMT1_ENTRY_STAGE_AND_MONTH_RECORD' "$evidence/omt1_frame_record_admission.json" || { printf 'REFUSE: OMT-1 record is incomplete\n' >&2; exit 73; }
  (cd "$evidence" && sha256sum omt1_*json *_plant.log "$calibration"/ORCA2_00000010_restart_*.nc "$twin_a"/ORCA2_00000010_restart_*.nc "$twin_b"/ORCA2_00000010_restart_*.nc "$twin_a"/oracle_r84_frame_*.bin "$twin_b"/oracle_r84_frame_*.bin >ROUND209_SHA256SUMS)
  printf 'ORCA2_ROUND210_OMT1_RECORD_PASS %s\n' "$twin_a"
}
if [[ "$mode" == --admit-existing ]]; then
  [[ -d "$calibration" && -d "$twin_a" && -d "$twin_b" && -d "$month" ]] || { printf 'REFUSE: one or more round-209 targets are absent\n' >&2; exit 68; }
  admit; exit 0
fi
for target in "$target_root" "$smoke" "$calibration" "$twin_a" "$twin_b" "$month"; do [[ ! -e "$target" ]] || { printf 'REFUSE: target exists; use --admit-existing: %s\n' "$target" >&2; exit 68; }; done
free_kb=$(df -Pk "$evidence" | awk 'NR==2 {print $4}')
[[ "$free_kb" -ge 6291456 ]] || { printf 'REFUSE: evidence volume has under 6 GiB free\n' >&2; exit 67; }

stage_run() {
  target=$1; binary=$2; itend=$3; stock=$4; steps=$5
  mkdir "$target"
  while read -r _ name; do cp -a "$base/$name" "$target/$name"; done <"$base/deck_files.sha256"
  while read -r _ name; do cp -a "$base/$name" "$target/$name"; done <"$base/input_files.sha256"
  cp "$binary" "$target/nemo"
  "$py" "$deck_gate" --candidate "$canonical" --render-run-deck "$target/namelist_cfg" --itend "$itend" --stock "$stock" --restart-steps "$steps"
  while read -r _ name; do (cd "$target" && sha256sum "$name"); done <"$base/deck_files.sha256" >"$target/deck_files.sha256"
  while read -r _ name; do (cd "$target" && sha256sum "$name"); done <"$base/input_files.sha256" >"$target/input_files.sha256"
  (cd "$target" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
}
run_clean() {
  target=$1
  ( cd "$target"; printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log; started=$SECONDS; set +e; mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log; pipe_rc=("${PIPESTATUS[@]}"); set -e; [[ "${pipe_rc[0]}" -eq 0 && "${pipe_rc[1]:-0}" -eq 0 ]] || { printf 'REFUSE: NEMO clean pipeline failed (%s,%s)\n' "${pipe_rc[0]}" "${pipe_rc[1]:-0}" >&2; exit 70; }; grep -q 'STOP 0' run.user.stdout.log; printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log )
}
run_month() {
  ( cd "$month"; printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log; set +e; mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log; pipe_rc=("${PIPESTATUS[@]}"); set -e
    if [[ "${pipe_rc[0]}" -eq 0 && "${pipe_rc[1]:-0}" -eq 0 ]]; then grep -q 'STOP 0' run.user.stdout.log; printf 'RUN_DONE\n' >>run.user.time.log; exit 0; fi
    [[ "${pipe_rc[0]}" -eq 123 && "${pipe_rc[1]:-0}" -eq 0 ]] || { printf 'REFUSE: unexpected OMT-1 month exit (%s,%s)\n' "${pipe_rc[0]}" "${pipe_rc[1]:-0}" >&2; exit 70; }
    grep -q 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' ocean.output
    last_step=$(awk '/^[[:space:]]*kt[[:space:]]+[0-9]+/{step=$2} END{print step+0}' ocean.output)
    [[ "$last_step" -gt 11 ]] || { printf 'REFUSE: OMT-1 did not outlive OMT-0 (kt=%s)\n' "$last_step" >&2; exit 70; }
    printf 'RUN_EXPECTED_STP_CTL kt=%s\n' "$last_step" >>run.user.time.log )
}

# Mandatory smoke: identical OMT-1 physics and run-control schema, shorter end.
stage_run "$smoke" "$base/nemo" 2 2 2
run_clean "$smoke"
cd "$nemo_root"
./makenemo -r "$reference_cfg" -n "$target_cfg" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$target_root/EXP00/$(basename "$source")"
done < <(find "$source_root/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$target_root/MY_SRC/$(basename "$source")"
done < <(find "$source_root/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$source_root/cpp_$source_cfg.fcm" "$target_root/cpp_$target_cfg.fcm"
(cd "$target_root/MY_SRC" && sha256sum -c "$source_manifest" >/dev/null)
touch "$target_root/MY_SRC/"*.F90
./makenemo -n "$target_cfg" -m conda-scalarmath del_key 'key_xios'
instrument_binary=$target_root/BLD/bin/nemo.exe
[[ -x "$instrument_binary" ]] || { printf 'REFUSE: record build produced no executable\n' >&2; exit 69; }
grep -q 'CALL r84_dump_frame' "$target_root/BLD/ppsrc/nemo/stprk3.f90" || { printf 'REFUSE: compiled P3 frame call is absent\n' >&2; exit 69; }
cd "$repo"
stage_run "$calibration" "$base/nemo" 10 10 10; run_clean "$calibration"
stage_run "$twin_a" "$instrument_binary" 10 10 10; run_clean "$twin_a"
stage_run "$twin_b" "$instrument_binary" 10 10 10; run_clean "$twin_b"
stage_run "$month" "$base/nemo" 96 96 10,20,30,40,50,60,70,80,90,95; run_month
admit
