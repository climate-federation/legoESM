#!/usr/bin/env bash
# Operator-executed rung-0 frame acquisition after the OFF-runoff probe repair.
set -Eeuo pipefail

printf 'REFUSE: round-88 runoff attribution retracted; use ../nemo_testcase_l4_orca2_round88_rung0_frame_debug/run.sh\n' >&2
exit 79

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-88 rung-0 frame acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 63 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4_R87FRAMES
readonly TARGET_CFG=ORCA2_OMIP_L4_R88FRAMES
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly FAILED_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round87/acquisition/orca2_rung0_entry_stage_fixed_10step_np2
readonly CALIBRATION_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round88/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_entry_stage_fixed2_10step_np2
readonly SOURCE_TRAADV_SHA=b5859bee9b8d9f918a456552c9759943583307a6eacb769c3180d3bd7a38103c
readonly SOURCE_STP_SHA=31f9d62f7ac06b84bc3ef6b5ec94da19695014663c671c31bfc21496e42d1e93
readonly FAILED_BINARY_SHA=bf254f6dd04463d4095427a40302a48597057f654fc242161e4d49d9c985914b
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/traadv_round88_skip_off_runoff_probe.patch
readonly FRAME_GATE=$here/../nemo_testcase_l4_orca2_round84_rung0_frame_gate.py
readonly SURFACE_GATE=$here/../nemo_testcase_l4_orca2_round87_surface_absence_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round88.md

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed producer tree\n' >&2; exit 64;
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

for path in "$0" "$PATCH" "$FRAME_GATE" "$SURFACE_GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
for path in "$SOURCE_ROOT/MY_SRC/traadv.F90" "$SOURCE_ROOT/MY_SRC/stprk3.F90" \
  "$FAILED_RUN/nemo" "$FAILED_RUN/namelist_cfg" "$FAILED_RUN/ocean.output" \
  "$FAILED_RUN/deck_files.sha256" "$FAILED_RUN/input_files.sha256"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing pinned input %s\n' "$path" >&2; exit 64; }
done
pin "$SOURCE_TRAADV_SHA" "$SOURCE_ROOT/MY_SRC/traadv.F90" 'failed-build traadv source'
pin "$SOURCE_STP_SHA" "$SOURCE_ROOT/MY_SRC/stprk3.F90" 'failed-build stprk3 source'
pin "$FAILED_BINARY_SHA" "$FAILED_RUN/nemo" 'failed optimized binary'
(cd "$FAILED_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
[[ "$(find "$FAILED_RUN" -maxdepth 1 -type f -name 'oracle_r84_frame_*.bin' | wc -l)" -eq 2 ]] || {
  printf 'REFUSE: failed record no longer has exactly two stage-0 frames\n' >&2; exit 65;
}
grep -q 'signal 11 (Segmentation fault)' "$FAILED_RUN/run.user.stdout.log" || {
  printf 'REFUSE: pinned record does not carry the observed SIGSEGV\n' >&2; exit 65;
}
[[ "$(addr2line -f -e "$FAILED_RUN/nemo" 0xcdfe5 | head -1)" == '__traadv_MOD_tra_adv_trp_t.constprop.0' ]] || {
  printf 'REFUSE: optimized failure offset no longer resolves to tra_adv_trp_t\n' >&2; exit 65;
}
grep -Fq "l4_canon_2d(rnf,'T')" "$SOURCE_ROOT/BLD/ppsrc/nemo/traadv.f90" || {
  printf 'REFUSE: compiled failing runoff dereference is absent\n' >&2; exit 65;
}
grep -Eq 'runoff / runoff mouths.*ln_rnf *= *F' "$FAILED_RUN/ocean.output" || {
  printf 'REFUSE: failed run does not resolve runoff off\n' >&2; exit 65;
}
grep -Fq 'IF(ln_rnf)   ALLOCATE( rnf' "$SOURCE_ROOT/BLD/ppsrc/nemo/sbc_oce.f90" || {
  printf 'REFUSE: compiled allocation guard is absent\n' >&2; exit 65;
}

mkdir -p "$EVIDENCE"
bash -n "$0"
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
[[ "$removed" -eq 0 ]] || { printf 'REFUSE: repair removes %s source lines\n' "$removed" >&2; exit 66; }
[[ "$(grep -Fc '+         IF( ln_rnf ) THEN' "$PATCH")" -eq 3 ]] || {
  printf 'REFUSE: repair must add exactly three runoff guards\n' >&2; exit 66;
}
dry=$(mktemp -d /tmp/orca2-r88-frame.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/traadv.F90" "$dry/traadv.F90"
patch -s --fuzz=0 "$dry/traadv.F90" <"$PATCH"
[[ "$(grep -Fc 'IF( ln_rnf ) THEN' "$dry/traadv.F90")" -eq 3 ]] || {
  printf 'REFUSE: patched source does not carry exactly three runoff guards\n' >&2; exit 66;
}
cpp -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$dry/traadv.F90" -o "$dry/traadv.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" "$dry/traadv.f90"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND88_RUNG0_FRAMES_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant name digest
  [[ "$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r84_frame_*.bin' | wc -l)" -eq 80 ]] || {
    printf 'REFUSE: target does not carry 80 frames\n' >&2; exit 71;
  }
  for name in "$TARGET_RUN"/oracle_r84_frame_*.bin; do
    digest=$(sha256sum "$name" | awk '{print $1}')
    printf '%s %s %s\n' "$digest" "$COMMIT" "$(basename "$name")" >"$name.stamp"
  done
  for plant in header field-name truncation nonfinite stamp; do
    if "$PY" "$FRAME_GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" --plant "$plant" \
      >"$TARGET_RUN/round88_frame_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: frame %s plant stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$TARGET_RUN/round88_frame_${plant}_plant.log"
  done
  "$PY" "$FRAME_GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" \
    --output "$TARGET_RUN/round88_frame_admission.json"
  for plant in header field-name truncation nonfinite absent-as-zero owner-on; do
    if "$PY" "$SURFACE_GATE" --record "$TARGET_RUN/oracle_ocean_surface_input_kt00000001.bin" \
      --namelist "$TARGET_RUN/namelist_cfg" --ocean-output "$TARGET_RUN/ocean.output" --plant "$plant" \
      >"$TARGET_RUN/round88_surface_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: surface %s plant stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$TARGET_RUN/round88_surface_${plant}_plant.log"
  done
  "$PY" "$SURFACE_GATE" --record "$TARGET_RUN/oracle_ocean_surface_input_kt00000001.bin" \
    --namelist "$TARGET_RUN/namelist_cfg" --ocean-output "$TARGET_RUN/ocean.output" \
    --output "$TARGET_RUN/round88_surface_admission.json"
  [[ ! -e "$TARGET_RUN/oracle_stage1_wzv_operands_kt00000001.bin" ]] || {
    printf 'REFUSE: OFF-runoff legacy probe was not suppressed\n' >&2; exit 73;
  }
  [[ "$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'ORCA2_000000??_restart_????.nc' | wc -l)" -eq 20 ]] || {
    printf 'REFUSE: target does not carry 20 terminal restarts\n' >&2; exit 73;
  }
  for name in "$CALIBRATION_RUN"/ORCA2_000000??_restart_????.nc; do
    cmp -s "$name" "$TARGET_RUN/$(basename "$name")" || {
      printf 'REFUSE: write-only repair changed %s\n' "$(basename "$name")" >&2; exit 73;
    }
  done
  (cd "$TARGET_RUN" && sha256sum oracle_r84_frame_*.bin oracle_r84_frame_*.bin.stamp \
    ORCA2_000000??_restart_????.nc round88_*_admission.json round88_*_plant.log >round88_outputs.sha256)
  printf 'ORCA2_ROUND88_RUNG0_FRAMES_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -d "$TARGET_RUN" ]] || { printf 'REFUSE: target run is absent\n' >&2; exit 68; }
  [[ "$(cat "$TARGET_RUN/producer_commit.txt")" == "$COMMIT" ]] || {
    printf 'REFUSE: target was produced by another commit\n' >&2; exit 68;
  }
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 68;
}
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || { printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67; }
done

manifest=$(mktemp -d /tmp/orca2-r88-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$PATCH" "$PREREG" "$SOURCE_ROOT/BLD/ppsrc/nemo/traadv.f90" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/traadv.F90" <"$PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 69; }
[[ "$(grep -Fc 'IF( ln_rnf ) THEN' "$TARGET_ROOT/BLD/ppsrc/nemo/traadv.f90")" -eq 3 ]] || {
  printf 'REFUSE: compiled repair guard census failed\n' >&2; exit 69;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2; exit 69
fi

mkdir "$TARGET_RUN"
while read -r digest name; do cp -a "$FAILED_RUN/$name" "$TARGET_RUN/$name"; done <"$FAILED_RUN/deck_files.sha256"
while read -r digest name; do cp -a "$FAILED_RUN/$name" "$TARGET_RUN/$name"; done <"$FAILED_RUN/input_files.sha256"
cp "$FAILED_RUN/deck_files.sha256" "$FAILED_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$FAILED_RUN/namelist_cfg" "$TARGET_RUN/namelist_cfg"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
(cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)

(
  cd "$TARGET_RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  [[ "${pipe_rc[0]}" -eq 0 ]] || { printf 'REFUSE: mpirun exited %s\n' "${pipe_rc[0]}" >&2; exit 70; }
  [[ "${pipe_rc[1]:-0}" -eq 0 ]] || { printf 'REFUSE: tee exited %s\n' "${pipe_rc[1]}" >&2; exit 70; }
  printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
grep -q 'STOP 0' "$TARGET_RUN/run.user.stdout.log" || {
  printf 'REFUSE: NEMO did not report STOP 0\n' >&2; exit 70;
}
admit
