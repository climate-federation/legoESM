#!/usr/bin/env bash
# ORCA2 round-69 acquisition: exact ocean surface operands through kt=240.
# The operator runs this file; agents never invoke mpirun in the sandbox.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-69 surface acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE
readonly SOURCE_STPRK3=$NEMO_ROOT/src/OCE/stprk3.F90
readonly SOURCE_CPP=$NEMO_ROOT/cfgs/$SOURCE_CFG/cpp_$SOURCE_CFG.fcm
readonly ARCH=$NEMO_ROOT/arch/arch-conda-scalarmath.fcm
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED_STPRK3=$TARGET_ROOT/BLD/ppsrc/nemo/stprk3.f90
readonly BASE_MONTH=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round67/acquisition/orca1ice_uninstrumented_fromrest_30day_np2
readonly CALIBRATION=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round69/acquisition
readonly RUN=$EVIDENCE/orca1ice_surface_only_240step_np2
readonly DECK_SHA=09a350860ff6eaef17d1f0e18aa8e16c4d929e994d9d6804d0976f531b06f66e
readonly INPUT_SHA=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
readonly BASE_STPRK3_SHA=d12b246db6b77b122ef1c53a485a3d742ce4acf113a58684a639c9031f0a9e1a
readonly CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly ARCH_SHA=132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561
readonly CALIBRATION_ADMISSION_SHA=d2e2d51f1962653b1264d7c875645fdb8f378ccfaa584cf8b6e92130d51b7589

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly MODULE=$here/l4_r69_surface.F90
readonly PATCH=$here/stprk3_round69.patch
readonly GATE=$here/../nemo_testcase_l4_orca2_round69_month_surface_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round69_month_surface_inputs.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 65; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 66;
  }
}

for path in "$MODULE" "$PATCH" "$GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 65;
  }
done
pin "$BASE_STPRK3_SHA" "$SOURCE_STPRK3" 'base stprk3'
pin "$CPP_SHA" "$SOURCE_CPP" 'CPP card'
pin "$ARCH_SHA" "$ARCH" 'compiler card'
pin "$DECK_SHA" "$BASE_MONTH/deck_files.sha256" 'month deck manifest'
pin "$INPUT_SHA" "$BASE_MONTH/input_files.sha256" 'input manifest'
pin "$CALIBRATION_ADMISSION_SHA" "$CALIBRATION/round1_surface_admission.json" 'surface calibration admission'
(cd "$BASE_MONTH" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
mkdir -p "$EVIDENCE"
[[ -d "$EVIDENCE" && ! -L "$EVIDENCE" ]] || {
  printf 'REFUSE: evidence path is not a real directory: %s\n' "$EVIDENCE" >&2; exit 65;
}
bash -n "$0"
"$PY" -m py_compile "$GATE"
"$PY" "$GATE" --preflight-only >"$EVIDENCE/preflight_gate.json"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND69_SURFACE_PREFLIGHT_READY %s\n' "$RUN"
  exit 0
fi

admit() {
  local plant
  for plant in field-name truncated calibration-ulp missing-frame extra-stream restart-ulp; do
    if "$PY" "$GATE" --record "$RUN" --calibration-root "$CALIBRATION" \
      --month-root "$BASE_MONTH" --expect-commit "$COMMIT" --plant "$plant" \
      >"$EVIDENCE/${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2
      exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/${plant}_plant.log"
  done
  "$PY" "$GATE" --record "$RUN" --calibration-root "$CALIBRATION" \
    --month-root "$BASE_MONTH" --expect-commit "$COMMIT" \
    --output "$EVIDENCE/month_surface_admission.json"
  (cd "$EVIDENCE" && sha256sum month_surface_admission.json *_plant.log \
    "$RUN"/ORCA2_00000240_restart*.nc >round69_outputs.sha256)
  printf 'ORCA2_ROUND69_SURFACE_ACQUISITION_PASS %s\n' "$RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -d "$RUN" ]] || { printf 'REFUSE: existing target is absent: %s\n' "$RUN" >&2; exit 68; }
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$RUN" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 68;
}
for mount in "$NEMO_ROOT" "$EVIDENCE"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67;
  }
done

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios' || {
  printf 'REFUSE: makenemo could not create isolated target\n' >&2; exit 69;
}
cp "$SOURCE_STPRK3" "$TARGET_ROOT/MY_SRC/stprk3.F90"
cp "$MODULE" "$TARGET_ROOT/MY_SRC/l4_r69_surface.F90"
cp "$SOURCE_CPP" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
(cd "$TARGET_ROOT/MY_SRC" && patch -p0 <"$PATCH") || {
  printf 'REFUSE: committed call-site patch failed\n' >&2; exit 69;
}
touch "$TARGET_ROOT/MY_SRC/stprk3.F90" "$TARGET_ROOT/MY_SRC/l4_r69_surface.F90"
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios' || {
  printf 'REFUSE: makenemo could not build isolated target\n' >&2; exit 69;
}
[[ -x "$BINARY" && -f "$COMPILED_STPRK3" ]] || {
  printf 'REFUSE: target build lacks binary or compiled stprk3\n' >&2; exit 69;
}
readonly COMPILED_MODULE=$(find "$TARGET_ROOT/BLD/ppsrc/nemo" -maxdepth 1 -type f -iname 'l4_r69_surface.f90' -print -quit)
[[ -n "$COMPILED_MODULE" ]] || { printf 'REFUSE: compiled writer module is absent\n' >&2; exit 69; }
grep -Fq 'CALL l4_r69_dump( kstp, Nbb )' "$COMPILED_STPRK3" || {
  printf 'REFUSE: compiled source lacks round-69 call\n' >&2; exit 69;
}
grep -Fq "STATUS='NEW'" "$COMPILED_MODULE" || {
  printf 'REFUSE: compiled writer does not refuse overwrite\n' >&2; exit 69;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol is present in acquisition binary\n' >&2; exit 69;
fi

mkdir "$RUN"
while read -r digest name; do cp -a "$BASE_MONTH/$name" "$RUN/$name"; done \
  <"$BASE_MONTH/deck_files.sha256"
while read -r digest name; do cp -a "$BASE_MONTH/$name" "$RUN/$name"; done \
  <"$BASE_MONTH/input_files.sha256"
cp "$BASE_MONTH/deck_files.sha256" "$BASE_MONTH/input_files.sha256" "$RUN/"
cp "$BINARY" "$RUN/nemo"
cp "$COMPILED_STPRK3" "$RUN/compiled_stprk3.f90"
cp "$COMPILED_MODULE" "$RUN/compiled_l4_r69_surface.f90"
printf '%s\n' "$COMMIT" >"$RUN/producer_commit.txt"
(cd "$RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
sha256sum "$RUN/nemo" >"$RUN/binary.sha256"
sha256sum "$RUN/compiled_stprk3.f90" "$RUN/compiled_l4_r69_surface.f90" \
  >"$RUN/compiled_source.sha256"
sha256sum "$MODULE" "$PATCH" >"$RUN/acquisition_sources.sha256"

(
  cd "$RUN"
  for output in ocean.output time.step run.user.stdout.log run.user.time.log; do
    [[ ! -e "$output" ]] || { printf 'REFUSE: output exists: %s/%s\n' "$RUN" "$output" >&2; exit 70; }
  done
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
grep -q 'STOP 0' "$RUN/run.user.stdout.log" || {
  printf 'REFUSE: NEMO did not report STOP 0\n' >&2; exit 70;
}
admit
