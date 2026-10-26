#!/usr/bin/env bash
# Operator-executed ORCA2 rung-0 entry/stage acquisition (both MPI ranks).
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-84 rung-0 frame acquisition failed at line %s (exit %s)\n' \
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
readonly SOURCE_CFG=ORCA2_OMIP_L4
readonly TARGET_CFG=ORCA2_OMIP_L4_R84FRAMES
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round84/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_entry_stage_10step_np2
readonly SOURCE_STP_SHA=9d0318fda246ef1ed3df50172b9d72a5e0b5df879b9a661b38622c9078078989
readonly SOURCE_BINARY_SHA=c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343
readonly SOURCE_NAMELIST_SHA=d25c69958aeb7d4dffeeab6b08c89f6b314dd7c6d130ed94acfee7cb90643c2c
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly MODULE=$here/l4_r84_frames.F90
readonly STP_PATCH=$here/stprk3_round84.patch
readonly GATE=$here/../nemo_testcase_l4_orca2_round84_rung0_frame_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round84.md

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

for path in "$MODULE" "$STP_PATCH" "$GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
for path in "$SOURCE_ROOT/MY_SRC/stprk3.F90" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
  "$SOURCE_RUN/nemo" "$SOURCE_RUN/namelist_cfg" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing pinned input %s\n' "$path" >&2; exit 64; }
done
pin "$SOURCE_STP_SHA" "$SOURCE_ROOT/MY_SRC/stprk3.F90" 'compiled-card stprk3 source'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'admitted rung-0 binary'
pin "$SOURCE_NAMELIST_SHA" "$SOURCE_RUN/namelist_cfg" 'admitted rung-0 namelist'
(cd "$SOURCE_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
grep -Eq 'number of the last time step.*nn_itend *= *10' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: admitted source run is not ten steps\n' >&2; exit 65;
}
grep -Eq 'constant vertical mixing coefficient.*ln_zdfcst *= *T' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: admitted source run is not rung 0\n' >&2; exit 65;
}
grep -Eq 'ice management in the sbc.*nn_ice *= *0' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: admitted source run enables ice\n' >&2; exit 65;
}

mkdir -p "$EVIDENCE"
[[ -d "$EVIDENCE" && ! -L "$EVIDENCE" ]] || {
  printf 'REFUSE: evidence path is not a real directory: %s\n' "$EVIDENCE" >&2; exit 65;
}
bash -n "$0"
"$PY" -m py_compile "$GATE"
"$PY" "$GATE" --preflight
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$STP_PATCH")
[[ "$removed" -eq 0 ]] || { printf 'REFUSE: writer patch removes %s source lines\n' "$removed" >&2; exit 66; }
dry=$(mktemp -d /tmp/orca2-r84-frame.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/stprk3.F90" "$dry/stprk3.F90"
patch -s --fuzz=0 "$dry/stprk3.F90" <"$STP_PATCH"
[[ "$(grep -Fc 'CALL r84_dump_frame' "$dry/stprk3.F90")" -eq 4 ]] || {
  printf 'REFUSE: patched source does not have four frame calls\n' >&2; exit 66;
}

syntax=$(mktemp -d /tmp/orca2-r84-syntax.XXXXXX)
cpp -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$MODULE" \
  -o "$syntax/l4_r84_frames.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/l4_r84_frames.f90"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND84_RUNG0_FRAMES_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant name digest count
  count=$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r84_frame_*.bin' | wc -l)
  [[ "$count" -eq 80 ]] || { printf 'REFUSE: frame count is %s, expected 80\n' "$count" >&2; exit 71; }
  for name in "$TARGET_RUN"/oracle_r84_frame_*.bin; do
    digest=$(sha256sum "$name" | awk '{print $1}')
    printf '%s %s %s\n' "$digest" "$COMMIT" "$(basename "$name")" >"$name.stamp"
  done
  for plant in header field-name truncation nonfinite stamp; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" \
      --plant "$plant" >"$TARGET_RUN/round84_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: frame %s plant stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$TARGET_RUN/round84_${plant}_plant.log"
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" \
    --output "$TARGET_RUN/round84_frame_admission.json"
  for name in "$SOURCE_RUN"/ORCA2_000000??_restart_????.nc; do
    cmp -s "$name" "$TARGET_RUN/$(basename "$name")" || {
      printf 'REFUSE: write-only calibration changed %s\n' "$(basename "$name")" >&2; exit 73;
    }
  done
  (cd "$TARGET_RUN" && sha256sum oracle_r84_frame_*.bin oracle_r84_frame_*.bin.stamp \
    ORCA2_000000??_restart_????.nc round84_frame_admission.json \
    round84_*_plant.log >round84_outputs.sha256)
  printf 'ORCA2_ROUND84_RUNG0_FRAMES_ACQUISITION_PASS %s\n' "$TARGET_RUN"
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
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67;
  }
done

manifest=$(mktemp -d /tmp/orca2-r84-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$MODULE" "$STP_PATCH" "$GATE" "$PREREG" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/stprk3.f90" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$MODULE" "$TARGET_ROOT/MY_SRC/l4_r84_frames.F90"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/stprk3.F90" <"$STP_PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 69; }
grep -q 'NEMO_L4_R84FRM1' "$TARGET_ROOT/BLD/ppsrc/nemo/l4_r84_frames.f90" || {
  printf 'REFUSE: compiled writer is absent\n' >&2; exit 69;
}
[[ "$(grep -Fc 'CALL r84_dump_frame' "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3.f90")" -eq 4 ]] || {
  printf 'REFUSE: compiled source does not have four frame calls\n' >&2; exit 69;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2; exit 69
fi

mkdir "$TARGET_RUN"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$SOURCE_RUN/namelist_cfg" "$TARGET_RUN/namelist_cfg"
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
