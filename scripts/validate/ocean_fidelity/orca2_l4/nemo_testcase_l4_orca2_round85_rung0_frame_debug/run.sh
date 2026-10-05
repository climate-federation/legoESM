#!/usr/bin/env bash
# Operator-executed debug reproduction of the round-84 rung-0 frame crash.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-85 rung-0 frame debug failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only]\n' "$0" >&2; exit 63 ;;
esac

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
export PATH=$NEMO_ROOT/ext/FCM/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4
readonly TEMPLATE_CFG=ORCA2_OMIP_L4_R84FRAMES
readonly TARGET_CFG=ORCA2_OMIP_L4_R85FRAMEDEBUG
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TEMPLATE_ROOT=$NEMO_ROOT/cfgs/$TEMPLATE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2
readonly INHERITED_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round84/acquisition/orca2_rung0_entry_stage_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round85/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_frame_debug_np2
readonly SOURCE_STP_SHA=9d0318fda246ef1ed3df50172b9d72a5e0b5df879b9a661b38622c9078078989
readonly TEMPLATE_BLD_SHA=7e369dbc74ce270921be47b3051276c213e6a77da2839dc82d0aa359a24bf8a7
readonly SOURCE_BINARY_SHA=c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343
readonly SOURCE_NAMELIST_SHA=d25c69958aeb7d4dffeeab6b08c89f6b314dd7c6d130ed94acfee7cb90643c2c
readonly MODULE_SHA=231b17f2a4b5ca28110efad413ea17af6ab2c3bb1f4875b5cbb0d6b6faa740f5
readonly PATCH_SHA=45a7e59f6ecfbcc36f39197fad21c9adc4b1f9a68b323ff260716ef1f44dfa77
readonly GATE_SHA=5ce68e40f4508d963ed4bcdf187af31329f86b54ebc61e1c2dce16047f568e94
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly DEBUG_FLAGS='-fdefault-real-8 -O0 -g -fbacktrace -fcheck=bounds -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize'

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly R84_DIR=$here/../nemo_testcase_l4_orca2_round84_rung0_frames
readonly MODULE=$R84_DIR/l4_r84_frames.F90
readonly STP_PATCH=$R84_DIR/stprk3_round84.patch
readonly GATE=$here/../nemo_testcase_l4_orca2_round84_rung0_frame_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round85.md

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: debug acquisition requires a clean committed producer tree\n' >&2; exit 64;
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

for path in "$MODULE" "$STP_PATCH" "$GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: debug artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
for path in "$SOURCE_ROOT/MY_SRC/stprk3.F90" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
  "$TEMPLATE_ROOT/BLD/bld.cfg" "$SOURCE_RUN/nemo" "$SOURCE_RUN/namelist_cfg" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing pinned input %s\n' "$path" >&2; exit 64; }
done
pin "$SOURCE_STP_SHA" "$SOURCE_ROOT/MY_SRC/stprk3.F90" 'compiled-card stprk3 source'
pin "$TEMPLATE_BLD_SHA" "$TEMPLATE_ROOT/BLD/bld.cfg" 'round-84 build template'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'admitted rung-0 binary'
pin "$SOURCE_NAMELIST_SHA" "$SOURCE_RUN/namelist_cfg" 'admitted rung-0 namelist'
pin "$MODULE_SHA" "$MODULE" 'round-84 frame module'
pin "$PATCH_SHA" "$STP_PATCH" 'round-84 stprk3 patch'
pin "$GATE_SHA" "$GATE" 'round-84 frame gate'
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
"$PY" "$GATE" --preflight >/dev/null
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$STP_PATCH")
[[ "$removed" -eq 0 ]] || { printf 'REFUSE: writer patch removes %s source lines\n' "$removed" >&2; exit 66; }
dry=$(mktemp -d /tmp/orca2-r85-debug.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/stprk3.F90" "$dry/stprk3.F90"
patch -s --fuzz=0 "$dry/stprk3.F90" <"$STP_PATCH"
[[ "$(grep -Fc 'CALL r84_dump_frame' "$dry/stprk3.F90")" -eq 4 ]] || {
  printf 'REFUSE: patched source does not have four frame calls\n' >&2; exit 66;
}
syntax=$(mktemp -d /tmp/orca2-r85-debug-syntax.XXXXXX)
cpp -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$MODULE" \
  -o "$syntax/l4_r84_frames.f90"
"$FC" -fsyntax-only $DEBUG_FLAGS -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/l4_r84_frames.f90"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND85_RUNG0_FRAME_DEBUG_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: debug target config or run directory already exists\n' >&2; exit 68;
}
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67;
  }
done

manifest=$(mktemp -d /tmp/orca2-r85-debug-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
printf '%s\n' "$DEBUG_FLAGS" >"$manifest/debug_flags.txt"
sha256sum "$MODULE" "$STP_PATCH" "$GATE" "$PREREG" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/stprk3.f90" "$TEMPLATE_ROOT/BLD/bld.cfg" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath -j 0 del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$MODULE" "$TARGET_ROOT/MY_SRC/l4_r84_frames.F90"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/stprk3.F90" <"$STP_PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath -j 0 del_key 'key_xios'

cp "$TEMPLATE_ROOT/BLD/bld.cfg" "$TARGET_ROOT/BLD/bld.cfg"
sed -i "s|$TEMPLATE_ROOT|$TARGET_ROOT|g" "$TARGET_ROOT/BLD/bld.cfg"
sed -i -E "s|^%FCFLAGS[[:space:]].*|%FCFLAGS             $DEBUG_FLAGS|" \
  "$TARGET_ROOT/BLD/arch_nemo.fcm"
[[ "$(grep -Fc '%FCFLAGS' "$TARGET_ROOT/BLD/arch_nemo.fcm")" -eq 1 ]] || {
  printf 'REFUSE: debug build does not have one FCFLAGS assignment\n' >&2; exit 69;
}
grep -F -- '-g -fbacktrace -fcheck=bounds' "$TARGET_ROOT/BLD/arch_nemo.fcm" >/dev/null || {
  printf 'REFUSE: debug flags are absent from generated architecture\n' >&2; exit 69;
}
fcm build --ignore-lock -v 1 -j 1 "$TARGET_ROOT/BLD/bld.cfg"
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: debug build produced no executable\n' >&2; exit 69; }
grep -q 'NEMO_L4_R84FRM1' "$TARGET_ROOT/BLD/ppsrc/nemo/l4_r84_frames.f90" || {
  printf 'REFUSE: compiled debug writer is absent\n' >&2; exit 69;
}
[[ "$(grep -Fc 'CALL r84_dump_frame' "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3.f90")" -eq 4 ]] || {
  printf 'REFUSE: compiled debug source does not have four frame calls\n' >&2; exit 69;
}
grep -F -- '-O0 -g -fbacktrace -fcheck=bounds' "$TARGET_ROOT/BLD/cfg/parsed_bld.cfg" >/dev/null || {
  printf 'REFUSE: resolved debug build flags are absent\n' >&2; exit 69;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in debug binary\n' >&2; exit 69
fi

mkdir "$TARGET_RUN"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$SOURCE_RUN/namelist_cfg" "$TARGET_RUN/namelist_cfg"
cp "$BINARY" "$TARGET_RUN/nemo.debug"
cp "$manifest"/* "$TARGET_RUN/"
(cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)

(
  cd "$TARGET_RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  mpirun -np 2 --oversubscribe ./nemo.debug >run.user.stdout.log 2>&1
  run_rc=$?
  set -e
  printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_EXIT=%s\n' "$((SECONDS-started))" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_rc" >>run.user.time.log
  printf '%s\n' "$run_rc" >run.exit_code.txt
)
run_rc=$(cat "$TARGET_RUN/run.exit_code.txt")
[[ "$run_rc" -ne 0 ]] || {
  printf 'REFUSE: debug run exited cleanly; crash prediction was refuted\n' >&2; exit 70;
}
grep -Eq '(stprk3|l4_r84_frames)\.f90:[0-9]+' "$TARGET_RUN/run.user.stdout.log" || {
  printf 'REFUSE: debug failure has no source-resolved frame line\n' >&2; exit 70;
}
grep -Eq 'l4_canon_2d|l4_dump_ocean_surface_input|r84_dump_frame' "$TARGET_RUN/run.user.stdout.log" || {
  printf 'REFUSE: debug failure does not name an instrument routine\n' >&2; exit 70;
}
cp "$TARGET_ROOT/BLD/cfg/parsed_bld.cfg" "$TARGET_RUN/parsed_bld.cfg"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3.f90" "$TARGET_RUN/stprk3.debug.f90"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/l4_r84_frames.f90" "$TARGET_RUN/l4_r84_frames.debug.f90"
(cd "$TARGET_RUN" && sha256sum nemo.debug parsed_bld.cfg stprk3.debug.f90 \
  l4_r84_frames.debug.f90 run.user.stdout.log run.user.time.log \
  run.exit_code.txt producer_commit.txt debug_flags.txt toolchain.sha256 \
  >round85_debug_outputs.sha256)
printf 'ORCA2_ROUND85_RUNG0_FRAME_DEBUG_REPRODUCED %s exit=%s\n' "$TARGET_RUN" "$run_rc"
