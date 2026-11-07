#!/usr/bin/env bash
# Operator-executed symbolized reproduction of the exact round-87 failure.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-88 rung-0 frame debug failed at line %s (exit %s)\n' \
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
readonly SOURCE_CFG=ORCA2_OMIP_L4_R87FRAMES
readonly TARGET_CFG=ORCA2_OMIP_L4_R88FRAMEDEBUG
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly FAILED_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round87/acquisition/orca2_rung0_entry_stage_fixed_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round88/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_frame_debug2_np2
readonly SOURCE_TRAADV_SHA=b5859bee9b8d9f918a456552c9759943583307a6eacb769c3180d3bd7a38103c
readonly SOURCE_STP_SHA=31f9d62f7ac06b84bc3ef6b5ec94da19695014663c671c31bfc21496e42d1e93
readonly FAILED_BINARY_SHA=bf254f6dd04463d4095427a40302a48597057f654fc242161e4d49d9c985914b
readonly DEBUG_FLAGS='-fdefault-real-8 -O0 -g -fbacktrace -fcheck=bounds -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize'

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
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
  printf 'REFUSE: debug acquisition requires a clean committed producer tree\n' >&2; exit 64;
}
readonly COMMIT=$(git rev-parse HEAD)
for path in "$0" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: debug artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
for path in "$SOURCE_ROOT/MY_SRC/traadv.F90" "$SOURCE_ROOT/MY_SRC/stprk3.F90" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$FAILED_RUN/nemo" "$FAILED_RUN/namelist_cfg" \
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
  printf 'REFUSE: pinned optimized run no longer carries SIGSEGV\n' >&2; exit 65;
}
[[ "$(addr2line -f -e "$FAILED_RUN/nemo" 0xcdfe5 | head -1)" == '__traadv_MOD_tra_adv_trp_t.constprop.0' ]] || {
  printf 'REFUSE: optimized failure offset no longer resolves to tra_adv_trp_t\n' >&2; exit 65;
}
grep -Eq 'runoff / runoff mouths.*ln_rnf *= *F' "$FAILED_RUN/ocean.output" || {
  printf 'REFUSE: failed run does not resolve runoff off\n' >&2; exit 65;
}

mkdir -p "$EVIDENCE"
bash -n "$0"
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND88_RUNG0_FRAME_DEBUG_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: debug target config or run directory already exists\n' >&2; exit 68;
}
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || { printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67; }
done

manifest=$(mktemp -d /tmp/orca2-r88-debug-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
printf '%s\n' "$DEBUG_FLAGS" >"$manifest/debug_flags.txt"
sha256sum "$PREREG" "$SOURCE_ROOT/MY_SRC/traadv.F90" "$SOURCE_ROOT/MY_SRC/stprk3.F90" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath -j 0 del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath -j 0 del_key 'key_xios'

sed -i -E "s|^%FCFLAGS[[:space:]].*|%FCFLAGS             $DEBUG_FLAGS|" \
  "$TARGET_ROOT/BLD/arch_nemo.fcm"
[[ "$(grep -Ec '^%FCFLAGS[[:space:]]' "$TARGET_ROOT/BLD/arch_nemo.fcm")" -eq 1 ]] || {
  printf 'REFUSE: debug build does not have one FCFLAGS assignment\n' >&2; exit 69;
}
grep -F -- '-O0 -g -fbacktrace -fcheck=bounds' "$TARGET_ROOT/BLD/arch_nemo.fcm" >/dev/null || {
  printf 'REFUSE: debug flags are absent from generated architecture\n' >&2; exit 69;
}
fcm build --ignore-lock -v 1 -j 1 "$TARGET_ROOT/BLD/bld.cfg"
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: debug build produced no executable\n' >&2; exit 69; }
grep -F -- '-O0 -g -fbacktrace -fcheck=bounds' "$TARGET_ROOT/BLD/cfg/parsed_bld.cfg" >/dev/null || {
  printf 'REFUSE: resolved debug flags are absent\n' >&2; exit 69;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in debug binary\n' >&2; exit 69
fi

mkdir "$TARGET_RUN"
while read -r digest name; do cp -a "$FAILED_RUN/$name" "$TARGET_RUN/$name"; done <"$FAILED_RUN/deck_files.sha256"
while read -r digest name; do cp -a "$FAILED_RUN/$name" "$TARGET_RUN/$name"; done <"$FAILED_RUN/input_files.sha256"
cp "$FAILED_RUN/deck_files.sha256" "$FAILED_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$FAILED_RUN/namelist_cfg" "$TARGET_RUN/namelist_cfg"
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
grep -Eq 'traadv\.f90:[0-9]+' "$TARGET_RUN/run.user.stdout.log" || {
  printf 'REFUSE: debug failure has no source-resolved traadv line\n' >&2; exit 70;
}
grep -Eq 'tra_adv_trp_t|tra_adv_trp' "$TARGET_RUN/run.user.stdout.log" || {
  printf 'REFUSE: debug failure does not name tracer transport\n' >&2; exit 70;
}
cp "$TARGET_ROOT/BLD/cfg/parsed_bld.cfg" "$TARGET_RUN/parsed_bld.cfg"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/traadv.f90" "$TARGET_RUN/traadv.debug.f90"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3.f90" "$TARGET_RUN/stprk3.debug.f90"
(cd "$TARGET_RUN" && sha256sum nemo.debug parsed_bld.cfg traadv.debug.f90 stprk3.debug.f90 \
  run.user.stdout.log run.user.time.log run.exit_code.txt producer_commit.txt debug_flags.txt \
  toolchain.sha256 >round88_debug_outputs.sha256)
printf 'ORCA2_ROUND88_RUNG0_FRAME_DEBUG_REPRODUCED %s exit=%s\n' "$TARGET_RUN" "$run_rc"
