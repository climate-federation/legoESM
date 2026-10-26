#!/usr/bin/env bash
# Resume the exact round-85 debug target after its false FCFLAGS refusal.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-86 rung-0 frame debug resume failed at line %s (exit %s)\n' \
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
readonly SOURCE_CFG=ORCA2_OMIP_L4
readonly TARGET_CFG=ORCA2_OMIP_L4_R85FRAMEDEBUG
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round86/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_frame_debug_resume_np2
readonly SOURCE_STP_SHA=9d0318fda246ef1ed3df50172b9d72a5e0b5df879b9a661b38622c9078078989
readonly TARGET_STP_SHA=642dc789ec709d78aaca2e821957d7c1c7468381729f136852540a4bf1f03bd1
readonly MODULE_SHA=231b17f2a4b5ca28110efad413ea17af6ab2c3bb1f4875b5cbb0d6b6faa740f5
readonly TARGET_ARCH_SHA=a46915a5a48c7cdcaabd4285b674824a0cadd7a09d4fc0d6a2308d9f603e919c
readonly TARGET_BLD_SHA=4bd5f451c1be908d805e4fd55e76b1f5bf372c6cc023ee2e53e087d5077df852
readonly TARGET_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly ARCH_HISTORY_SHA=e6dd1bdc556d5ee92425e7a4131250574bd9881c432e62e2f8c109effe6492a0
readonly CPP_HISTORY_SHA=a7b620a94152d62286789d347bad69f641f1b77c1125f04e0b045c52b00e553c
readonly KEY_LIST_SHA=10dad4030c8b70ab922506ef4007b307dad4c7c7ddff41a093f0504a7bfb3115
readonly SOURCE_BINARY_SHA=c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343
readonly SOURCE_NAMELIST_SHA=d25c69958aeb7d4dffeeab6b08c89f6b314dd7c6d130ed94acfee7cb90643c2c
readonly DEBUG_FLAGS='-fdefault-real-8 -O0 -g -fbacktrace -fcheck=bounds -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize'

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly SCRIPT=$here/run.sh
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly R84_DIR=$here/../nemo_testcase_l4_orca2_round84_rung0_frames
readonly MODULE=$R84_DIR/l4_r84_frames.F90
readonly STP_PATCH=$R84_DIR/stprk3_round84.patch
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round86.md
readonly TARGET_ARCH=$TARGET_ROOT/BLD/arch_nemo.fcm
readonly TARGET_BLD=$TARGET_ROOT/BLD/bld.cfg

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2
    exit 65
  }
}

count_fcflags_assignments() {
  awk '$1 == "%FCFLAGS" { count++ } END { print count + 0 }' "$1"
}

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: debug resume requires a clean committed producer tree\n' >&2
  exit 64
}
readonly COMMIT=$(git rev-parse HEAD)
for path in "$MODULE" "$STP_PATCH" "$PREREG" "$SCRIPT"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: debug artifact is not committed: %s\n' "$path" >&2
    exit 64
  }
done

[[ -d "$TARGET_ROOT" && ! -L "$TARGET_ROOT" ]] || {
  printf 'REFUSE: inherited debug target is absent or is a symlink\n' >&2
  exit 64
}
[[ ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: round-86 debug run directory already exists\n' >&2
  exit 64
}
[[ ! -e "$TARGET_ROOT/BLD/bin/nemo.exe" && ! -e "$TARGET_ROOT/BLD/cfg/parsed_bld.cfg" ]] || {
  printf 'REFUSE: inherited target advanced past the recorded pre-build refusal\n' >&2
  exit 65
}

pin "$SOURCE_STP_SHA" "$SOURCE_ROOT/MY_SRC/stprk3.F90" 'pinned source stprk3'
pin "$SOURCE_STP_SHA" "$TARGET_ROOT/MY_SRC/stprk3.F90.orig" 'inherited original stprk3'
pin "$TARGET_STP_SHA" "$TARGET_ROOT/MY_SRC/stprk3.F90" 'inherited patched stprk3'
pin "$MODULE_SHA" "$MODULE" 'committed round-84 frame module'
pin "$MODULE_SHA" "$TARGET_ROOT/MY_SRC/l4_r84_frames.F90" 'inherited frame module'
pin "$TARGET_ARCH_SHA" "$TARGET_ARCH" 'inherited debug architecture'
pin "$TARGET_BLD_SHA" "$TARGET_BLD" 'inherited build configuration'
pin "$TARGET_CPP_SHA" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm" 'inherited CPP keys'
pin "$ARCH_HISTORY_SHA" "$TARGET_ROOT/BLD/arch.history" 'inherited architecture history'
pin "$CPP_HISTORY_SHA" "$TARGET_ROOT/BLD/cpp.history" 'inherited CPP history'
pin "$KEY_LIST_SHA" "$TARGET_ROOT/BLD/full_key_list.txt" 'inherited full key list'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'admitted rung-0 binary'
pin "$SOURCE_NAMELIST_SHA" "$SOURCE_RUN/namelist_cfg" 'admitted rung-0 namelist'

reconstructed=$(mktemp -d /tmp/orca2-r86-reconstruct.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/stprk3.F90" "$reconstructed/stprk3.F90"
patch -s --fuzz=0 "$reconstructed/stprk3.F90" <"$STP_PATCH"
cmp -s "$reconstructed/stprk3.F90" "$TARGET_ROOT/MY_SRC/stprk3.F90" || {
  printf 'REFUSE: inherited stprk3 is not the exact round-84 additions-only patch\n' >&2
  exit 65
}

expected=$(mktemp /tmp/orca2-r86-expected.XXXXXX)
actual=$(mktemp /tmp/orca2-r86-actual.XXXXXX)
find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -printf '%f\n' \
  | sort -u >"$expected"
printf '%s\n' l4_r84_frames.F90 stprk3.F90.orig >>"$expected"
sort -u -o "$expected" "$expected"
find "$TARGET_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -printf '%f\n' \
  | sort -u >"$actual"
if ! diff -u "$expected" "$actual" >/dev/null; then
  printf 'REFUSE: inherited MY_SRC inventory differs from the exact resumed target\n' >&2
  diff -u "$expected" "$actual" >&2 || true
  exit 65
fi
while IFS= read -r name; do
  case "$name" in
    stprk3.F90) continue ;;
  esac
  cmp -s "$SOURCE_ROOT/MY_SRC/$name" "$TARGET_ROOT/MY_SRC/$name" || {
    printf 'REFUSE: inherited MY_SRC file changed: %s\n' "$name" >&2
    exit 65
  }
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -printf '%f\n' | sort)

assignment_count=$(count_fcflags_assignments "$TARGET_ARCH")
literal_count=$(grep -Fc '%FCFLAGS' "$TARGET_ARCH")
[[ "$assignment_count" -eq 1 && "$literal_count" -eq 2 ]] || {
  printf 'REFUSE: inherited FCFLAGS census is assignments=%s literals=%s, expected 1/2\n' \
    "$assignment_count" "$literal_count" >&2
  exit 66
}
grep -F -- '-O0 -g -fbacktrace -fcheck=bounds' "$TARGET_ARCH" >/dev/null || {
  printf 'REFUSE: debug flags are absent from inherited architecture\n' >&2
  exit 66
}

plant_dir=$(mktemp -d /tmp/orca2-r86-fcflags-plants.XXXXXX)
cp "$TARGET_ARCH" "$plant_dir/duplicate.fcm"
printf '%%FCFLAGS -O3\n' >>"$plant_dir/duplicate.fcm"
[[ "$(count_fcflags_assignments "$plant_dir/duplicate.fcm")" -eq 2 ]] || {
  printf 'REFUSE: duplicate-FCFLAGS plant stayed green\n' >&2
  exit 66
}
awk '$1 != "%FCFLAGS"' "$TARGET_ARCH" >"$plant_dir/reference-only.fcm"
[[ "$(count_fcflags_assignments "$plant_dir/reference-only.fcm")" -eq 0 ]] || {
  printf 'REFUSE: reference-only FCFLAGS plant stayed green\n' >&2
  exit 66
}
[[ "$(grep -Fc '%FCFLAGS' "$plant_dir/reference-only.fcm")" -eq 1 ]] || {
  printf 'REFUSE: reference-only plant removed the required reference\n' >&2
  exit 66
}

mkdir -p "$EVIDENCE"
[[ -d "$EVIDENCE" && ! -L "$EVIDENCE" ]] || {
  printf 'REFUSE: evidence path is not a real directory: %s\n' "$EVIDENCE" >&2
  exit 65
}
bash -n "$SCRIPT"
(cd "$SOURCE_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND86_RUNG0_FRAME_DEBUG_RESUME_PREFLIGHT_READY %s assignments=%s literals=%s\n' \
    "$TARGET_RUN" "$assignment_count" "$literal_count"
  exit 0
fi

for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2
    exit 67
  }
done

cd "$NEMO_ROOT"
fcm build --ignore-lock -v 1 -j 1 "$TARGET_BLD"
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: resumed debug build produced no executable\n' >&2; exit 69; }
grep -q 'NEMO_L4_R84FRM1' "$TARGET_ROOT/BLD/ppsrc/nemo/l4_r84_frames.f90" || {
  printf 'REFUSE: compiled debug writer is absent\n' >&2
  exit 69
}
[[ "$(grep -Fc 'CALL r84_dump_frame' "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3.f90")" -eq 4 ]] || {
  printf 'REFUSE: compiled debug source does not have four frame calls\n' >&2
  exit 69
}
grep -F -- '-O0 -g -fbacktrace -fcheck=bounds' "$TARGET_ROOT/BLD/cfg/parsed_bld.cfg" >/dev/null || {
  printf 'REFUSE: resolved debug build flags are absent\n' >&2
  exit 69
}
symbols=$(mktemp /tmp/orca2-r86-symbols.XXXXXX)
nm -D "$BINARY" >"$symbols"
if grep -q '_ZGV' "$symbols"; then
  printf 'REFUSE: vector-math symbol present in debug binary\n' >&2
  exit 69
fi

mkdir "$TARGET_RUN"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$SOURCE_RUN/namelist_cfg" "$TARGET_RUN/namelist_cfg"
cp "$BINARY" "$TARGET_RUN/nemo.debug"
printf '%s\n' "$COMMIT" >"$TARGET_RUN/producer_commit.txt"
printf '%s\n' "$DEBUG_FLAGS" >"$TARGET_RUN/debug_flags.txt"
sha256sum "$MODULE" "$STP_PATCH" "$PREREG" "$SCRIPT" "$TARGET_ARCH" "$TARGET_BLD" \
  >"$TARGET_RUN/toolchain.sha256"
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
cp "$TARGET_ROOT/BLD/cfg/parsed_bld.cfg" "$TARGET_RUN/parsed_bld.cfg"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3.f90" "$TARGET_RUN/stprk3.debug.f90"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/l4_r84_frames.f90" "$TARGET_RUN/l4_r84_frames.debug.f90"
(cd "$TARGET_RUN" && sha256sum nemo.debug parsed_bld.cfg stprk3.debug.f90 \
  l4_r84_frames.debug.f90 run.user.stdout.log run.user.time.log \
  run.exit_code.txt producer_commit.txt debug_flags.txt toolchain.sha256 \
  >round86_debug_outputs.sha256)

[[ "$run_rc" -ne 0 ]] || {
  printf 'REFUSE: debug run exited cleanly; crash prediction was refuted\n' >&2
  exit 70
}
grep -Eiq '(stprk3|l4_r84_frames|l4_oracle_canon_subroutines)\.(f90|h90):[0-9]+' \
  "$TARGET_RUN/run.user.stdout.log" || {
  printf 'REFUSE: debug failure has no source-resolved instrument line\n' >&2
  exit 70
}
grep -Eq 'l4_canon_2d|l4_dump_ocean_surface_input|r84_dump_frame' \
  "$TARGET_RUN/run.user.stdout.log" || {
  printf 'REFUSE: debug failure does not name an instrument routine\n' >&2
  exit 70
}
stage0_count=$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r84_frame_rank*_kt00000001_s0.bin' | wc -l)
later_count=$(find "$TARGET_RUN" -maxdepth 1 -type f \( -name 'oracle_r84_frame_rank*_s1.bin' -o -name 'oracle_r84_frame_rank*_s2.bin' -o -name 'oracle_r84_frame_rank*_s3.bin' \) | wc -l)
[[ "$stage0_count" -eq 2 && "$later_count" -eq 0 ]] || {
  printf 'REFUSE: debug boundary census refuted before-stage-1 prediction: stage0=%s later=%s\n' \
    "$stage0_count" "$later_count" >&2
  exit 70
}
printf 'ORCA2_ROUND86_RUNG0_FRAME_DEBUG_RESUMED %s exit=%s\n' "$TARGET_RUN" "$run_rc"
