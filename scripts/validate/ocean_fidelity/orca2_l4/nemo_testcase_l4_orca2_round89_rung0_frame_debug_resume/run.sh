#!/usr/bin/env bash
# Resume the exact round-88 debug target after its launcher omitted bld.cfg.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-89 rung-0 frame debug resume failed at line %s (exit %s)\n' \
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
readonly SOURCE_CFG=ORCA2_OMIP_L4_R87FRAMES
readonly TARGET_CFG=ORCA2_OMIP_L4_R88FRAMEDEBUG
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly TEMPLATE_BLD=$SOURCE_ROOT/BLD/bld.cfg
readonly FAILED_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round87/acquisition/orca2_rung0_entry_stage_fixed_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round89/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_frame_debug2_resume_np2
readonly PARTIAL_FINGERPRINT=c0bfee549829a77a5443c0a4fba0caa4946dbf4de5054fdfc1f46d287b9bc341
readonly TEMPLATE_BLD_SHA=a2ee6761d650ca8805269c0dbc8cbecf230d3355c65f84bc6b71f4edf49e6bda
readonly TARGET_BLD_SHA=3da736606fab778a4b60c6b2db61c77124e4d2aeb7e33e3153aa730b5574cb54
readonly FAILED_BINARY_SHA=bf254f6dd04463d4095427a40302a48597057f654fc242161e4d49d9c985914b
readonly FAILED_NAMELIST_SHA=d25c69958aeb7d4dffeeab6b08c89f6b314dd7c6d130ed94acfee7cb90643c2c
readonly SOURCE_TRAADV_SHA=b5859bee9b8d9f918a456552c9759943583307a6eacb769c3180d3bd7a38103c
readonly SOURCE_STP_SHA=31f9d62f7ac06b84bc3ef6b5ec94da19695014663c671c31bfc21496e42d1e93
readonly COMPILED_TRAADV_SHA=028411d4c8e2dc09507d340a4afdf5c22c0bf940d63e91e79b51ade84d3aa04a
readonly COMPILED_STP_SHA=3081b9c258f404127d76654069f2e23c8bcaa54cdb5c6dd1dd2fce61b282115e
readonly DEBUG_FLAGS='-fdefault-real-8 -O0 -g -fbacktrace -fcheck=bounds -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize'

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round89.md

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

checkpoint_fingerprint() {
  local root=$1 path rel target digest
  {
    find "$root" -maxdepth 2 -type d -printf '%P\n' | LC_ALL=C sort |
      while IFS= read -r rel; do printf 'D %s\n' "$rel"; done
    find "$root" -maxdepth 2 -type l -printf '%P\n' | LC_ALL=C sort |
      while IFS= read -r rel; do
        target=$(readlink "$root/$rel")
        target=${target//"$root"/TARGET_ROOT}
        target=${target//"$TARGET_ROOT"/TARGET_ROOT}
        printf 'L %s %s\n' "$rel" "$target"
      done
    find "$root" -maxdepth 2 -type f -printf '%P\n' | LC_ALL=C sort |
      while IFS= read -r rel; do
        digest=$(sha256sum "$root/$rel" | awk '{print $1}')
        printf 'F %s %s\n' "$rel" "$digest"
      done
  } | sha256sum | awk '{print $1}'
}

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: debug resume requires a clean committed producer tree\n' >&2; exit 64;
}
readonly COMMIT=$(git rev-parse HEAD)
for path in "$0" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: debug artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
[[ -d "$TARGET_ROOT" && ! -L "$TARGET_ROOT" ]] || {
  printf 'REFUSE: exact partial debug target is absent or a symlink\n' >&2; exit 64;
}
pin "$TEMPLATE_BLD_SHA" "$TEMPLATE_BLD" 'round-87 build template'
pin "$SOURCE_TRAADV_SHA" "$TARGET_ROOT/MY_SRC/traadv.F90" 'partial-target traadv source'
pin "$SOURCE_STP_SHA" "$TARGET_ROOT/MY_SRC/stprk3.F90" 'partial-target stprk3 source'
pin "$FAILED_BINARY_SHA" "$FAILED_RUN/nemo" 'failed optimized binary'
pin "$FAILED_NAMELIST_SHA" "$FAILED_RUN/namelist_cfg" 'failed-run namelist'
(cd "$FAILED_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)

actual_fingerprint=$(checkpoint_fingerprint "$TARGET_ROOT")
[[ "$actual_fingerprint" == "$PARTIAL_FINGERPRINT" ]] || {
  printf 'REFUSE: partial debug target fingerprint changed: %s\n' "$actual_fingerprint" >&2; exit 65;
}
[[ ! -e "$TARGET_RUN" ]] || { printf 'REFUSE: target run already exists\n' >&2; exit 65; }

plant_root=$(mktemp -d /tmp/orca2-r89-checkpoint-plant.XXXXXX)
cp -a "$TARGET_ROOT/." "$plant_root/"
[[ "$(checkpoint_fingerprint "$plant_root")" == "$PARTIAL_FINGERPRINT" ]] || {
  printf 'REFUSE: copied checkpoint control does not reproduce baseline\n' >&2; exit 66;
}
printf '! fingerprint plant\n' >>"$plant_root/MY_SRC/traadv.F90"
[[ "$(checkpoint_fingerprint "$plant_root")" != "$PARTIAL_FINGERPRINT" ]] || {
  printf 'REFUSE: source-change checkpoint plant stayed green\n' >&2; exit 66;
}
advanced_root=$(mktemp -d /tmp/orca2-r89-advanced-plant.XXXXXX)
cp -a "$TARGET_ROOT/." "$advanced_root/"
[[ "$(checkpoint_fingerprint "$advanced_root")" == "$PARTIAL_FINGERPRINT" ]] || {
  printf 'REFUSE: copied advanced-target control does not reproduce baseline\n' >&2; exit 66;
}
mkdir -p "$advanced_root/BLD/bin"
printf 'advanced\n' >"$advanced_root/BLD/bin/nemo.exe"
[[ "$(checkpoint_fingerprint "$advanced_root")" != "$PARTIAL_FINGERPRINT" ]] || {
  printf 'REFUSE: advanced-target checkpoint plant stayed green\n' >&2; exit 66;
}

expected_bld=$(mktemp /tmp/orca2-r89-bld.XXXXXX)
sed "s|$SOURCE_ROOT|$TARGET_ROOT|g" "$TEMPLATE_BLD" >"$expected_bld"
pin "$TARGET_BLD_SHA" "$expected_bld" 're-anchored target build configuration'
bash -n "$0"
mkdir -p "$EVIDENCE"
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND89_RUNG0_FRAME_DEBUG_RESUME_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || { printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67; }
done

cp "$expected_bld" "$TARGET_ROOT/BLD/bld.cfg"
pin "$TARGET_BLD_SHA" "$TARGET_ROOT/BLD/bld.cfg" 'installed target build configuration'
cd "$NEMO_ROOT"
fcm build --ignore-lock -v 1 -j 1 "$TARGET_ROOT/BLD/bld.cfg"
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: resumed debug build produced no executable\n' >&2; exit 69; }
grep -F -- '-O0 -g -fbacktrace -fcheck=bounds' "$TARGET_ROOT/BLD/cfg/parsed_bld.cfg" >/dev/null || {
  printf 'REFUSE: resolved debug flags are absent\n' >&2; exit 69;
}
pin "$COMPILED_TRAADV_SHA" "$TARGET_ROOT/BLD/ppsrc/nemo/traadv.f90" 'compiled debug traadv source'
pin "$COMPILED_STP_SHA" "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3.f90" 'compiled debug stprk3 source'
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in debug binary\n' >&2; exit 69
fi

mkdir "$TARGET_RUN"
while read -r _ name; do cp -a "$FAILED_RUN/$name" "$TARGET_RUN/$name"; done <"$FAILED_RUN/deck_files.sha256"
while read -r _ name; do cp -a "$FAILED_RUN/$name" "$TARGET_RUN/$name"; done <"$FAILED_RUN/input_files.sha256"
cp "$FAILED_RUN/deck_files.sha256" "$FAILED_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$FAILED_RUN/namelist_cfg" "$TARGET_RUN/namelist_cfg"
cp "$BINARY" "$TARGET_RUN/nemo.debug"
printf '%s\n' "$COMMIT" >"$TARGET_RUN/producer_commit.txt"
printf '%s\n' "$DEBUG_FLAGS" >"$TARGET_RUN/debug_flags.txt"
sha256sum "$0" "$PREREG" "$TEMPLATE_BLD" "$TARGET_ROOT/BLD/bld.cfg" \
  "$TARGET_ROOT/BLD/arch_nemo.fcm" >"$TARGET_RUN/toolchain.sha256"
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
grep -Eiq 'traadv\.f90:[0-9]+' "$TARGET_RUN/run.user.stdout.log" || {
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
  toolchain.sha256 >round89_debug_outputs.sha256)
printf 'ORCA2_ROUND89_RUNG0_FRAME_DEBUG_REPRODUCED %s exit=%s\n' "$TARGET_RUN" "$run_rc"
