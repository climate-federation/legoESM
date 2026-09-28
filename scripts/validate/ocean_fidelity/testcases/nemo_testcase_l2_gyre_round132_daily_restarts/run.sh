#!/usr/bin/env bash
set -Eeuo pipefail

# USER-EXECUTED ACQUISITION ONLY. The agent may run --preflight or
# --resume-preflight, but must not run the default path because it invokes
# mpirun. The default resumes the already-built Round-132 target; it never
# invokes makenemo.
refuse_on_error() {
  local status=$?
  trap - ERR
  printf 'REFUSE: round132 acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

refuse() {
  local status=$1
  shift
  trap - ERR
  printf 'REFUSE: %s\n' "$*" >&2
  exit "$status"
}

if [[ $# -gt 1 ]]; then
  refuse 64 "usage: $0 [--acquire|--preflight|--resume|--resume-preflight|--plant-binary]"
fi
readonly MODE=${1:---resume}
case "$MODE" in
  --acquire|--preflight|--resume|--resume-preflight|--plant-binary) ;;
  *) refuse 64 "unknown mode $MODE; use --resume-preflight or no argument" ;;
esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R41ADVSP
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R132DAILY
readonly ARCH=conda-scalarmath
readonly PHASE3=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
readonly ROUND132=$PHASE3/round132
readonly ROUND133=$PHASE3/round133
readonly DAILY_CONTROL=$PHASE3/year_owners/nemo_seed0
readonly MONTHLY_CONTROL=$PHASE3/year_fromrest/nemo_seed0
readonly TARGET_RUN=$ROUND132/oracle_daily_restarts

readonly NN_ITEND=2160
readonly NN_STOCK=6
readonly NN_WRITE=2160
readonly EXPECTED_COUNT=360
readonly EXPECTED_SIZE=1466328
readonly EXPECTED_TOTAL=527878080
readonly SOURCE_EXP_COUNT=55
readonly SOURCE_MY_COUNT=14
readonly SOURCE_BINARY_SHA=a759e8b478e3bda5ba731009fd353159db892c5ac2ba4c48351f5136411960cd
readonly SOURCE_CARD_MANIFEST_SHA=977818735d03095ffbbac145014db463b781e543a0d77bc3ae6228ce79e7887e
readonly SOURCE_CPP_SHA=54bd2cead92cee1eaa8cb257c7f7cdea8fe39cc9fb0f47c7c0d5853b909b34b7
readonly ARCH_SHA=132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561
readonly DAY30_SHA=853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6
readonly TARGET_BINARY_SHA=24aefbfb9f4b596c4c0c8002c17811577f3622b4c4bf34add13d770caf988eff
readonly ROUND132_COMMIT=e9f3bb6a557581f0cc5b18a60a27757b60f49ef7

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly HERE=$here
readonly REPO=$(CDPATH= cd -- "$HERE/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_BINARY=$SOURCE_ROOT/BLD/bin/nemo.exe
readonly SOURCE_CPP=$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm
readonly TARGET_CPP=$TARGET_ROOT/cpp_$TARGET_CFG.fcm
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly MPIRUN=/home/dbalwada/miniconda3/envs/nemo-build/bin/mpirun
readonly NM=/usr/bin/nm
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly GATE=$HERE/../nemo_testcase_l2_gyre_round131_daily_record_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round132.md
readonly RECOVERY_PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round133.md
readonly OLD_DAY30=$DAILY_CONTROL/GYRE_OMIP_L2_P3_00000180_restart.nc
readonly MONTHLY_DAY30=$MONTHLY_CONTROL/GYRE_OMIP_L2_P3_00000180_restart.nc
readonly PREPARED="namelist_ref namelist_top_cfg namelist_top_ref \
namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
field_def_nemo-oce.xml field_def_nemo-pisces.xml"

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  refuse 63 "acquisition requires a clean committed tree"
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LC_ALL=C

for path in "$SOURCE_ROOT/EXP00" "$SOURCE_ROOT/MY_SRC"; do
  [[ -d "$path" ]] || refuse 64 "missing source-card directory $path"
done
for path in "$SOURCE_BINARY" "$SOURCE_CPP" \
            "$NEMO_ROOT/arch/arch-$ARCH.fcm" "$FC" "$MPIRUN" "$NM" \
            "$PY" "$GATE" \
            "$PREREG" "$RECOVERY_PREREG" "$DAILY_CONTROL/namelist_cfg" \
            "$MONTHLY_CONTROL/namelist_cfg" "$OLD_DAY30" "$MONTHLY_DAY30"; do
  [[ -f "$path" ]] || refuse 64 "missing frozen acquisition input $path"
done
[[ -x "$SOURCE_BINARY" ]] || refuse 64 "source binary is not executable"
[[ -x "$FC" && -x "$MPIRUN" && -x "$NM" && -x "$PY" ]] || \
  refuse 64 "compiler, MPI launcher, nm, or Python is not executable"

if [[ "$(readlink -m "$TARGET_ROOT")" != \
      "$(readlink -m "$NEMO_ROOT/cfgs/$TARGET_CFG")" ]]; then
  refuse 64 "target configuration escaped its registered path"
fi
if [[ "$(readlink -m "$TARGET_RUN")" != \
      "$(readlink -m "$ROUND132/oracle_daily_restarts")" ]]; then
  refuse 64 "target run escaped its registered path"
fi
if [[ "$MODE" == "--acquire" || "$MODE" == "--preflight" ]]; then
  if [[ -e "$TARGET_ROOT" || -e "$TARGET_RUN" ]]; then
    refuse 64 "new target already exists: $TARGET_ROOT or $TARGET_RUN"
  fi
else
  [[ -d "$TARGET_ROOT" && -d "$TARGET_RUN" ]] || \
    refuse 64 "retained target is incomplete: $TARGET_ROOT or $TARGET_RUN"
fi

source_card_manifest() {
  (
    cd "$1"
    find EXP00 MY_SRC -maxdepth 2 \( -type f -o -type l \) -print0 \
      | sort -z | xargs -0 sha256sum
  )
}

copy_card_manifest() {
  (
    cd "$1"
    while IFS= read -r -d '' path; do
      if [[ -L "$path" ]]; then
        printf 'L\t%s\t%s\n' "$path" "$(readlink "$path")"
      else
        fields=$(sha256sum "$path")
        printf 'F\t%s\t%s\n' "$path" "${fields%% *}"
      fi
    done < <(find EXP00 MY_SRC -maxdepth 2 \
      \( -type f -o -type l \) -print0 | sort -z)
  )
}

digest_of() {
  local fields
  fields=$(sha256sum "$1")
  printf '%s\n' "${fields%% *}"
}

verify_retained_preparation() {
  local path name producer expected_binary recorded_binary extra
  local built_binary staged_binary partial_output
  for path in "$BINARY" "$TARGET_RUN/nemo" "$TARGET_RUN/binary.sha256" \
              "$TARGET_RUN/producer_commit.txt" \
              "$TARGET_RUN/source_cfg.sha256" \
              "$TARGET_RUN/source_copy_card.manifest" \
              "$TARGET_RUN/target_copy_card_before_build.manifest" \
              "$TARGET_RUN/toolchain.sha256" "$TARGET_RUN/namelist_cfg" \
              "$TARGET_RUN/run.user.stdout.log" \
              "$TARGET_RUN/run.user.time.log" "$TARGET_CPP"; do
    [[ -f "$path" ]] || refuse 66 "retained acquisition lacks $path"
  done
  [[ -x "$BINARY" && -x "$TARGET_RUN/nemo" ]] || \
    refuse 66 "retained built or staged binary is not executable"

  producer=$(tr -d '[:space:]' <"$TARGET_RUN/producer_commit.txt")
  [[ "$producer" == "$ROUND132_COMMIT" ]] || \
    refuse 66 "retained producer commit is $producer, expected $ROUND132_COMMIT"

  [[ "$(wc -l <"$TARGET_RUN/binary.sha256")" -eq 1 ]] || \
    refuse 66 "retained binary manifest is not exactly one row"
  read -r expected_binary recorded_binary extra <"$TARGET_RUN/binary.sha256"
  [[ -z "${extra:-}" && "$recorded_binary" == "$BINARY" ]] || \
    refuse 66 "retained binary manifest names an unexpected target"
  [[ "$expected_binary" == "$TARGET_BINARY_SHA" ]] || \
    refuse 66 "retained binary manifest moved from $TARGET_BINARY_SHA"
  if [[ "$MODE" == "--plant-binary" ]]; then
    expected_binary=0000000000000000000000000000000000000000000000000000000000000000
  fi
  built_binary=$(digest_of "$BINARY")
  staged_binary=$(digest_of "$TARGET_RUN/nemo")
  if [[ "$built_binary" != "$expected_binary" || \
        "$staged_binary" != "$expected_binary" ]]; then
    if [[ "$MODE" == "--plant-binary" ]]; then
      printf 'REFUSE: STATUS PLANT-FIRED: retained-binary-identity\n' >&2
      exit 1
    fi
    refuse 66 "retained built or staged binary differs from binary.sha256"
  fi
  if [[ "$MODE" == "--plant-binary" ]]; then
    refuse 2 "retained-binary-identity plant stayed green"
  fi

  cmp -s "$PROVENANCE/source_cfg.sha256" \
    "$TARGET_RUN/source_cfg.sha256" || \
    refuse 66 "frozen source-card digest manifest moved"
  cmp -s "$PROVENANCE/source_copy_card.manifest" \
    "$TARGET_RUN/source_copy_card.manifest" || \
    refuse 66 "frozen link-aware source-card manifest moved"
  cmp -s "$TARGET_RUN/source_copy_card.manifest" \
    "$TARGET_RUN/target_copy_card_before_build.manifest" || \
    refuse 66 "retained pre-build target card differs from the source card"
  [[ "$(awk '{print $1}' "$PROVENANCE/toolchain.sha256")" == \
     "$(awk '{print $1}' "$TARGET_RUN/toolchain.sha256")" ]] || \
    refuse 66 "retained Round-132 toolchain manifest moved"
  cmp -s "$SOURCE_CPP" "$TARGET_CPP" || \
    refuse 66 "retained target cpp keys differ from the source card"

  for name in restart stprk3 dynspg_ts zdftke; do
    path=$TARGET_ROOT/BLD/ppsrc/nemo/$name.f90
    [[ -f "$path" ]] || refuse 66 "retained target lacks compiled $name.f90"
    cmp -s "$SOURCE_ROOT/BLD/ppsrc/nemo/$name.f90" "$path" || \
      refuse 66 "retained compiled $name.f90 differs from the source branch"
  done
  if "$NM" -D "$BINARY" | grep -q '_ZGV'; then
    refuse 66 "vector-math symbol present in the retained binary"
  fi

  for name in $PREPARED; do
    [[ -e "$TARGET_ROOT/EXP00/$name" && -f "$TARGET_RUN/$name" ]] || \
      refuse 66 "retained staged input is missing: $name"
    cmp -s "$TARGET_ROOT/EXP00/$name" "$TARGET_RUN/$name" || \
      refuse 66 "retained staged input differs from target card: $name"
  done
  cmp -s "$PROVENANCE/namelist_cfg" "$TARGET_RUN/namelist_cfg" || \
    refuse 66 "retained staged namelist differs from the proved three-row delta"

  partial_output=$(find "$TARGET_RUN" -maxdepth 1 -type f \
    \( -name '*.nc' -o -name 'ocean.output*' \) -print -quit)
  [[ -z "$partial_output" ]] || \
    refuse 66 "retained run is not empty; found NEMO output $partial_output"
  grep -Fq '/usr/bin/time: No such file or directory' \
    "$TARGET_RUN/run.user.stdout.log" || \
    refuse 66 "retained failure log lacks the registered missing-wrapper error"
  if grep -hEq '^(STOP 0|NEMO_DONE|RUN_DONE)$' \
      "$TARGET_RUN"/run.user.*.log; then
    refuse 66 "retained failed run contains a completion marker"
  fi
  for path in "$TARGET_RUN/run.resume.time.log" \
              "$TARGET_RUN/run.resume.stdout.log" \
              "$TARGET_RUN/round133_resume_commit.txt" \
              "$TARGET_RUN/round133_daily_record_audit.json" \
              "$TARGET_RUN/round133_daily_record_audit.log" \
              "$TARGET_RUN/round133_daily_restarts.sha256" \
              "$TARGET_RUN/round133_daily_restarts.stamp"; do
    [[ ! -e "$path" ]] || \
      refuse 66 "retained target already has a recovery output: $path"
  done

  printf 'FAILED_RUN_PASS restarts=0 ocean_output=0 completion_markers=0\n'
  printf 'RETAINED_BUILD_PASS binary_sha256=%s compiled_files=4 staged_inputs=13 producer=%s\n' \
    "$TARGET_BINARY_SHA" "$producer"
}

readonly PROVENANCE=$(mktemp -d /tmp/gyre-r132-provenance.XXXXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$PROVENANCE"
printf '%s\n' "$COMMIT" >"$PROVENANCE/producer_commit.txt"
source_card_manifest "$SOURCE_ROOT" >"$PROVENANCE/source_cfg.sha256"
copy_card_manifest "$SOURCE_ROOT" >"$PROVENANCE/source_copy_card.manifest"
sha256sum "$NEMO_ROOT/arch/arch-$ARCH.fcm" "$SOURCE_CPP" \
  "$SOURCE_BINARY" "$GATE" "$PREREG" \
  >"$PROVENANCE/toolchain.sha256"

actual=$(digest_of "$SOURCE_BINARY")
[[ "$actual" == "$SOURCE_BINARY_SHA" ]] || \
  refuse 65 "source binary hash is $actual, expected $SOURCE_BINARY_SHA"
actual=$(digest_of "$PROVENANCE/source_cfg.sha256")
[[ "$actual" == "$SOURCE_CARD_MANIFEST_SHA" ]] || \
  refuse 65 "source-card manifest hash is $actual, expected $SOURCE_CARD_MANIFEST_SHA"
actual=$(digest_of "$SOURCE_CPP")
[[ "$actual" == "$SOURCE_CPP_SHA" ]] || \
  refuse 65 "source cpp-key hash is $actual, expected $SOURCE_CPP_SHA"
actual=$(digest_of "$NEMO_ROOT/arch/arch-$ARCH.fcm")
[[ "$actual" == "$ARCH_SHA" ]] || \
  refuse 65 "scalar-math architecture hash is $actual, expected $ARCH_SHA"

exp_count=$(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \
  \( -type f -o -type l \) | wc -l)
my_count=$(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \
  \( -type f -o -type l \) | wc -l)
[[ "$exp_count" -eq "$SOURCE_EXP_COUNT" ]] || \
  refuse 65 "source EXP00 count is $exp_count, expected $SOURCE_EXP_COUNT"
[[ "$my_count" -eq "$SOURCE_MY_COUNT" ]] || \
  refuse 65 "source MY_SRC count is $my_count, expected $SOURCE_MY_COUNT"

actual=$(digest_of "$OLD_DAY30")
[[ "$actual" == "$DAY30_SHA" ]] || \
  refuse 65 "old daily day-30 hash is $actual, expected $DAY30_SHA"
actual=$(digest_of "$MONTHLY_DAY30")
[[ "$actual" == "$DAY30_SHA" ]] || \
  refuse 65 "monthly day-30 hash is $actual, expected $DAY30_SHA"
cmp -s "$OLD_DAY30" "$MONTHLY_DAY30" || \
  refuse 65 "existing daily and monthly day-30 controls differ"

"$PY" - "$DAILY_CONTROL" "$MONTHLY_CONTROL" \
  "$EXPECTED_SIZE" <<'PYCONTROL'
import re
import sys
from pathlib import Path

daily, monthly = map(Path, sys.argv[1:3])
expected_size = int(sys.argv[3])
pattern = re.compile(r"^GYRE_OMIP_L2_P3_([0-9]{8})_restart[.]nc$")

def steps(root):
    rows = {}
    for path in root.glob("*restart.nc"):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise SystemExit(f"REFUSE: unexpected control restart {path.name}")
        step = int(match.group(1))
        if step in rows:
            raise SystemExit(f"REFUSE: duplicate control restart kt={step}")
        if path.stat().st_size != expected_size:
            raise SystemExit(
                f"REFUSE: control restart {path.name} has {path.stat().st_size} "
                f"bytes, expected {expected_size}")
        rows[step] = path
    return rows

daily_rows = steps(daily)
monthly_rows = steps(monthly)
if set(daily_rows) != set(range(6, 181, 6)):
    raise SystemExit("REFUSE: existing daily control is not kt=6..180 by 6")
if set(monthly_rows) != set(range(180, 2161, 180)):
    raise SystemExit("REFUSE: existing monthly control is not kt=180..2160 by 180")
print("CONTROL_CENSUS_PASS daily=30 monthly=12 bytes=1466328")
PYCONTROL

write_namelist() {
  "$PY" - "$SOURCE_ROOT/EXP00/namelist_cfg" "$1" \
    "$NN_ITEND" "$NN_STOCK" "$NN_WRITE" <<'PYNAMELIST'
import re
import sys

src, dst, itend, stock, write = sys.argv[1:6]
text = open(src, encoding="utf-8").read()
for key, value in (("nn_itend", itend), ("nn_stock", stock),
                   ("nn_write", write)):
    text, count = re.subn(
        rf"^(\s*{key}\s*=\s*)(\S+)", rf"\g<1>{value}", text,
        count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"REFUSE: {key} not found exactly once in {src}")
open(dst, "w", encoding="utf-8").write(text)

assignment = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*(?:!.*)?$")

def rows(blob):
    result = {}
    for line in blob.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("&"):
            continue
        match = assignment.match(line)
        if match:
            result[match.group(1)] = match.group(2)
    return result

before = rows(open(src, encoding="utf-8").read())
after = rows(text)
changed = sorted(
    key for key in before if key in after and before[key] != after[key])
added = sorted(set(after) - set(before))
removed = sorted(set(before) - set(after))
allowed = {"nn_itend", "nn_stock", "nn_write"}
expected = {"nn_itend": itend, "nn_stock": stock, "nn_write": write}
print("namelist assignments changed from the source card:")
for key in changed:
    print(f"  {key}: {before[key]} -> {after[key]}")
if set(changed) != allowed or added or removed:
    raise SystemExit(
        f"REFUSE: changed={changed} added={added} removed={removed}; "
        f"only {sorted(allowed)} may change")
for key, value in expected.items():
    if after.get(key) != value:
        raise SystemExit(
            f"REFUSE: {key} resolved to {after.get(key)}, expected {value}")
if "nn_pert_seed" in after:
    raise SystemExit("REFUSE: unperturbed R41 card gained nn_pert_seed")
PYNAMELIST
}

write_namelist "$PROVENANCE/namelist_cfg"

readonly SYNTAX_DIR=$(mktemp -d /tmp/gyre-r132-syntax.XXXXXXXX)
printf 'temporary syntax-proof directory (retained): %s\n' "$SYNTAX_DIR"
: >"$PROVENANCE/syntax_proof.log"
for name in restart stprk3 dynspg_ts zdftke; do
  "$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
    -J "$SYNTAX_DIR" "$SOURCE_ROOT/BLD/ppsrc/nemo/$name.f90"
  printf 'SYNTAX_PROOF_PASS %s.f90\n' "$name" \
    | tee -a "$PROVENANCE/syntax_proof.log"
done

for mount in /tmp "$ROUND132" "$ROUND133" "$NEMO_ROOT"; do
  mkdir -p "$mount"
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || \
    refuse 67 "$mount has $free_kb kB free, under the 4 GB floor"
done

if [[ "$MODE" == "--preflight" ]]; then
  printf 'ROUND132_DAILY_RESTART_PREFLIGHT_READY %s\n' "$PROVENANCE"
  exit 0
fi

if [[ "$MODE" == "--resume" || "$MODE" == "--resume-preflight" || \
      "$MODE" == "--plant-binary" ]]; then
  verify_retained_preparation
  if [[ "$MODE" == "--resume-preflight" ]]; then
    printf 'ROUND133_DAILY_RESTART_RESUME_PREFLIGHT_READY %s\n' "$TARGET_RUN"
    exit 0
  fi
fi

if [[ "$MODE" == "--acquire" ]]; then
  cd "$NEMO_ROOT"
  ./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m "$ARCH" del_key 'key_xios'
  while IFS= read -r -d '' source; do
    cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
  done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \
    \( -type f -o -type l \) -print0 | sort -z)
  while IFS= read -r -d '' source; do
    cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
  done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \
    \( -type f -o -type l \) -print0 | sort -z)
  cp "$SOURCE_CPP" "$TARGET_CPP"

  copy_card_manifest "$TARGET_ROOT" \
    >"$PROVENANCE/target_copy_card_before_build.manifest"
  cmp -s "$PROVENANCE/source_copy_card.manifest" \
    "$PROVENANCE/target_copy_card_before_build.manifest" || \
    refuse 68 "file-by-file target card differs from the source card"
  cmp -s "$SOURCE_CPP" "$TARGET_CPP" || \
    refuse 68 "target cpp keys differ from the source card"
  target_exp_count=$(find "$TARGET_ROOT/EXP00" -maxdepth 1 \
    \( -type f -o -type l \) | wc -l)
  target_my_count=$(find "$TARGET_ROOT/MY_SRC" -maxdepth 1 \
    \( -type f -o -type l \) | wc -l)
  [[ "$target_exp_count" -eq "$SOURCE_EXP_COUNT" ]] || \
    refuse 68 "target EXP00 count is $target_exp_count, expected $SOURCE_EXP_COUNT"
  [[ "$target_my_count" -eq "$SOURCE_MY_COUNT" ]] || \
    refuse 68 "target MY_SRC count is $target_my_count, expected $SOURCE_MY_COUNT"

  touch "$TARGET_ROOT/MY_SRC/"*.F90
  ./makenemo -n "$TARGET_CFG" -m "$ARCH"
  [[ -x "$BINARY" ]] || refuse 68 "new target binary is not executable"
  for name in restart stprk3 dynspg_ts zdftke; do
    source=$SOURCE_ROOT/BLD/ppsrc/nemo/$name.f90
    target=$TARGET_ROOT/BLD/ppsrc/nemo/$name.f90
    [[ -f "$target" ]] || refuse 68 "new target lacks compiled $name.f90"
    cmp -s "$source" "$target" || \
      refuse 68 "compiled $name.f90 differs from the source-card branch"
  done
  if "$NM" -D "$BINARY" | grep -q '_ZGV'; then
    refuse 68 "vector-math symbol present in the new target binary"
  fi
  sha256sum "$BINARY" >"$PROVENANCE/binary.sha256"

  mkdir "$TARGET_RUN"
  for name in $PREPARED; do
    [[ -e "$TARGET_ROOT/EXP00/$name" ]] || \
      refuse 68 "copied source card lacks prepared input $name"
    cp -L "$TARGET_ROOT/EXP00/$name" "$TARGET_RUN/$name"
  done
  write_namelist "$TARGET_RUN/namelist_cfg"
  cmp -s "$PROVENANCE/namelist_cfg" "$TARGET_RUN/namelist_cfg" || \
    refuse 68 "staged namelist differs from the preflight-proved namelist"
  cp "$BINARY" "$TARGET_RUN/nemo"
  cmp -s "$BINARY" "$TARGET_RUN/nemo" || \
    refuse 68 "staged executable differs from the new target binary"
  cp "$PROVENANCE"/* "$TARGET_RUN/"
else
  printf '%s\n' "$COMMIT" >"$TARGET_RUN/round133_resume_commit.txt"
fi

if [[ "$MODE" == "--resume" ]]; then
  readonly RUN_TIME_LOG=$TARGET_RUN/run.resume.time.log
  readonly RUN_STDOUT_LOG=$TARGET_RUN/run.resume.stdout.log
else
  readonly RUN_TIME_LOG=$TARGET_RUN/run.user.time.log
  readonly RUN_STDOUT_LOG=$TARGET_RUN/run.user.stdout.log
fi
for path in "$RUN_TIME_LOG" "$RUN_STDOUT_LOG"; do
  [[ ! -e "$path" ]] || refuse 69 "run log already exists: $path"
done

cd "$TARGET_RUN"
export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$RUN_TIME_LOG"
if "$MPIRUN" -np 1 --oversubscribe ./nemo 2>&1 | tee "$RUN_STDOUT_LOG"; then
  :
else
  run_status=$?
  refuse 69 "NEMO acquisition pipeline exited $run_status"
fi
if ! grep -Fxq 'STOP 0' "$RUN_STDOUT_LOG"; then
  refuse 69 "NEMO run did not terminate with STOP 0"
fi
printf 'NEMO_FINISHED_UTC=%s\nNEMO_DONE\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$RUN_TIME_LOG"

for pattern in \
  'ocean time step.*rn_Dt.*14400\.000000000000' \
  'number of the last time step.*nn_itend.*= *2160' \
  'frequency of restart file.*nn_stock.*= *6' \
  'frequency of output file.*nn_write.*= *2160'; do
  grep -Eq "$pattern" "$TARGET_RUN/ocean.output" || \
    refuse 69 "resolved run lacks expected row: $pattern"
done

"$PY" - "$TARGET_RUN" "$EXPECTED_COUNT" "$EXPECTED_SIZE" \
  "$EXPECTED_TOTAL" <<'PYRECORDS'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_count, expected_size, expected_total = map(int, sys.argv[2:5])
expected_steps = set(range(6, 2161, 6))
pattern = re.compile(r"^GYRE_OMIP_L2_P3_([0-9]{8})_restart[.]nc$")
records = {}
unexpected = []
for path in root.glob("*restart*.nc"):
    match = pattern.fullmatch(path.name)
    if match is None:
        unexpected.append(path.name)
        continue
    step = int(match.group(1))
    if step in records:
        raise SystemExit(f"REFUSE: duplicate target restart kt={step}")
    records[step] = path
if unexpected:
    raise SystemExit(f"REFUSE: unexpected restart files {sorted(unexpected)}")
if set(records) != expected_steps:
    missing = sorted(expected_steps - set(records))
    extra = sorted(set(records) - expected_steps)
    raise SystemExit(
        f"REFUSE: restart steps differ; missing={missing[:4]} extra={extra[:4]}")
if len(records) != expected_count:
    raise SystemExit(
        f"REFUSE: restart count is {len(records)}, expected {expected_count}")
wrong = [(step, path.stat().st_size) for step, path in records.items()
         if path.stat().st_size != expected_size]
if wrong:
    raise SystemExit(
        f"REFUSE: restart size mismatch begins at kt={wrong[0][0]}: "
        f"{wrong[0][1]} bytes, expected {expected_size}")
total = sum(path.stat().st_size for path in records.values())
if total != expected_total:
    raise SystemExit(
        f"REFUSE: restart total is {total}, expected {expected_total}")
print(
    f"RESTART_CENSUS_PASS count={len(records)} bytes_each={expected_size} "
    f"total={total}")
PYRECORDS

for step in $(seq 180 180 2160); do
  name=$(printf 'GYRE_OMIP_L2_P3_%08d_restart.nc' "$step")
  cmp -s "$TARGET_RUN/$name" "$MONTHLY_CONTROL/$name" || \
    refuse 71 "monthly twin differs at kt=$step ($name)"
  printf 'TWIN_IDENTICAL %s\n' "$name"
done
cmp -s "$TARGET_RUN/GYRE_OMIP_L2_P3_00000180_restart.nc" \
  "$OLD_DAY30" || refuse 71 "new day-30 restart differs from old daily control"
printf 'TWIN_IDENTICAL_OLD_DAILY GYRE_OMIP_L2_P3_00000180_restart.nc\n'

"$PY" "$GATE" --self-check \
  >"$TARGET_RUN/round133_gate_self_check.log" 2>&1
grep -Fq 'SELF-CHECK OK' "$TARGET_RUN/round133_gate_self_check.log" || \
  refuse 72 "Round-131 complete-record self-check lacks its success marker"
for plant in missing-boundary required-variable; do
  log=$TARGET_RUN/round133_${plant}_plant.log
  if "$PY" "$GATE" --self-check --plant "$plant" >"$log" 2>&1; then
    refuse 72 "Round-131 $plant plant stayed green"
  else
    plant_status=$?
  fi
  if [[ "$plant_status" -ne 1 ]]; then
    refuse 72 "Round-131 $plant plant exited $plant_status, expected 1"
  fi
  grep -Fq "STATUS PLANT-FIRED: $plant" "$log" || \
    refuse 72 "Round-131 $plant plant lacks its fired marker"
done

"$PY" "$GATE" --audit --daily-root "$TARGET_RUN" \
  --monthly-root "$MONTHLY_CONTROL" --expect-commit "$COMMIT" \
  --output "$TARGET_RUN/round133_daily_record_audit.json" \
  | tee "$TARGET_RUN/round133_daily_record_audit.log"
grep -Fxq 'STATUS ADMITTED' "$TARGET_RUN/round133_daily_record_audit.log" || \
  refuse 72 "daily-record gate did not print STATUS ADMITTED"

cd "$TARGET_RUN"
sha256sum GYRE_OMIP_L2_P3_*_restart.nc >round133_daily_restarts.sha256
[[ "$(wc -l <round133_daily_restarts.sha256)" -eq "$EXPECTED_COUNT" ]] || \
  refuse 73 "restart digest manifest does not contain $EXPECTED_COUNT rows"
manifest_digest=$(digest_of round133_daily_restarts.sha256)
printf '%s %s %s\n' "$manifest_digest" "$COMMIT" \
  round133_daily_restarts.sha256 >round133_daily_restarts.stamp
sha256sum round133_daily_restarts.sha256 round133_daily_restarts.stamp \
  round133_daily_record_audit.json round133_daily_record_audit.log \
  round133_gate_self_check.log round133_*_plant.log \
  >round133_outputs.sha256
printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$RUN_TIME_LOG"

printf 'ROUND133_DAILY_RESTART_RECORD_READY %s\n' "$TARGET_RUN"
