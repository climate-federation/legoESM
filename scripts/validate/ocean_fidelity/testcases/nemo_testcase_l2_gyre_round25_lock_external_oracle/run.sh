#!/usr/bin/env bash
set -euo pipefail

# User-executed acquisition only.  Codex must not run NEMO from its sandbox.
# The resulting LOCK record closes the one-bit external-mode boundary that the
# existing final-Kaa record cannot localize within dynspg_ts's substep loop.
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=LOCK_EXCHANGE_OMIP_L1_P3
readonly TARGET_CFG=LOCK_EXCHANGE_OMIP_L1_P3_R25BT_SM
readonly INSTRUMENT_CFG=OVERFLOW_OMIP_L1_BTWALK4
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_10
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round25_lock_external_oracle

source_cfg=$NEMO_ROOT/tests/$SOURCE_CFG
target_cfg=$NEMO_ROOT/tests/$TARGET_CFG
instrument=$NEMO_ROOT/tests/$INSTRUMENT_CFG/MY_SRC/dynspg_ts.F90

[[ -d "$source_cfg/MY_SRC" && -d "$source_cfg/EXP00" ]]
[[ -f "$instrument" ]]
if [[ -e "$target_cfg" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: target already exists: %s or %s\n' "$target_cfg" "$TARGET_RUN" >&2
  exit 64
fi

# OVERFLOW and LOCK compile the same shipped dynspg_ts body, and the
# transplanted instrument only ADDS to it.  The previous form here compared
# two WORK entries that are both symlinks to src/OCE/DYN/dynspg_ts.F90 -- a
# file with itself -- so it asserted nothing at all.
readonly SHIPPED=$NEMO_ROOT/src/OCE/DYN/dynspg_ts.F90
[[ -f "$SHIPPED" ]]
for cfg in OVERFLOW_OMIP_L1_P3 "$SOURCE_CFG"; do
  if [[ -e "$NEMO_ROOT/tests/$cfg/MY_SRC/dynspg_ts.F90" ]]; then
    printf 'REFUSE: %s overrides dynspg_ts.F90 in MY_SRC; the shared-shipped-body premise is false\n' \
      "$cfg" >&2
    exit 66
  fi
  if [[ "$(readlink -f "$NEMO_ROOT/tests/$cfg/WORK/dynspg_ts.F90")" != "$SHIPPED" ]]; then
    printf 'REFUSE: %s does not compile the shipped dynspg_ts body\n' "$cfg" >&2
    exit 66
  fi
done
# A WRITE-only instrument may add lines; it may not delete or change one.
# diff prints '<' for every shipped line absent from the instrument.  Count
# them: under `set -o pipefail` the pipeline `diff | grep -q` returns diff's
# own exit 1 whenever the files differ, so the `if` could NEVER fire -- that
# form passed a deliberately deleted-line instrument in a direct test.
if [[ $(diff "$SHIPPED" "$instrument" | grep -c '^<') -ne 0 ]]; then
  printf 'REFUSE: the instrument deletes or changes a shipped line; it must only ADD\n' >&2
  exit 67
fi

work_manifest=$(mktemp -d /tmp/gyre-r25-lock-source.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$work_manifest"
(
  cd "$source_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$instrument" \
  >"$work_manifest/toolchain_and_instrument.sha256"

cd "$NEMO_ROOT"
./makenemo -a LOCK_EXCHANGE -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
cp -r "$source_cfg/EXP00/." "$target_cfg/EXP00/"
cp -r "$source_cfg/MY_SRC/." "$target_cfg/MY_SRC/"
touch "$target_cfg"/MY_SRC/*.F90
cp "$instrument" "$target_cfg/MY_SRC/dynspg_ts.F90"
cp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
./makenemo -n "$TARGET_CFG" -m conda-scalarmath

target_binary=$target_cfg/BLD/bin/nemo.exe
[[ -x "$target_binary" ]]
if nm -D "$target_binary" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in %s\n' "$target_binary" >&2
  exit 65
fi
sha256sum "$target_binary" >"$work_manifest/binary.sha256"

mkdir "$TARGET_RUN"
for name in namelist_cfg namelist_ref context_nemo.xml file_def_nemo-oce.xml \
  iodef.xml; do
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
cp "$target_binary" "$TARGET_RUN/nemo"
cp "$work_manifest"/*.sha256 "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } \
    2>>run.user.time.log
  test "${PIPESTATUS[0]}" -eq 0
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>run.user.time.log
)

# The transplanted block only snapshots arrays.  Require all pre-existing
# identity records and the final state to remain byte-identical before admitting
# the new substep record.  Its historical OVERFLOW filename is deliberately
# retained so the existing parser can consume it through --oracle.
# Binding control against stale-timestamp builds: every MY_SRC writer must be
# in the compiled preprocessed source, or the run cannot have written them.
for writer in stprk3 stprk3_stg dynspg_ts; do
  grep -q 'oracle_' "$target_cfg/BLD/ppsrc/nemo/$writer.f90" \
    || { printf 'REFUSE: MY_SRC writer %s not compiled (stale build)\n' "$writer" >&2; exit 69; }
done
for source_record in "$SOURCE_RUN"/oracle_step_entry_kt*.bin "$SOURCE_RUN"/oracle_bt_frames_kt*.bin; do
  record=${source_record##*/}
  cmp "$source_record" "$TARGET_RUN/$record"
done
cmp "$SOURCE_RUN/LOCK_EXCHANGE_OMIP_L1_ZCO_00000010_restart.nc" \
    "$TARGET_RUN/LOCK_EXCHANGE_OMIP_L1_ZCO_00000010_restart.nc"
test -s "$TARGET_RUN/oracle_overflow_bt_substeps_kt00000001_call1.bin"
(
  cd "$TARGET_RUN"
  sha256sum oracle_*.bin LOCK_EXCHANGE_OMIP_L1_ZCO_00000010_restart.nc \
    >round25_outputs.sha256
)
printf 'ROUND25_LOCK_EXTERNAL_ORACLE_READY %s\n' "$TARGET_RUN"
