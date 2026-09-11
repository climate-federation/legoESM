#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED ACQUISITION. The agent writes this script and stops; only the
# operator may invoke makenemo/mpirun. Canonical NEMO src/ is never modified.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R46KT2
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R48BTMEM
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/oracle_kt2_stage
readonly ROUND48=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round48
readonly TARGET_RUN=$ROUND48/oracle_bt_memory
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly GATE=$here/../nemo_testcase_l2_gyre_round48_bt_memory_gate.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_phase3_round48.md
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly PATCH=$here/dynspg_ts_round48.patch
readonly WRITER=$here/l2_r48_bt_memory.F90

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63;
}
readonly COMMIT=$(git rev-parse HEAD)
git merge-base --is-ancestor f826f1e6c8f9 "$COMMIT"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$GATE" "$ADMISSION" "$PREREG" "$PATCH" "$WRITER"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 64; }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]]
[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2; exit 64;
}
grep -Eq 'nn_itend[[:space:]]*=[[:space:]]*([2-9]|[1-9][0-9]+)' \
  "$SOURCE_ROOT/EXP00/namelist_cfg"
grep -Eq 'nn_bt_flt[[:space:]]*=[[:space:]]*3' "$SOURCE_ROOT/EXP00/namelist_cfg"
grep -Eq 'ln_bt_fw[[:space:]]*=[[:space:]]*\.true\.' "$SOURCE_ROOT/EXP00/namelist_ref"
grep -q 'ln_bt_fw=T => Forward integration' "$SOURCE_RUN/ocean.output"
grep -q 'ROUND46_STAGE_BEGIN' "$SOURCE_ROOT/MY_SRC/l2_r46_stage.F90"

# Patch proof: insertion only relative to the exact R46 card; no parent model
# or inherited writer statement is removed. The recorder has INTENT(IN) model
# arrays and opens only ACTION='WRITE' streams.
dry=$(mktemp -d /tmp/gyre-r48-source.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$dry/dynspg_ts.F90"
patch -s "$dry/dynspg_ts.F90" <"$PATCH"
removed=$(diff "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$dry/dynspg_ts.F90" | grep -c '^<' || true)
[[ "$removed" -eq 0 ]] || { printf 'REFUSE: patch removes %s parent lines\n' "$removed" >&2; exit 65; }
[[ "$(grep -c 'CALL r48_bt_memory' "$dry/dynspg_ts.F90")" -eq 2 ]]
grep -q "STATUS='REPLACE', ACTION='WRITE'" "$WRITER"
grep -q 'INTENT(in) :: puu_b, pvv_b, pssh' "$WRITER"
grep -q 'INTENT(in) :: pubb_e, pub_e, pvbb_e, pvb_e' "$WRITER"

for mount in /tmp "$ROUND48" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2; exit 66;
  }
done

manifest=$(mktemp -d /tmp/gyre-r48-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$WRITER" "$PATCH" \
  "$GATE" "$ADMISSION" "$PREREG" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
cp -r "$SOURCE_ROOT/EXP00/." "$TARGET_ROOT/EXP00/"
cp -r "$SOURCE_ROOT/MY_SRC/." "$TARGET_ROOT/MY_SRC/"
cp "$SOURCE_ROOT/cpp_${SOURCE_CFG}.fcm" "$TARGET_ROOT/cpp_${TARGET_CFG}.fcm"
(
  cd "$TARGET_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/copied_cfg_before_patch.sha256"
cmp "$manifest/source_cfg.sha256" "$manifest/copied_cfg_before_patch.sha256"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l2_r48_bt_memory.F90"
patch -s "$TARGET_ROOT/MY_SRC/dynspg_ts.F90" <"$PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
binary=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$binary" ]]
grep -q 'CALL r48_bt_memory' "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90"
grep -q 'NEMO_L2_R48BTM1' "$TARGET_ROOT/BLD/ppsrc/nemo/l2_r48_bt_memory.f90"
grep -q 'puu_b(:,:,Kaa) = ua_e' "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90"
if nm -D "$binary" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2; exit 67
fi
sha256sum "$binary" >"$manifest/binary.sha256"

mkdir -p "$ROUND48"
mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref \
namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done
cp "$binary" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  for name in $PREPARED; do cmp "$SOURCE_RUN/$name" "$name"; done
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } \
    2>>run.user.time.log
  test "${PIPESTATUS[0]}" -eq 0
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>run.user.time.log
)

readonly NEW="oracle_bt_memory_kt00000001_end.bin oracle_bt_memory_kt00000002_start.bin"
"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new $NEW --output "$TARGET_RUN/round48_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new $NEW --plant-consumed \
     --output "$TARGET_RUN/round48_admission_plant.json" \
     >"$TARGET_RUN/round48_admission_plant.log" 2>&1; then
  printf 'REFUSE: consumed-field twin plant stayed green\n' >&2; exit 68
fi
grep -q '"plant_applied": true' "$TARGET_RUN/round48_admission_plant.json"

gate_common=(--root "$TARGET_RUN" --expect-commit "$COMMIT")
"$PY" "$GATE" "${gate_common[@]}" --output "$TARGET_RUN/round48_memory_gate.json"
for plant in header truncation boundary seed reset stamp; do
  if "$PY" "$GATE" "${gate_common[@]}" --plant "$plant" \
       >"$TARGET_RUN/round48_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 69
  fi
done
(
  cd "$TARGET_RUN"
  sha256sum oracle_*.bin "$FINAL_RESTART" mesh_mask.nc round48_*.json \
    round48_*_plant.log >round48_outputs.sha256
)
printf 'ROUND48_GYRE_BT_MEMORY_READY %s\n' "$TARGET_RUN"
