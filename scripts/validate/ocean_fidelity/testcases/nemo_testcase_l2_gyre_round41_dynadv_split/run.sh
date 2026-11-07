#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED acquisition only.  The agent must not run this file: it invokes
# makenemo, mpirun, and NEMO.  It creates a new configuration and evidence
# directory and never edits NEMO's canonical src/ tree.

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R40STG3TRM
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R41ADVSP
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
readonly SOURCE_RUN=$L2/round40_oracle_stage3_terms
readonly ROUND41=$L2/round41
readonly TARGET_RUN=$ROUND41/oracle_dynadv_split
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly KT1_RECORD=oracle_trazdf_matrix_kt00000001.bin
readonly NEW_RECORD=oracle_dynadv_split_kt00000001_s3.bin

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly DYN_PATCH=$here/dynadv_round41_split.patch
readonly STG_PATCH=$here/stprk3_stg_round41_split.patch
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly GATE=$here/../nemo_testcase_l2_gyre_round41_dynadv_split.py
readonly PREREG=$here/../manifests/nemo_testcase_l2_gyre_round41_preregister.json
readonly SOURCE_DYN=$NEMO_ROOT/src/OCE/DYN/dynadv.F90
readonly SOURCE_STG=$NEMO_ROOT/cfgs/$SOURCE_CFG/MY_SRC/stprk3_stg.F90
readonly PY=/home/dbalwada/legoESM/.venv/bin/python

[[ -d "$REPO/packages/ocean" && -d "$REPO/src" ]]
cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
git merge-base --is-ancestor c9feb3205c9f "$COMMIT"

export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

source_cfg=$NEMO_ROOT/cfgs/$SOURCE_CFG
target_cfg=$NEMO_ROOT/cfgs/$TARGET_CFG
for path in "$SOURCE_DYN" "$SOURCE_STG" "$DYN_PATCH" "$STG_PATCH" \
            "$ADMISSION" "$GATE" "$PREREG"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 64; }
done
[[ -d "$SOURCE_RUN" && -d "$source_cfg/EXP00" && -d "$source_cfg/MY_SRC" ]]
[[ ! -e "$target_cfg" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target exists: %s or %s\n' "$target_cfg" "$TARGET_RUN" >&2
  exit 64
}
[[ ! -e "$source_cfg/MY_SRC/dynadv.F90" ]] || {
  printf 'REFUSE: parent already overrides dynadv.F90\n' >&2; exit 65;
}
grep -q 'NEMO_L2_RKTS3_1' "$SOURCE_STG" || {
  printf 'REFUSE: parent is not the admitted round-40 stage-3 instrument\n' >&2
  exit 65
}
grep -q 'Surface wave (forced or coupled).*ln_wave.*=  F' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: ln_wave is not resolved false; wsd_effective contract invalid\n' >&2
  exit 65
}
grep -q 'Vector form: 2nd order centered scheme.*ln_dynadv_vec.*=  T' "$SOURCE_RUN/ocean.output"
grep -q 'nn_dynkeg.*=            0' "$SOURCE_RUN/ocean.output"

# Verify the result, not merely the patch text: both deltas apply and remove
# zero lines from their exact parents.
dry=$(mktemp -d /tmp/gyre-r41-advsp-dry.XXXXXX)
cp "$SOURCE_DYN" "$dry/dynadv.F90"
cp "$SOURCE_STG" "$dry/stprk3_stg.F90"
patch -s "$dry/dynadv.F90" <"$DYN_PATCH"
patch -s "$dry/stprk3_stg.F90" <"$STG_PATCH"
removed_dyn=$(diff "$SOURCE_DYN" "$dry/dynadv.F90" | grep -c '^<' || true)
removed_stg=$(diff "$SOURCE_STG" "$dry/stprk3_stg.F90" | grep -c '^<' || true)
[[ "$removed_dyn" -eq 0 && "$removed_stg" -eq 0 ]] || {
  printf 'REFUSE: instrument removes NEMO lines: dynadv=%s stprk3=%s\n' \
    "$removed_dyn" "$removed_stg" >&2
  exit 66
}

for mount in /tmp "$ROUND41" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 2097152 ]] || {
    printf 'REFUSE: %s has under 2 GB free\n' "$mount" >&2; exit 67;
  }
done

work_manifest=$(mktemp -d /tmp/gyre-r41-advsp-source.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$work_manifest"
(
  cd "$source_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$SOURCE_DYN" "$SOURCE_STG" \
  "$DYN_PATCH" "$STG_PATCH" "$PREREG" "$GATE" \
  >"$work_manifest/toolchain.sha256"
printf '%s\n' "$COMMIT" >"$work_manifest/producer_commit.txt"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath
cp -r "$source_cfg/EXP00/." "$target_cfg/EXP00/"
cp -r "$source_cfg/MY_SRC/." "$target_cfg/MY_SRC/"
cp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
(
  cd "$target_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/copied_cfg_before_patch.sha256"
cmp "$work_manifest/source_cfg.sha256" "$work_manifest/copied_cfg_before_patch.sha256"
cp "$SOURCE_DYN" "$target_cfg/MY_SRC/dynadv.F90"
patch -s "$target_cfg/MY_SRC/dynadv.F90" <"$DYN_PATCH"
patch -s "$target_cfg/MY_SRC/stprk3_stg.F90" <"$STG_PATCH"
touch "$target_cfg/MY_SRC/"*.F90
sha256sum "$target_cfg/MY_SRC/dynadv.F90" \
  "$target_cfg/MY_SRC/stprk3_stg.F90" "$DYN_PATCH" "$STG_PATCH" \
  >"$work_manifest/round41_instrument.sha256"

./makenemo -n "$TARGET_CFG" -m conda-scalarmath
target_binary=$target_cfg/BLD/bin/nemo.exe
[[ -x "$target_binary" ]]
grep -q 'NEMO_L2_ADVSP_1' "$target_cfg/BLD/ppsrc/nemo/dynadv.f90" || {
  printf 'REFUSE: split writer absent from compiled ppsrc\n' >&2; exit 68;
}
grep -q 'dyn_adv_round41_stage' "$target_cfg/BLD/ppsrc/nemo/stprk3_stg.f90" || {
  printf 'REFUSE: stage arm absent from compiled ppsrc\n' >&2; exit 68;
}
grep -q 'wsd_effective' "$target_cfg/BLD/ppsrc/nemo/dynadv.f90"
if nm -D "$target_binary" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in %s\n' "$target_binary" >&2
  exit 68
fi
sha256sum "$target_binary" >"$work_manifest/binary.sha256"

mkdir -p "$ROUND41"
mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref \
namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
cp "$target_binary" "$TARGET_RUN/nemo"
cp "$work_manifest"/* "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  for name in $PREPARED; do cmp "$SOURCE_RUN/$name" "$name"; done
  sha256sum $PREPARED >prepared_files.sha256
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } \
    2>>run.user.time.log
  test "${PIPESTATUS[0]}" -eq 0
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>run.user.time.log
)

# Raw identity is reported for every inherited record.  Known whole-array halo
# bytes can be indeterminate, so differences fall through to the consumed-field
# admission.  The deterministic kt=1 tracer matrix must remain byte-identical.
: >"$TARGET_RUN/round41_raw_twin_cmp.log"
for source_record in "$SOURCE_RUN"/oracle_*.bin; do
  name=${source_record##*/}
  [[ -e "$TARGET_RUN/$name" ]] || {
    printf 'REFUSE: inherited record missing: %s\n' "$name" >&2; exit 69;
  }
  if cmp -s "$source_record" "$TARGET_RUN/$name"; then
    printf 'RAW_IDENTICAL %s\n' "$name"
  else
    printf 'RAW_DIFFERS   %s (consumed-field admission follows)\n' "$name"
  fi
done | tee "$TARGET_RUN/round41_raw_twin_cmp.log"
grep -q "^RAW_IDENTICAL $KT1_RECORD\$" "$TARGET_RUN/round41_raw_twin_cmp.log" || {
  printf 'REFUSE: deterministic kt=1 tracer matrix moved\n' >&2; exit 69;
}

"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$NEW_RECORD" --output "$TARGET_RUN/round41_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$NEW_RECORD" --plant-consumed \
     --output "$TARGET_RUN/round41_admission_plant.json" \
     >"$TARGET_RUN/round41_admission_plant.log" 2>&1; then
  printf 'REFUSE: consumed-field admission plant stayed green\n' >&2; exit 70
fi
grep -q '"plant_applied": true' "$TARGET_RUN/round41_admission_plant.json"

[[ -s "$TARGET_RUN/$NEW_RECORD" ]] || {
  printf 'REFUSE: split record absent or empty\n' >&2; exit 71;
}
"$PY" "$GATE" --record "$TARGET_RUN/$NEW_RECORD" \
  --round40-root "$SOURCE_RUN" --expect-commit "$COMMIT" --validate-only \
  --output "$TARGET_RUN/round41_record_validation.json"
for plant in header calibration closure stamp; do
  if "$PY" "$GATE" --record "$TARGET_RUN/$NEW_RECORD" \
       --round40-root "$SOURCE_RUN" --expect-commit "$COMMIT" --validate-only \
       --plant "$plant" >"$TARGET_RUN/round41_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 72
  fi
done

(
  cd "$TARGET_RUN"
  sha256sum oracle_*.bin "$FINAL_RESTART" mesh_mask.nc \
    round41_admission.json round41_admission_plant.json \
    round41_record_validation.json round41_raw_twin_cmp.log \
    round41_*_plant.log >round41_outputs.sha256
)
printf 'ROUND41_GYRE_DYNADV_SPLIT_READY %s\n' "$TARGET_RUN"
