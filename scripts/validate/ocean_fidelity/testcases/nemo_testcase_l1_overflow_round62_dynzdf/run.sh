#!/usr/bin/env bash
set -euo pipefail

# OPERATOR-EXECUTED ONLY. Sandbox PMIx socket creation is forbidden.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=OVERFLOW_OMIP_L1_P3_R50PAIR
readonly TARGET_CFG=OVERFLOW_OMIP_L1_P3_R62ZDF
readonly SOURCE_ROOT=$NEMO_ROOT/tests/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/tests/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round50/acquisition/oracle_overflow_kt3_pair
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round62/acquisition/oracle_overflow_dynzdf_internals
readonly FINAL_RESTART=OVERFLOW_OMIP_L1_ZPS_00000010_restart.nc
readonly SOURCE_BINARY_SHA=6088587f198291cd9fd9c37cb0635d6218e0116ad8c3c1eaa11e60e3c06053de
readonly NEW_RECORD=oracle_r62_dynzdf_kt00000003_s3.bin
readonly PARENT_RECORD=oracle_r50_momentum_kt00000003_s3.bin
readonly R50_RECORDS=(
  oracle_r50_momentum_kt00000003_s1.bin
  oracle_r50_momentum_kt00000003_s2.bin
  oracle_r50_momentum_kt00000003_s3.bin
  oracle_r50_tracer_kt00000003_s1.bin
  oracle_r50_tracer_kt00000003_s2.bin
  oracle_r50_tracer_kt00000003_s3.bin
)

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly WRITER_REL=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_overflow_round62_dynzdf/l1_r62_dynzdf.F90
readonly PATCH_REL=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_overflow_round62_dynzdf/dynzdf_round62.patch
readonly GATE_REL=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_overflow_round62_dynzdf_gate.py
readonly ADMISSION_REL=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round21_admission.py
readonly R50_GATE_REL=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_overflow_round50_pair_gate.py
readonly PREREG_REL=docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round62.md
readonly WRITER=$REPO/$WRITER_REL
readonly PATCH=$REPO/$PATCH_REL
readonly GATE=$REPO/$GATE_REL
readonly ADMISSION=$REPO/$ADMISSION_REL
readonly R50_GATE=$REPO/$R50_GATE_REL
readonly PREREG=$REPO/$PREREG_REL
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

if [[ ${1:-} != --run || $# -ne 1 ]]; then
  printf 'REFUSE: usage is %s --run\n' "$0" >&2
  exit 63
fi
cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed producer tree\n' >&2; exit 64;
}
readonly COMMIT=$(git rev-parse HEAD)
for path in "$SOURCE_ROOT/EXP00" "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" \
  "$SOURCE_ROOT/MY_SRC/l1_r50_pair.F90" "$SOURCE_RUN" "$WRITER" "$PATCH" \
  "$GATE" "$ADMISSION" "$R50_GATE" "$PREREG"; do
  [[ -e "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 65; }
done
[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: new target exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 66
}
printf '%s  %s\n' "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo.exe" | sha256sum -c -
for pattern in \
  'number of the first time step.*nn_it000 *= *1' \
  'number of the last time step.*nn_itend *= *10' \
  'Tiling .*ln_tile *= *F' \
  'Vector form: 2nd order centered scheme.*ln_dynadv_vec *= *F' \
  'Courant number targeted application.*ln_zad_Aimp *= *T' \
  'constant vertical mixing coefficient.*ln_zdfcst *= *T' \
  'free-slip.*ln_drg_OFF *= *T' \
  'implicit friction.*ln_drgimp *= *T' \
  'Free surface with time splitting.*ln_dynspg_ts *= *T'; do
  grep -Eq "$pattern" "$SOURCE_RUN/ocean.output" || {
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2; exit 67;
  }
done

export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src:$REPO/scripts/validate/ocean_fidelity/testcases
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" "$GATE" --preflight

dry=$(mktemp -d /tmp/orca2-r62-source.XXXXXX)
cp "$NEMO_ROOT/src/OCE/DYN/dynzdf.F90" "$dry/dynzdf.F90"
patch -s "$dry/dynzdf.F90" <"$PATCH"
if [[ -n "$(diff -u "$NEMO_ROOT/src/OCE/DYN/dynzdf.F90" "$dry/dynzdf.F90" | awk '/^---|^\+\+\+/{next} /^-/{print}')" ]]; then
  printf 'REFUSE: writer patch removes or replaces a source line\n' >&2; exit 68
fi
syntax=$(mktemp -d /tmp/orca2-r62-syntax.XXXXXX)
preprocess() {
  cpp -Dkey_qco -Dkey_vco_3d -Dkey_RK3 -P -traditional \
    -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$1" -o "$2"
}
preprocess "$WRITER" "$syntax/l1_r62_dynzdf.f90"
preprocess "$dry/dynzdf.F90" "$syntax/dynzdf.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/l1_r62_dynzdf.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" \
  -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/dynzdf.f90"

mkdir -p "$(dirname "$TARGET_RUN")"
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 2097152 ]] || {
    printf 'REFUSE: %s has under 2 GB free\n' "$mount" >&2; exit 69;
  }
done

manifest=$(mktemp -d /tmp/orca2-r62-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$WRITER" "$PATCH" "$GATE" "$ADMISSION" "$R50_GATE" "$PREREG" \
  "$SOURCE_RUN/ocean.output" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -a OVERFLOW -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
cp -r "$SOURCE_ROOT/EXP00/." "$TARGET_ROOT/EXP00/"
cp -r "$SOURCE_ROOT/MY_SRC/." "$TARGET_ROOT/MY_SRC/"
cp "$SOURCE_ROOT/cpp_${SOURCE_CFG}.fcm" "$TARGET_ROOT/cpp_${TARGET_CFG}.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l1_r62_dynzdf.F90"
cp "$NEMO_ROOT/src/OCE/DYN/dynzdf.F90" "$TARGET_ROOT/MY_SRC/dynzdf.F90"
patch -s "$TARGET_ROOT/MY_SRC/dynzdf.F90" <"$PATCH"
cmp "$SOURCE_ROOT/EXP00/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg"
cmp "$SOURCE_ROOT/cpp_${SOURCE_CFG}.fcm" "$TARGET_ROOT/cpp_${TARGET_CFG}.fcm"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]]
for check in \
  'l1_r62_dynzdf.f90:NEMO_L1_R62ZDF01' \
  'dynzdf.f90:CALL r62_zdf_row(1' \
  'dynzdf.f90:CALL r62_zdf_row(2' \
  'dynzdf.f90:CALL r62_zdf_row(3' \
  'dynzdf.f90:CALL r62_zdf_u_solve' \
  'dynzdf.f90:CALL r62_zdf_v_solve'; do
  file=${check%%:*}; needle=${check#*:}
  grep -Fq "$needle" "$TARGET_ROOT/BLD/ppsrc/nemo/$file" || {
    printf 'REFUSE: compiled writer check failed: %s\n' "$check" >&2; exit 70;
  }
done
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2; exit 70
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
for name in namelist_cfg namelist_ref context_nemo.xml file_def_nemo-oce.xml iodef.xml; do
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
  cmp "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
cp "$BINARY" "$TARGET_RUN/nemo.exe"
cp "$manifest"/* "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo.exe 2>&1 | tee run.user.stdout.log ; } 2>>run.user.time.log
  test "${PIPESTATUS[0]}" -eq 0
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  for record in "${R50_RECORDS[@]}" "$NEW_RECORD"; do
    [[ -s "$record" ]] || { printf 'REFUSE: missing stream %s\n' "$record" >&2; exit 71; }
    digest=$(sha256sum "$record" | awk '{print $1}')
    printf '%s %s %s\n' "$digest" "$COMMIT" "$record" >"$record.stamp"
  done
)

"$PY" "$R50_GATE" --record-dir "$TARGET_RUN" \
  --producer-commit "$TARGET_RUN/producer_commit.txt" --expect-commit "$COMMIT" \
  --output "$TARGET_RUN/round62_parent_admission.json"
"$PY" "$GATE" --record "$TARGET_RUN/$NEW_RECORD" \
  --parent-record "$TARGET_RUN/$PARENT_RECORD" \
  --producer-commit "$TARGET_RUN/producer_commit.txt" --expect-commit "$COMMIT" \
  --output "$TARGET_RUN/round62_record_admission.json"
for plant in payload stamp; do
  if "$PY" "$GATE" --record "$TARGET_RUN/$NEW_RECORD" \
      --parent-record "$TARGET_RUN/$PARENT_RECORD" \
      --producer-commit "$TARGET_RUN/producer_commit.txt" --expect-commit "$COMMIT" \
      --plant "$plant" >"$TARGET_RUN/round62_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: round62 %s plant stayed green\n' "$plant" >&2; exit 72
  fi
done

"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$NEW_RECORD" --output "$TARGET_RUN/round62_inherited_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
    --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
    --allowed-new "$NEW_RECORD" --plant-consumed \
    >"$TARGET_RUN/round62_inherited_admission_plant.log" 2>&1; then
  printf 'REFUSE: inherited-record admission plant stayed green\n' >&2; exit 73
fi
(
  cd "$TARGET_RUN"
  sha256sum "${R50_RECORDS[@]}" "${R50_RECORDS[@]/%/.stamp}" \
    "$NEW_RECORD" "$NEW_RECORD.stamp" round62_*admission*.json \
    round62_*plant.log "$FINAL_RESTART" mesh_mask.nc >round62_outputs.sha256
)
printf 'ORCA2_ROUND62_DYNZDF_INTERNALS_READY %s\n' "$TARGET_RUN"
