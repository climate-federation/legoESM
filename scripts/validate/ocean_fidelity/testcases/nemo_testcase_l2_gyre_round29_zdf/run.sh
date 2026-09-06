#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED acquisition only.  The agent must not run NEMO, makenemo or
# mpirun from its sandbox; it wrote this file and stopped.
#
# WHAT THIS ACQUIRES.  GYRE's kt=1 momentum divergence is localised per stage
# by the campaign's own gate -- stage 1 2.710505431213761e-19 (AT-BAR),
# stage 2 4.740083109008864e-13, stage 3 9.481924730527598e-07 -- so
# essentially all of it enters at stage 3.  Two momentum operators run at
# stage 3 and nowhere else, dyn_ldf (stprk3_stg.F90:493, compiled ppsrc :485)
# and dyn_zdf (:523 / ppsrc :498), and NO dumped record holds a stage-3
# momentum frame of any kind: every momentum-operand writer in the reviewed
# MY_SRC is gated on kstg == 2.  This run adds two WRITE-only records that
# close that gap -- the momentum RHS entering dyn_ldf, and dyn_zdf's own
# matrix, operands and solve.
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R29ZDF
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round19_oracle_v2_external
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round29_oracle_v2_zdf_matrix
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly ZDF_PATCH=$here/dynzdf_round29.patch
readonly STG_PATCH=$here/stprk3_stg_round29.patch
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly GATE=$here/../nemo_testcase_l2_gyre_round29_zdf_matrix.py

source_cfg=$NEMO_ROOT/cfgs/$SOURCE_CFG
target_cfg=$NEMO_ROOT/cfgs/$TARGET_CFG
[[ -d "$source_cfg/MY_SRC" && -d "$source_cfg/EXP00" ]]
[[ -f "$ZDF_PATCH" && -f "$STG_PATCH" ]]
if [[ -e "$target_cfg" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: target already exists: %s or %s\n' "$target_cfg" "$TARGET_RUN" >&2
  exit 64
fi

work_manifest=$(mktemp -d /tmp/gyre-r29-source.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$work_manifest"
(
  cd "$source_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$source_cfg/cpp_${SOURCE_CFG}.fcm" \
  "$NEMO_ROOT/src/OCE/DYN/dynzdf.F90" >"$work_manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath
# cp -r, NOT cp -a: preserved mtimes let fcm decide a patched file is already
# built.  MY_SRC is touched below for the same reason.
cp -r "$source_cfg/EXP00/." "$target_cfg/EXP00/"
cp -r "$source_cfg/MY_SRC/." "$target_cfg/MY_SRC/"
cp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
# dynzdf has no MY_SRC copy on the reviewed deck; create it from the SHIPPED
# source.  Copying the shipped file is allowed; editing it is not.
[[ ! -e "$target_cfg/MY_SRC/dynzdf.F90" ]]
cp "$NEMO_ROOT/src/OCE/DYN/dynzdf.F90" "$target_cfg/MY_SRC/dynzdf.F90"
(
  cd "$target_cfg"
  find EXP00 MY_SRC -type f ! -name dynzdf.F90 -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/copied_cfg_before_patch.sha256"
cmp "$work_manifest/source_cfg.sha256" "$work_manifest/copied_cfg_before_patch.sha256"
cmp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
cmp "$NEMO_ROOT/src/OCE/DYN/dynzdf.F90" "$target_cfg/MY_SRC/dynzdf.F90"

patch "$target_cfg/MY_SRC/dynzdf.F90"     <"$ZDF_PATCH"
patch "$target_cfg/MY_SRC/stprk3_stg.F90" <"$STG_PATCH"
touch "$target_cfg/MY_SRC/"*.F90
sha256sum "$target_cfg/MY_SRC/dynzdf.F90" "$target_cfg/MY_SRC/stprk3_stg.F90" \
  "$ZDF_PATCH" "$STG_PATCH" >"$work_manifest/round29_instrument.sha256"

# No del_key: the instrument adds statements to two files and removes no key.
./makenemo -n "$TARGET_CFG" -m conda-scalarmath
target_binary=$target_cfg/BLD/bin/nemo.exe
[[ -x "$target_binary" ]]
# The build must have CARRIED the instrument, not reused a stale object.
grep -q 'NEMO_L2_ZDFMX_1' "$target_cfg/BLD/ppsrc/nemo/dynzdf.f90"
grep -q 'NEMO_L2_RKPLD_1' "$target_cfg/BLD/ppsrc/nemo/stprk3_stg.f90"
if nm -D "$target_binary" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in %s\n' "$target_binary" >&2
  exit 65
fi
sha256sum "$target_binary" >"$work_manifest/binary.sha256"

mkdir "$TARGET_RUN"
for name in namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref \
  namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
  iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
  field_def_nemo-oce.xml field_def_nemo-pisces.xml; do
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
cp "$target_binary" "$TARGET_RUN/nemo"
cp "$work_manifest"/*.sha256 "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  sha256sum namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref \
    namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
    iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
    field_def_nemo-oce.xml field_def_nemo-pisces.xml >prepared_files.sha256
  sha256sum -c prepared_files.sha256
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

# WRITE-only admission: every pre-existing record and the final restart must be
# byte-identical to the round-19 twin, apart from the known uninitialised zFw
# slot the round-21 classifier already names.  A single changed value in a
# parser-scored consumed field REFUTES the instrument and stops the round.
python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --allowed-new oracle_zdf_matrix_kt00000001.bin \
                oracle_rkstage3_preldf_kt00000001.bin
for name in oracle_zdf_matrix_kt00000001.bin oracle_rkstage3_preldf_kt00000001.bin; do
  test -s "$TARGET_RUN/$name"
done

# The record's own calibration: NEMO's matrix rebuilt from NEMO's operands and
# NEMO's solve replayed on NEMO's matrix, both at the bit bar.  Exits non-zero
# unless every row is AT-BAR, and --plant must exit non-zero too.
python "$GATE" --record "$TARGET_RUN/oracle_zdf_matrix_kt00000001.bin" \
  --json "$TARGET_RUN/round29_zdf_matrix_gate.json"
if python "$GATE" --record "$TARGET_RUN/oracle_zdf_matrix_kt00000001.bin" --plant \
     >"$TARGET_RUN/round29_zdf_matrix_plant.json" 2>&1; then
  printf 'REFUSE: the planted control did not turn the gate red\n' >&2
  exit 66
fi
(
  cd "$TARGET_RUN"
  sha256sum oracle_*.bin round29_zdf_matrix_gate.json \
    GYRE_OMIP_L2_P3_00000010_restart.nc >round29_outputs.sha256
)
printf 'ROUND29_GYRE_ZDF_MATRIX_ORACLE_READY %s\n' "$TARGET_RUN"
