#!/usr/bin/env bash
set -euo pipefail

# User-executed acquisition only.  Codex must not run NEMO from its sandbox.
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R21W
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round19_oracle_v2_external
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round21_oracle_v2_stage_ww
readonly PATCH_FILE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)/traadv_round21.patch
readonly ADMISSION=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)/nemo_testcase_l2_gyre_round21_admission.py

source_cfg=$NEMO_ROOT/cfgs/$SOURCE_CFG
target_cfg=$NEMO_ROOT/cfgs/$TARGET_CFG
[[ -d "$source_cfg/MY_SRC" && -d "$source_cfg/EXP00" ]]
[[ -f "$PATCH_FILE" ]]
if [[ -e "$target_cfg" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: target already exists: %s or %s\n' "$target_cfg" "$TARGET_RUN" >&2
  exit 64
fi

work_manifest=$(mktemp -d /tmp/gyre-r21-source.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$work_manifest"
(
  cd "$source_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$source_cfg/cpp_${SOURCE_CFG}.fcm" >"$work_manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath
cp -a "$source_cfg/EXP00/." "$target_cfg/EXP00/"
cp -a "$source_cfg/MY_SRC/." "$target_cfg/MY_SRC/"
cp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
(
  cd "$target_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/copied_cfg_before_patch.sha256"
sed "s|$target_cfg|$source_cfg|g" "$work_manifest/copied_cfg_before_patch.sha256" \
  >/dev/null  # paths are relative; this documents that no path rewrite is needed
cmp "$work_manifest/source_cfg.sha256" "$work_manifest/copied_cfg_before_patch.sha256"
cmp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"

patch "$target_cfg/MY_SRC/traadv.F90" <"$PATCH_FILE"
sha256sum "$target_cfg/MY_SRC/traadv.F90" "$PATCH_FILE" \
  >"$work_manifest/round21_instrument.sha256"
./makenemo -n "$TARGET_CFG" -m conda-scalarmath
target_binary=$target_cfg/BLD/bin/nemo.exe
[[ -x "$target_binary" ]]
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

# WRITE-only admission compares every parser-visible consumed field and the
# final restart.  Raw uninitialised workspace/halo bytes remain in the evidence
# but cannot veto an otherwise bit-identical instrument extension.
python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN"
for stage in 1 2 3; do
  test -s "$TARGET_RUN/oracle_rkstage_ww_kt00000001_s${stage}.bin"
done
(
  cd "$TARGET_RUN"
  sha256sum oracle_*.bin GYRE_OMIP_L2_P3_00000010_restart.nc \
    >round21_outputs.sha256
)
printf 'ROUND21_GYRE_STAGE_W_ORACLE_READY %s\n' "$TARGET_RUN"
