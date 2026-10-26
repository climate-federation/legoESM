#!/usr/bin/env bash
set -Eeuo pipefail

# USER-EXECUTED ACQUISITION ONLY. This script invokes makenemo/mpirun.
# It creates a new configuration and never modifies canonical NEMO src/.
refuse_on_error() {
  local status=$?
  printf 'REFUSE: round111 acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R64KRHS
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R111FCTW
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round64/oracle_krhs_split
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round111/oracle_fct_writers
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly RECORD=oracle_fct_writers_kt00000002_s3.bin

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly WRITER=$here/l2_r111_fct.F90
readonly ADV_PATCH=$here/traadv_fct_round111.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round111_fct_writer_gate.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round111.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$WRITER" "$ADV_PATCH" "$GATE" "$ADMISSION" "$PREREG"; do
  if [[ ! -f "$path" ]]; then
    printf 'REFUSE: missing source-card file %s\n' "$path" >&2
    exit 64
  fi
done
for path in "$SOURCE_ROOT/EXP00" "$SOURCE_ROOT/MY_SRC"; do
  if [[ ! -d "$path" ]]; then
    printf 'REFUSE: missing source-card directory %s\n' "$path" >&2
    exit 64
  fi
done
for path in "$SOURCE_ROOT/MY_SRC/traadv_fct.F90" \
            "$SOURCE_RUN/$FINAL_RESTART" "$SOURCE_RUN/mesh_mask.nc" \
            "$SOURCE_RUN/oracle_krhs_split_kt00000002.bin" \
            "$SOURCE_RUN/oracle_tke_rhs_split_kt00000002.bin"; do
  if [[ ! -f "$path" ]]; then
    printf 'REFUSE: missing source-card input %s\n' "$path" >&2
    exit 64
  fi
done
if [[ -e "$TARGET_ROOT" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: new target already exists: %s or %s\n' \
    "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
fi

for pattern in 'number of the last time step.*nn_itend *= *10' \
  'Assimilation cycle.*nn_no *= *0' 'ln_tile *= *F' 'ln_traqsr *= *T' \
  'ln_bdy *= *F' 'ln_traadv_fct *= *T' 'nn_fct_h *= *2' \
  'nn_fct_v *= *2' 'nn_fct_imp *= *1' 'ln_zad_Aimp *= *F' \
  'ln_traldf_msc *= *F' 'ln_trabbc *= *F' 'ln_trabbl *= *F' \
  'ln_tradmp *= *F' 'ln_zdfmfc *= *F' 'ln_zdfosm *= *F' \
  'ln_zdfnpc *= *F'; do
  if ! grep -Eq "$pattern" "$SOURCE_RUN/ocean.output"; then
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2
    exit 65
  fi
done

# Prove the card is additive against the exact R64 source, then parse both
# translation units with the same FCM preprocessor definitions as the build.
dry=$(mktemp -d /tmp/gyre-r111-source.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/traadv_fct.F90" "$dry/traadv_fct.F90"
patch -s --fuzz=0 "$dry/traadv_fct.F90" <"$ADV_PATCH"
if diff "$SOURCE_ROOT/MY_SRC/traadv_fct.F90" "$dry/traadv_fct.F90" | \
     grep -q '^<'; then
  printf 'REFUSE: round111 patch removes or replaces an R64 source line\n' >&2
  exit 66
fi
for needle in 'CALL r111_begin' 'CALL r111_first_flux' \
  'CALL r111_midpoint' 'CALL r111_average_flux' 'CALL r111_upstream'; do
  if ! grep -q "$needle" "$dry/traadv_fct.F90"; then
    printf 'REFUSE: dry source lacks writer call %s\n' "$needle" >&2
    exit 66
  fi
done

syntax=$(mktemp -d /tmp/gyre-r111-syntax.XXXXXX)
preprocess() {
  cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P \
    -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$1" -o "$2"
}
preprocess "$WRITER" "$syntax/l2_r111_fct.f90"
preprocess "$dry/traadv_fct.F90" "$syntax/traadv_fct.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/l2_r111_fct.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" \
  -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/traadv_fct.f90"

for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 4194304 ]]; then
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2
    exit 67
  fi
done

manifest=$(mktemp -d /tmp/gyre-r111-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$WRITER" \
  "$ADV_PATCH" "$GATE" "$ADMISSION" "$PREREG" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/traadv_fct.f90" \
  "$SOURCE_RUN/ocean.output" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) \
  -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) \
  -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l2_r111_fct.F90"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/traadv_fct.F90" <"$ADV_PATCH"
if ! cmp "$SOURCE_ROOT/EXP00/namelist_cfg" \
     "$TARGET_ROOT/EXP00/namelist_cfg"; then
  printf 'REFUSE: source card changed namelist_cfg\n' >&2
  exit 68
fi
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
if [[ ! -x "$BINARY" ]]; then
  printf 'REFUSE: built binary is missing or not executable\n' >&2
  exit 68
fi
for check in 'l2_r111_fct.f90:NEMO_L2_R111F1' \
  'traadv_fct.f90:CALL r111_begin' 'traadv_fct.f90:CALL r111_first_flux' \
  'traadv_fct.f90:CALL r111_midpoint' \
  'traadv_fct.f90:CALL r111_average_flux' \
  'traadv_fct.f90:CALL r111_upstream' \
  'traadv_fct.f90:pt_rhs(ji,jj,jk) = pt_rhs(ji,jj,jk) + ztra /'; do
  file=${check%%:*}
  needle=${check#*:}
  if ! grep -Fq "$needle" "$TARGET_ROOT/BLD/ppsrc/nemo/$file"; then
    printf 'REFUSE: compiled writer check failed: %s\n' "$check" >&2
    exit 68
  fi
done
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in acquisition binary\n' >&2
  exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do
  if [[ ! -e "$TARGET_ROOT/EXP00/$name" ]]; then
    printf 'REFUSE: prepared run input is missing: %s\n' "$name" >&2
    exit 68
  fi
  cp -L "$TARGET_ROOT/EXP00/$name" "$TARGET_RUN/$name"
done
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | \
      tee run.user.stdout.log ; } 2>>run.user.time.log
  if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
    printf 'REFUSE: NEMO acquisition process failed\n' >&2
    exit 69
  fi
  if [[ ! -s "$RECORD" ]]; then
    printf 'REFUSE: NEMO did not emit %s\n' "$RECORD" >&2
    exit 69
  fi
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  digest=$(sha256sum "$RECORD" | awk '{print $1}')
  printf '%s %s %s\n' "$digest" "$COMMIT" "$RECORD" >"$RECORD.stamp"
)

r111_gate() {
  "$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" "$@"
}
r111_gate --output "$TARGET_RUN/round111_fct_writer_validation.json"
for plant in stamp truncation rhs-entry-ulp; do
  if r111_gate --plant "$plant" \
       --output "$TARGET_RUN/round111_${plant}_plant.json" \
       >"$TARGET_RUN/round111_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: round111 %s plant stayed green\n' "$plant" >&2
    exit 70
  fi
done

"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  oracle_krhs_split_kt00000002.bin oracle_tke_rhs_split_kt00000002.bin \
  --allowed-new "$RECORD" --output "$TARGET_RUN/round111_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     oracle_krhs_split_kt00000002.bin oracle_tke_rhs_split_kt00000002.bin \
     --allowed-new "$RECORD" --plant-consumed \
     --output "$TARGET_RUN/round111_admission_plant.json" \
     >"$TARGET_RUN/round111_admission_plant.log" 2>&1; then
  printf 'REFUSE: round111 admission plant stayed green\n' >&2
  exit 70
fi

(
  cd "$TARGET_RUN"
  sha256sum "$RECORD" "$RECORD.stamp" "$FINAL_RESTART" mesh_mask.nc \
    round111_fct_writer_validation.json round111_*_plant.log \
    round111_admission.json round111_admission_plant.log \
    >round111_outputs.sha256
)
printf 'ROUND111_FCT_WRITERS_READY %s\n' "$TARGET_RUN"
