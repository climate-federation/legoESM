#!/usr/bin/env bash
set -Eeuo pipefail

# USER-EXECUTED ACQUISITION ONLY.  --run invokes makenemo/mpirun; Codex may
# execute --preflight-only, which stops after the additive and syntax proofs.
refuse_on_error() {
  local status=$?
  printf 'REFUSE: round117 acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

die() {
  printf 'REFUSE: %s\n' "$1" >&2
  exit "${2:-64}"
}

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only) ;;
  *) die "usage: run.sh [--run|--preflight-only]" 62 ;;
esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R111FCTW
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R117PRELOOP
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round111/oracle_fct_writers
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round117/oracle_preloop_forcing
readonly REFERENCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round81/oracle_btstep_kt2
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly SLOW_RECORD=oracle_slow_forcing_kt00000002.bin
readonly PRELOOP_RECORD=oracle_preloop_forcing_kt00000002.bin
readonly SLOW_EXPECTED_SIZE=1486548
readonly PRELOOP_EXPECTED_SIZE=127408

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly STP_PATCH=$here/stp2d_round117.patch
readonly SPG_PATCH=$here/dynspg_ts_round117.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round117_preloop_gate.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round117.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  die "acquisition requires a clean committed tree" 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$STP_PATCH" "$SPG_PATCH" "$GATE" "$ADMISSION" "$PREREG" \
            "$SOURCE_ROOT/MY_SRC/stp2d.F90" \
            "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" \
            "$SOURCE_RUN/$FINAL_RESTART" "$SOURCE_RUN/mesh_mask.nc" \
            "$SOURCE_RUN/ocean.output" \
            "$REFERENCE_RUN/oracle_bt_step_operands_kt00000002.bin"; do
  [[ -f "$path" ]] || die "missing source-card input $path"
done
for path in "$SOURCE_ROOT/EXP00" "$SOURCE_ROOT/MY_SRC" \
            "$SOURCE_ROOT/WORK" "$SOURCE_ROOT/BLD/inc"; do
  [[ -d "$path" ]] || die "missing source-card directory $path"
done
if [[ -e "$TARGET_ROOT" || -e "$TARGET_RUN" ]]; then
  die "new target already exists: $TARGET_ROOT or $TARGET_RUN"
fi

# Freeze the exact resolved source card.  The acquisition changes no namelist.
for pattern in 'number of the last time step.*nn_itend *= *10' \
  'Assimilation cycle.*nn_no *= *0' 'ln_tile *= *F' 'ln_traqsr *= *T' \
  'ln_bdy *= *F' 'ln_traadv_fct *= *T' 'nn_fct_h *= *2' \
  'nn_fct_v *= *2' 'nn_fct_imp *= *1' 'ln_zad_Aimp *= *F' \
  'ln_traldf_msc *= *F' 'ln_trabbc *= *F' 'ln_trabbl *= *F' \
  'ln_tradmp *= *F' 'ln_zdfmfc *= *F' 'ln_zdfosm *= *F' \
  'ln_zdfnpc *= *F'; do
  grep -Eq "$pattern" "$SOURCE_RUN/ocean.output" || \
    die "source run lacks resolved row: $pattern" 65
done

# Patch exact R111 source copies with zero fuzz.  A line beginning '<' in a
# normal diff would mean that the supposedly additive instrument removed or
# replaced a source line, which is forbidden.
dry=$(mktemp -d /tmp/gyre-r117-source.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/stp2d.F90" "$dry/stp2d.F90"
cp "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$dry/dynspg_ts.F90"
patch -s --fuzz=0 "$dry/stp2d.F90" <"$STP_PATCH"
patch -s --fuzz=0 "$dry/dynspg_ts.F90" <"$SPG_PATCH"
for name in stp2d dynspg_ts; do
  if diff "$SOURCE_ROOT/MY_SRC/$name.F90" "$dry/$name.F90" | grep -q '^<'; then
    die "round117 $name patch removes or replaces an R111 source line" 66
  fi
done
for check in \
  'stp2d.F90:NEMO_L2_SLOW_2' \
  'stp2d.F90:kt == nit000 + 1' \
  'dynspg_ts.F90:NEMO_L2_R117PF1' \
  'dynspg_ts.F90:SIZE(zu_frc)' \
  'dynspg_ts.F90:CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )'; do
  file=${check%%:*}
  needle=${check#*:}
  grep -Fq "$needle" "$dry/$file" || \
    die "dry source lacks $file statement: $needle" 66
done

# Use the exact preprocessor keys and include roots of the admitted R111 build,
# then make gfortran parse both complete translation units.
syntax=$(mktemp -d /tmp/gyre-r117-syntax.XXXXXX)
preprocess() {
  cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P \
    -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$1" -o "$2"
}
preprocess "$dry/stp2d.F90" "$syntax/stp2d.f90"
preprocess "$dry/dynspg_ts.F90" "$syntax/dynspg_ts.f90"
for name in stp2d dynspg_ts; do
  "$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
    -J "$syntax" "$syntax/$name.f90"
done
printf 'ROUND117_SYNTAX_PASS stp2d.f90 dynspg_ts.f90\n'
printf 'ROUND117_LAYOUT slow=%s preloop=%s full2=%s owned2=%s\n' \
  "$SLOW_EXPECTED_SIZE" "$PRELOOP_EXPECTED_SIZE" "$((36 * 26))" "$((32 * 22))"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND117_PREFLIGHT_READY commit=%s target=%s\n' "$COMMIT" "$TARGET_CFG"
  exit 0
fi

for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || die "$mount has under 4 GB free" 67
done

manifest=$(mktemp -d /tmp/gyre-r117-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$STP_PATCH" "$SPG_PATCH" "$GATE" "$ADMISSION" "$PREREG" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/stp2d.f90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" \
  "$SOURCE_RUN/ocean.output" \
  "$REFERENCE_RUN/oracle_bt_step_operands_kt00000002.bin" \
  >"$manifest/toolchain.sha256"

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
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/stp2d.F90" <"$STP_PATCH"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/dynspg_ts.F90" <"$SPG_PATCH"
cmp "$SOURCE_ROOT/EXP00/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg" || \
  die "source card changed namelist_cfg" 68
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || die "built binary is missing or not executable" 68
for check in \
  'stp2d.f90:NEMO_L2_SLOW_2' \
  'stp2d.f90:ROUND117_SLOW_FORCING_DUMP' \
  'dynspg_ts.f90:NEMO_L2_R117PF1' \
  'dynspg_ts.f90:SIZE(zu_frc)' \
  'dynspg_ts.f90:ROUND117_PRELOOP_FORCING_DUMP' \
  'dynspg_ts.f90:CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )'; do
  file=${check%%:*}
  needle=${check#*:}
  grep -Fq "$needle" "$TARGET_ROOT/BLD/ppsrc/nemo/$file" || \
    die "compiled writer check failed: $check" 68
done
if nm -D "$BINARY" | grep -q '_ZGV'; then
  die "vector-math symbol present in acquisition binary" 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do
  [[ -e "$TARGET_ROOT/EXP00/$name" ]] || die "prepared run input is missing: $name" 68
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
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || die "NEMO acquisition process failed" 69
  grep -Fxq 'STOP 0' run.user.stdout.log || die "NEMO run lacks STOP 0" 69
  for item in "$SLOW_RECORD:$SLOW_EXPECTED_SIZE" \
              "$PRELOOP_RECORD:$PRELOOP_EXPECTED_SIZE"; do
    record=${item%%:*}
    expected=${item##*:}
    [[ -f "$record" ]] || die "NEMO did not emit $record" 69
    actual=$(stat -c %s "$record")
    [[ "$actual" -eq "$expected" ]] || \
      die "$record has $actual bytes, expected $expected" 69
    digest=$(sha256sum "$record" | awk '{print $1}')
    printf '%s %s %s\n' "$digest" "$COMMIT" "$record" >"$record.stamp"
  done
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)

r117_gate() {
  "$PY" "$GATE" --root "$TARGET_RUN" --reference-root "$REFERENCE_RUN" \
    --expect-commit "$COMMIT" "$@"
}
r117_gate --output "$TARGET_RUN/round117_preloop_validation.json"
for plant in stamp truncation header input-ulp reference-ulp; do
  if r117_gate --plant "$plant" \
       --output "$TARGET_RUN/round117_${plant}_plant.json" \
       >"$TARGET_RUN/round117_${plant}_plant.log" 2>&1; then
    die "round117 $plant plant stayed green" 70
  fi
  grep -Fq "STATUS PLANT-FIRED: $plant" \
    "$TARGET_RUN/round117_${plant}_plant.log" || \
    die "round117 $plant plant exited without its named firing" 70
done

"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$SLOW_RECORD" "$PRELOOP_RECORD" \
  --output "$TARGET_RUN/round117_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$SLOW_RECORD" "$PRELOOP_RECORD" --plant-consumed \
     --output "$TARGET_RUN/round117_admission_plant.json" \
     >"$TARGET_RUN/round117_admission_plant.log" 2>&1; then
  die "round117 admission plant stayed green" 70
fi

(
  cd "$TARGET_RUN"
  sha256sum "$SLOW_RECORD" "$PRELOOP_RECORD" \
    "$SLOW_RECORD.stamp" "$PRELOOP_RECORD.stamp" \
    "$FINAL_RESTART" mesh_mask.nc round117_preloop_validation.json \
    round117_*_plant.log round117_admission.json \
    round117_admission_plant.log >round117_outputs.sha256
)
printf 'ROUND117_PRELOOP_RECORDS_READY %s\n' "$TARGET_RUN"
