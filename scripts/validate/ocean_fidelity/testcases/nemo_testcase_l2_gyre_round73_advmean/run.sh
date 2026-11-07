#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED ACQUISITION ONLY. This script invokes makenemo/mpirun.
# Round 75 repair: it creates a new configuration and never modifies canonical NEMO src/.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R72ZFOP
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R75ADV3
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round72/oracle_stage1_transport
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round75/oracle_advmean_kt2
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly RECORD=oracle_bt_advmean_operands_kt00000002.bin

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--plant-resolved-row|--plant-layout) ;;
  *) printf 'usage: %s [--run|--preflight-only|--plant-resolved-row|--plant-layout]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly ROUND73=$here/../nemo_testcase_l2_gyre_round73_advmean
readonly DYN_PATCH=$ROUND73/dynspg_ts_round73.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round73_advmean_gate.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round75.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63;
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$DYN_PATCH" "$GATE" "$ADMISSION" "$PREREG"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 64; }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]]
[[ -f "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" ]]
[[ -f "$SOURCE_RUN/$FINAL_RESTART" && -f "$SOURCE_RUN/mesh_mask.nc" ]]
[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
}

resolved_output=$SOURCE_RUN/ocean.output
if [[ "$MODE" == --plant-resolved-row ]]; then
  resolved_output=$(mktemp /tmp/gyre-r75-resolved-plant.XXXXXX)
  cp "$SOURCE_RUN/ocean.output" "$resolved_output"
  grep -Eq 'in iterations nn_e *= *50' "$resolved_output"
  sed -i -E 's/in iterations nn_e *= *50/in iterations nn_e = 49/' "$resolved_output"
  grep -Eq 'in iterations nn_e *= *49' "$resolved_output"
fi
for pattern in 'number of the last time step.*nn_itend *= *10' 'Assimilation cycle.*nn_no *= *0' 'ln_tile *= *F' 'ln_dynadv_vec *= *T' 'ocean time step.*rn_Dt *= *14400\.0+' 'Barotropic time steps => in seconds *= *288\.0+' 'in iterations nn_e *= *50' 'set auto \(ln_bt_auto=T\)'; do
  grep -Eq "$pattern" "$resolved_output" || {
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2
    exit 65
  }
done
[[ "$MODE" != --plant-resolved-row ]] || {
  printf 'REFUSE: resolved-row plant stayed green\n' >&2; exit 69
}

# Dry-apply the exact source card. Four inherited WRITE-only predicates widen;
# the new block only opens the already-defined transport-mean stream at kt=2.
dry=$(mktemp -d /tmp/gyre-r75-source.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$dry/dynspg_ts.F90"
patch -s "$dry/dynspg_ts.F90" <"$DYN_PATCH"
parent_delta=$(diff "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$dry/dynspg_ts.F90" | grep '^<' || true)
[[ "$(printf '%s\n' "$parent_delta" | grep -c '^<')" -eq 6 ]]
[[ -z "$(printf '%s\n' "$parent_delta" | grep -Ev 'IF.*lwp.*kt == nit000.*|WRITE.*l2_adv_unit|& l2_u_mid, l2_v_mid, zhup2_e' || true)" ]] || {
  printf 'REFUSE: patch replaces a non-writer parent line\n' >&2; exit 66;
}
grep -q 'kt == nit000 + 1' "$dry/dynspg_ts.F90"
[[ "$(grep -c 'kt <= nit000 + 1' "$dry/dynspg_ts.F90")" -eq 4 ]]
grep -q 'oracle_bt_advmean_operands_kt",I8.8' "$dry/dynspg_ts.F90"

check_layout() {
  local source=$1
  [[ "$(grep -c 'WRITE(l2_adv_unit) l2_adv_before_u, l2_adv_before_v, zhU, zhV$' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'WRITE(l2_adv_unit) l2_u_mid, l2_v_mid$' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'WRITE(l2_adv_unit) ua_e, va_e$' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'WRITE(l2_adv_unit) zhup2_e, zhvp2_e, un_adv, vn_adv$' "$source")" -eq 1 ]]
}
check_layout "$dry/dynspg_ts.F90"
readonly WRITER_SIZE=$((16 + 6 * 4 + 8 + 50 * 8 + 2 * 36 * 26 * 8 + 50 * (4 + 8 + 10 * 36 * 26 * 8) + 4 * 36 * 26 * 8))
readonly READER_SIZE=$("$PY" -c "import runpy; print(runpy.run_path('$GATE')['EXPECTED_SIZE'])")
[[ "$WRITER_SIZE" -eq 3789976 && "$READER_SIZE" -eq "$WRITER_SIZE" ]] || {
  printf 'REFUSE: writer/reader byte layouts disagree: %s != %s\n' "$WRITER_SIZE" "$READER_SIZE" >&2
  exit 66
}

# Reproduce FCM preprocessing and prove the exact patched source parses.
syntax=$(mktemp -d /tmp/gyre-r75-syntax.XXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/dynspg_ts.F90" -o "$syntax/dynspg_ts.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/dynspg_ts.f90"

if [[ "$MODE" == --plant-layout ]]; then
  planted=$dry/dynspg_ts.planted.F90
  cp "$dry/dynspg_ts.F90" "$planted"
  sed -i 's/WRITE(l2_adv_unit) ua_e, va_e/WRITE(l2_adv_unit) ua_e/' "$planted"
  if check_layout "$planted"; then
    printf 'REFUSE: layout plant stayed green\n' >&2
  else
    printf 'REFUSE: layout plant removed one kt=2 velocity field\n' >&2
  fi
  exit 69
fi

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND75_ADVMEAN_PREFLIGHT_READY %s\n' "$SOURCE_RUN"
  exit 0
fi

for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2; exit 67;
  }
done

manifest=$(mktemp -d /tmp/gyre-r75-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$DYN_PATCH" \
  "$GATE" "$ADMISSION" "$PREREG" "$SOURCE_RUN/ocean.output" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
patch -s "$TARGET_ROOT/MY_SRC/dynspg_ts.F90" <"$DYN_PATCH"
cmp "$SOURCE_ROOT/EXP00/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED=$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90
[[ -x "$BINARY" ]]
grep -q 'kt == nit000 + 1' "$COMPILED"
[[ "$(grep -c 'kt <= nit000 + 1' "$COMPILED")" -eq 4 ]]
check_layout "$COMPILED"
grep -q 'un_adv(ji,jj) = un_adv(ji,jj) + za2 \* zhU(ji,jj) \* r1_e2u(ji,jj)' "$COMPILED"
grep -q 'un_adv(:,:) = un_adv(:,:) / r1_wgt2s' "$COMPILED"
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2; exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do cp -L "$TARGET_ROOT/EXP00/$name" "$TARGET_RUN/$name"; done
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } 2>>run.user.time.log
  test "${PIPESTATUS[0]}" -eq 0
  [[ -s "$RECORD" ]]
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  digest=$(sha256sum "$RECORD" | awk '{print $1}')
  printf '%s %s %s\n' "$digest" "$COMMIT" "$RECORD" >"$RECORD.stamp"
)

r75_gate() {
  "$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" "$@"
}
r75_gate --output "$TARGET_RUN/round75_advmean_validation.json"
for plant in stamp header truncation replay-ulp; do
  if r75_gate --plant "$plant" >"$TARGET_RUN/round75_$plant-plant.log" 2>&1; then
    printf 'REFUSE: round75 %s plant stayed green\n' "$plant" >&2; exit 69
  fi
done

"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$RECORD" --output "$TARGET_RUN/round75_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$RECORD" --plant-consumed \
     --output "$TARGET_RUN/round75_admission_plant.json" \
     >"$TARGET_RUN/round75_admission_plant.log" 2>&1; then
  printf 'REFUSE: round75 admission plant stayed green\n' >&2; exit 69
fi

(
  cd "$TARGET_RUN"
  sha256sum "$RECORD" "$RECORD.stamp" "$FINAL_RESTART" mesh_mask.nc \
    round75_*.json round75_*-plant.log round75_admission_plant.log \
    >round75_outputs.sha256
)
printf 'ROUND75_ADVMEAN_READY %s\n' "$TARGET_RUN"
