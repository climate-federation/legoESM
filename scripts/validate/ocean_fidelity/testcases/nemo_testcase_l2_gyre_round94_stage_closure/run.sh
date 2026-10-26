#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED NEMO ACQUISITION ONLY. Creates a new configuration/run and
# never modifies canonical NEMO src/ or any existing record directory.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R75ADV3
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R94STGCLS
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round75/oracle_advmean_kt2
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round94/oracle_stage_closure
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly RECORD=oracle_tracer_transport_kt00000002_s3.bin

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--plant-layout) ;;
  *) printf 'usage: %s [--run|--preflight-only|--plant-layout]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly PATCH=$here/stprk3_stg_round94.patch
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly STAGE_GATE=$here/../nemo_testcase_l2_gyre_round46_kt2_stage_gate.py
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$PATCH" "$ADMISSION" "$STAGE_GATE"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 64; }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]]
[[ -f "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" ]]
[[ -f "$SOURCE_RUN/$FINAL_RESTART" && -f "$SOURCE_RUN/mesh_mask.nc" ]]

for pattern in 'number of the last time step.*nn_itend *= *10' \
  'Assimilation cycle.*nn_no *= *0' 'ln_tile *= *F' \
  'ln_dynadv_vec *= *T' 'ocean time step.*rn_Dt *= *14400\.0+' \
  'tide_init : tidal components not used \(ln_tide = F\)' \
  'bdy_init : open boundaries not used \(ln_bdy = F\)'; do
  grep -Eq "$pattern" "$SOURCE_RUN/ocean.output" || {
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2
    exit 65
  }
done

dry=$(mktemp -d /tmp/gyre-r94-source.XXXXXXXX)
cp "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90"
patch -s "$dry/stprk3_stg.F90" <"$PATCH"
[[ -z "$(diff "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90" | grep '^<' || true)" ]] || {
  printf 'REFUSE: patch removes or replaces a source-card line\n' >&2
  exit 66
}

check_layout() {
  local source=$1
  [[ "$(grep -c "l1_magic = 'NEMO_L2_TRTRP_1'" "$source")" -eq 2 ]] &&
  [[ "$(grep -c 'kstp == nit000 + 1 .AND. kstg == 3' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'ROUND94_STAGE_CLOSURE_DUMP' "$source")" -eq 1 ]] &&
  [[ "$(grep -c 'WRITE(l1_unit) zFu, zFv, zFw' "$source")" -eq 4 ]]
}
check_layout "$dry/stprk3_stg.F90"
readonly EXPECTED_SIZE=$((16 + 12 * 4 + 3 * 36 * 26 * 31 * 8))
readonly READER_SIZE=$((16 + 12 * 4 + 3 * 36 * 26 * 31 * 8))
[[ "$EXPECTED_SIZE" -eq 696448 && "$READER_SIZE" -eq "$EXPECTED_SIZE" ]] || {
  printf 'REFUSE: writer/reader byte layouts disagree\n' >&2
  exit 66
}

syntax=$(mktemp -d /tmp/gyre-r94-syntax.XXXXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/stprk3_stg.F90" -o "$syntax/stprk3_stg.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/stprk3_stg.f90"

if [[ "$MODE" == --plant-layout ]]; then
  planted=$dry/stprk3_stg.planted.F90
  cp "$dry/stprk3_stg.F90" "$planted"
  sed -i '0,/WRITE(l1_unit) zFu, zFv, zFw/s//WRITE(l1_unit) zFu, zFv/' "$planted"
  if check_layout "$planted"; then
    printf 'REFUSE: layout plant stayed green\n' >&2
  else
    printf 'REFUSE: layout plant removed a transport payload field\n' >&2
  fi
  exit 69
fi

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND94_STAGE_CLOSURE_PREFLIGHT_READY %s\n' "$SOURCE_RUN"
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
}
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2
    exit 67
  }
done

manifest=$(mktemp -d /tmp/gyre-r94-manifest.XXXXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$PATCH" \
  "$ADMISSION" "$STAGE_GATE" "$SOURCE_RUN/ocean.output" \
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
patch -s "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" <"$PATCH"
cmp "$SOURCE_ROOT/EXP00/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED=$TARGET_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90
[[ -x "$BINARY" ]]
check_layout "$COMPILED"
grep -Fq 'CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )' "$COMPILED"
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2
  exit 68
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

"$PY" -c "import sys; sys.path.insert(0,'$here/..'); from nemo_testcase_l2_gyre_round46_kt2_stage_gate import read_transport; r=read_transport(__import__('pathlib').Path('$TARGET_RUN/$RECORD'),3,expected_kt=2,expected_slots=(3,2,1,1)); assert r['stage']==3"
"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$RECORD" --output "$TARGET_RUN/round94_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$RECORD" --plant-consumed \
     --output "$TARGET_RUN/round94_admission_plant.json" \
     >"$TARGET_RUN/round94_admission_plant.log" 2>&1; then
  printf 'REFUSE: admission consumed-field plant stayed green\n' >&2
  exit 69
fi
(
  cd "$TARGET_RUN"
  sha256sum "$RECORD" "$RECORD.stamp" "$FINAL_RESTART" mesh_mask.nc \
    round94_admission.json round94_admission_plant.json \
    round94_admission_plant.log >round94_outputs.sha256
)
printf 'ROUND94_STAGE_CLOSURE_READY %s\n' "$TARGET_RUN"
