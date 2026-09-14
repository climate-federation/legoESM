#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED ACQUISITION ONLY. This script invokes makenemo/mpirun.
# It clones the admitted Round-64 card and never modifies canonical NEMO src/.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R64KRHS
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R90BARO
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round64/oracle_krhs_split
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round90/oracle_baro_correction
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly WRITER=$here/l2_r90_baro.F90
readonly STP_PATCH=$here/stprk3_stg_round90.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round90_baro_record.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l2_gyre_round90_barotropic_correction.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly RECORD=oracle_baro_correction_kt00000002_s1.bin

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63;
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$WRITER" "$STP_PATCH" "$GATE" "$ADMISSION" "$PREREG"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 64; }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]]
[[ -f "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" ]]
[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
}

# No resolved configuration changes are permitted. Prove the inherited card's
# ten-step horizon, active vector branch, and non-tiled whole-domain writer.
for pattern in 'number of the last time step.*nn_itend *= *10' 'ln_tile *= *F' 'ln_dynadv_vec *= *T' 'ln_zad_Aimp *= *F'; do
  grep -Eq "$pattern" "$SOURCE_RUN/ocean.output" || {
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2; exit 65;
  }
done

dry=$(mktemp -d /tmp/gyre-r90-source.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90"
patch -s "$dry/stprk3_stg.F90" <"$STP_PATCH"
[[ -z "$(diff "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90" | grep '^<' || true)" ]] || {
  printf 'REFUSE: Round-90 patch removes or replaces a source line\n' >&2; exit 66;
}

syntax=$(mktemp -d /tmp/gyre-r90-syntax.XXXXXX)
preprocess() {
  cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$1" -o "$2"
}
preprocess "$WRITER" "$syntax/l2_r90_baro.f90"
preprocess "$dry/stprk3_stg.F90" "$syntax/stprk3_stg.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/l2_r90_baro.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/stprk3_stg.f90"

for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2; exit 67;
  }
done

manifest=$(mktemp -d /tmp/gyre-r90-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$WRITER" "$STP_PATCH" "$GATE" "$ADMISSION" "$PREREG" "$SOURCE_RUN/ocean.output" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l2_r90_baro.F90"
patch -s "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" <"$STP_PATCH"
cmp "$SOURCE_ROOT/EXP00/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]]
for check in 'l2_r90_baro.f90:NEMO_L2_R90BARO1' 'stprk3_stg.f90:CALL r90_baro_begin' 'stprk3_stg.f90:CALL r90_baro_finish'; do
  file=${check%%:*}; needle=${check#*:}
  grep -q "$needle" "$TARGET_ROOT/BLD/ppsrc/nemo/$file" || {
    printf 'REFUSE: compiled writer check failed: %s\n' "$check" >&2; exit 68;
  }
done
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

baro_gate() {
  "$PY" "$GATE" --record "$TARGET_RUN/$RECORD" --stamp "$TARGET_RUN/$RECORD.stamp" --expect-commit "$COMMIT" "$@"
}
baro_gate --output "$TARGET_RUN/round90_baro_record.json"
if baro_gate --plant final-ulp --output "$TARGET_RUN/round90_baro_record_plant.json" >"$TARGET_RUN/round90_baro_record_plant.log" 2>&1; then
  printf 'REFUSE: Round-90 final-add plant stayed green\n' >&2; exit 69
fi

"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" --twin /nonexistent --identical mesh_mask.nc --allowed-new "$RECORD" --output "$TARGET_RUN/round90_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" --twin /nonexistent --identical mesh_mask.nc --allowed-new "$RECORD" --plant-consumed --output "$TARGET_RUN/round90_admission_plant.json" >"$TARGET_RUN/round90_admission_plant.log" 2>&1; then
  printf 'REFUSE: Round-90 admission plant stayed green\n' >&2; exit 69
fi
(
  cd "$TARGET_RUN"
  sha256sum "$RECORD" "$RECORD.stamp" mesh_mask.nc round90_*.json round90_*plant.log >round90_outputs.sha256
)
printf 'ROUND90_BARO_CORRECTION_READY %s\n' "$TARGET_RUN"
