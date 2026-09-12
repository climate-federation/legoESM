#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED ACQUISITION ONLY. This script invokes makenemo/mpirun.
# It creates a new configuration and never modifies canonical NEMO src/.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R64KRHS
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R71FCTST2
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round64/oracle_krhs_split
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round71/oracle_fct_stage2
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly STG_PATCH=$here/stprk3_stg_round71.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round71_fct_stage2_gate.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round71.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly RECORD1=oracle_rktracer_operands_kt00000002_s1.bin
readonly RECORD2=oracle_rktracer_operands_kt00000002_s2.bin

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63;
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$STG_PATCH" "$GATE" "$ADMISSION" "$PREREG"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 64; }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]]
[[ -f "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" ]]
[[ -f "$SOURCE_RUN/$FINAL_RESTART" && -f "$SOURCE_RUN/mesh_mask.nc" ]]
[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
}

for pattern in 'number of the last time step.*nn_itend *= *10' 'Assimilation cycle.*nn_no *= *0' 'ln_tile *= *F' 'ln_traqsr *= *T' 'ln_bdy *= *F' 'ln_traadv_fct *= *T' 'nn_fct_h *= *2' 'nn_fct_v *= *2' 'nn_fct_imp *= *1' 'ln_zad_Aimp *= *F' 'ln_traldf_msc *= *F' 'ln_trabbc *= *F' 'ln_trabbl *= *F' 'ln_tradmp *= *F' 'ln_zdfmfc *= *F' 'ln_zdfosm *= *F' 'ln_zdfnpc *= *F'; do
  grep -Eq "$pattern" "$SOURCE_RUN/ocean.output" || {
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2
    exit 65
  }
done

# Dry-apply the source card on the exact R64 source. The four replaced lines
# are inherited WRITE-only predicates; no compiled physical statement moves.
dry=$(mktemp -d /tmp/gyre-r71-source.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90"
patch -s "$dry/stprk3_stg.F90" <"$STG_PATCH"
parent_delta=$(diff "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90" | grep '^<' || true)
[[ "$(printf '%s\n' "$parent_delta" | grep -c '^<')" -eq 4 ]]
[[ -z "$(printf '%s\n' "$parent_delta" | grep -Ev 'kstp == nit000.*kstg <= 2' || true)" ]] || {
  printf 'REFUSE: patch replaces a non-writer parent line\n' >&2; exit 66;
}
[[ "$(grep -c 'kstp <= nit000 + 1.*kstg <= 2' "$dry/stprk3_stg.F90")" -eq 4 ]]

# Reproduce FCM preprocessing and prove the patched source parses in gfortran.
syntax=$(mktemp -d /tmp/gyre-r71-syntax.XXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/stprk3_stg.F90" -o "$syntax/stprk3_stg.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/stprk3_stg.f90"

for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2; exit 67;
  }
done

manifest=$(mktemp -d /tmp/gyre-r71-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$STG_PATCH" \
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
patch -s "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" <"$STG_PATCH"
cmp "$SOURCE_ROOT/EXP00/namelist_cfg" "$TARGET_ROOT/EXP00/namelist_cfg"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]]
[[ "$(grep -c 'kstp <= nit000 + 1.*kstg <= 2' "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90")" -eq 4 ]]
grep -q 'ts(ji,jj,jk,jn,Kaa) = (        ( 1._wp + r3t(ji,jj,Kbb) )' \
  "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90"
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
  [[ -s "$RECORD1" && -s "$RECORD2" ]]
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  for record in "$RECORD1" "$RECORD2"; do
    digest=$(sha256sum "$record" | awk '{print $1}')
    printf '%s %s %s\n' "$digest" "$COMMIT" "$record" >"$record.stamp"
  done
)

r71_gate() {
  "$PY" "$GATE" --root "$TARGET_RUN" --expect-commit "$COMMIT" "$@"
}
r71_gate --output "$TARGET_RUN/round71_fct_stage2_validation.json"
for plant in stamp bridge-ulp truncation; do
  if r71_gate --plant "$plant" >"$TARGET_RUN/round71_$plant-plant.log" 2>&1; then
    printf 'REFUSE: round71 %s plant stayed green\n' "$plant" >&2; exit 69
  fi
done

"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$RECORD1" "$RECORD2" \
  --output "$TARGET_RUN/round71_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$RECORD1" "$RECORD2" --plant-consumed \
     --output "$TARGET_RUN/round71_admission_plant.json" \
     >"$TARGET_RUN/round71_admission_plant.log" 2>&1; then
  printf 'REFUSE: round71 admission plant stayed green\n' >&2; exit 69
fi

(
  cd "$TARGET_RUN"
  sha256sum "$RECORD1" "$RECORD2" "$RECORD1.stamp" "$RECORD2.stamp" \
    "$FINAL_RESTART" mesh_mask.nc round71_*.json round71_*-plant.log \
    round71_admission_plant.log >round71_outputs.sha256
)
printf 'ROUND71_FCT_STAGE2_READY %s\n' "$TARGET_RUN"
