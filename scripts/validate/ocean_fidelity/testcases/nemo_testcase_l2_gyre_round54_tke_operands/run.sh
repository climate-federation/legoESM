#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED ACQUISITION ONLY.  The agent must not run this script: it
# invokes makenemo and mpirun, solely to obtain the first unresolved kt=2 TKE
# statement after the existing round-54 ZDF causal gate localized the tracer
# owner to the effective stable-branch avt coefficient.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R46KT2
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R59TKE
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/oracle_kt2_stage
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round59/oracle_tke_operands
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly PATCH=$here/zdftke_round54.patch
readonly ZDFPHY_PATCH=$here/zdfphy_round55.patch
readonly WRITER=$here/l2_r54_tke.F90
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly GATE=$here/../nemo_testcase_l2_gyre_round54_tke_operands.py
readonly PY=/home/dbalwada/legoESM/.venv/bin/python

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63;
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$PATCH" "$ZDFPHY_PATCH" "$WRITER" "$ADMISSION" "$GATE"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 64; }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]]
[[ -f "$SOURCE_ROOT/MY_SRC/zdftke.F90" ]]
[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2; exit 64;
}
grep -Eq 'nn_itend[[:space:]]*=[[:space:]]*([2-9]|[1-9][0-9]+)' "$SOURCE_ROOT/EXP00/namelist_cfg"
# namzdf_tke is read from namelist_ref and then selectively overridden by
# namelist_cfg (compiled zdftke.f90:734-738).  Gate the resolved run, not the
# case file: GYRE inherits nn_pdl=1 and overrides only nn_etau=0.
resolved=$(sed -n '/Namelist namzdf_tke/,/start from rest/p' "$SOURCE_RUN/ocean.output")
for pattern in \
  'prandl number flag.*nn_pdl.*= *1' \
  'mixing length type.*nn_mxl.*= *3' \
  'surface mixing length = F\(stress\).*ln_mxl0.*= *T' \
  'surface  mixing length minimum value.*rn_mxl0.*4\.0000000000000001E-002' \
  'test param. to add tke induced by wind.*nn_etau.*= *0' \
  'type of tke penetration profile.*nn_htau.*= *1' \
  'Langmuir cells parametrization.*ln_lc.*= *T' \
  'coef to compute vertical velocity of LC.*rn_lc.*0\.14999999999999999' \
  'langmuir & surface wave breaking under ice.*nn_eice.*= *0' \
  'minimum mixing length with your parameters rmxl_min.*9\.9999999999999985E-003' \
  'set rn_mxl0 = rmxl_min'; do
  grep -Eq "$pattern" <<<"$resolved" || {
    printf 'REFUSE: source run lacks resolved TKE row: %s\n' "$pattern" >&2; exit 65;
  }
done
grep -q 'PUBLIC   dissl' "$SOURCE_ROOT/MY_SRC/zdftke.F90" || {
  printf 'REFUSE: source is not the round-46 zdftke instrument\n' >&2; exit 65;
}

# Dry-apply and enforce WRITE-only source deltas before creating anything.
dry=$(mktemp -d /tmp/gyre-r54-tke-source.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/zdftke.F90" "$dry/zdftke.F90"
patch -s "$dry/zdftke.F90" <"$PATCH"
cp "$NEMO_ROOT/src/OCE/ZDF/zdfphy.F90" "$dry/zdfphy.F90"
patch -s "$dry/zdfphy.F90" <"$ZDFPHY_PATCH"
removed=$(diff "$NEMO_ROOT/src/OCE/ZDF/zdftke.F90" "$dry/zdftke.F90" | grep -c '^<' || true)
[[ "$removed" -eq 0 ]] || {
  printf 'REFUSE: final zdftke removes %s shipped line(s)\n' "$removed" >&2; exit 66;
}
removed_zdfphy=$(diff "$NEMO_ROOT/src/OCE/ZDF/zdfphy.F90" "$dry/zdfphy.F90" | grep -c '^<' || true)
[[ "$removed_zdfphy" -eq 0 ]] || {
  printf 'REFUSE: final zdfphy removes %s shipped line(s)\n' "$removed_zdfphy" >&2; exit 66;
}
grep -q 'CALL r54_tke_begin' "$dry/zdftke.F90"
grep -q 'CALL r54_tke_matrix_row' "$dry/zdftke.F90"
grep -q 'CALL r54_tke_avn_row' "$dry/zdftke.F90"
grep -q 'CALL r54_zdfphy_finish' "$dry/zdfphy.F90"

for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2; exit 67;
  }
done

manifest=$(mktemp -d /tmp/gyre-r54-tke-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$PATCH" \
  "$ZDFPHY_PATCH" "$WRITER" "$SOURCE_RUN/ocean.output" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/zdftke.f90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/zdfphy.f90" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
# Clone the SHIPPED reference configuration, as every prior acquisition did
# (round 46 pattern): `makenemo -r <work cfg>` links only MY_SRC into WORK and
# then stops with "key key_vco_1d3d is not found in WORK routines".  The
# source card's EXP00, MY_SRC and cpp keys are then copied file-by-file (no
# cfgs directory is copied; RUN_* outputs are never touched).
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
cp -r "$SOURCE_ROOT/EXP00/." "$TARGET_ROOT/EXP00/"
cp -r "$SOURCE_ROOT/MY_SRC/." "$TARGET_ROOT/MY_SRC/"
cp "$SOURCE_ROOT/cpp_${SOURCE_CFG}.fcm" "$TARGET_ROOT/cpp_${TARGET_CFG}.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l2_r54_tke.F90"
patch -s "$TARGET_ROOT/MY_SRC/zdftke.F90" <"$PATCH"
cp "$NEMO_ROOT/src/OCE/ZDF/zdfphy.F90" "$TARGET_ROOT/MY_SRC/zdfphy.F90"
patch -s "$TARGET_ROOT/MY_SRC/zdfphy.F90" <"$ZDFPHY_PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]]
grep -q 'NEMO_L2_R56TKE2' "$TARGET_ROOT/BLD/ppsrc/nemo/l2_r54_tke.f90"
grep -q 'CALL r54_tke_matrix_row' "$TARGET_ROOT/BLD/ppsrc/nemo/zdftke.f90"
grep -q 'CALL r54_zdfphy_finish' "$TARGET_ROOT/BLD/ppsrc/nemo/zdfphy.f90"
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2; exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref \
namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  for name in $PREPARED; do cmp "$SOURCE_RUN/$name" "$name"; done
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } 2>>run.user.time.log
  test "${PIPESTATUS[0]}" -eq 0
  [[ -s oracle_tke_operands_kt00000002.bin ]]
  resolved=$(sed -n '/Namelist namzdf_tke/,/start from rest/p' ocean.output)
  grep -Eq 'prandl number flag.*nn_pdl.*= *1' <<<"$resolved"
  grep -Eq 'set rn_mxl0 = rmxl_min' <<<"$resolved"
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)

"$PY" "$GATE" --record "$TARGET_RUN/oracle_tke_operands_kt00000002.bin" \
  --expect-commit "$COMMIT" --producer-commit "$TARGET_RUN/producer_commit.txt" \
  --output "$TARGET_RUN/round59_tke_validation.json"
for plant in header truncation nan config copy shape sweep prandtl stamp; do
  if "$PY" "$GATE" --record "$TARGET_RUN/oracle_tke_operands_kt00000002.bin" \
       --expect-commit "$COMMIT" --producer-commit "$TARGET_RUN/producer_commit.txt" \
       --plant "$plant" >"$TARGET_RUN/round59_tke_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: TKE %s plant stayed green\n' "$plant" >&2; exit 69
  fi
done

# Twin admission may add only the new WRITE record.  Every inherited consumed
# record plus restart/mesh must remain identical under the schema-aware gate.
"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new oracle_tke_operands_kt00000002.bin \
  --output "$TARGET_RUN/round59_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new oracle_tke_operands_kt00000002.bin --plant-consumed \
     --output "$TARGET_RUN/round59_admission_plant.json" \
     >"$TARGET_RUN/round59_admission_plant.log" 2>&1; then
  printf 'REFUSE: consumed-field admission plant stayed green\n' >&2; exit 69
fi
(
  cd "$TARGET_RUN"
  sha256sum oracle_*.bin "$FINAL_RESTART" mesh_mask.nc round59_admission*.json \
    round59_admission_plant.log round59_tke_validation.json \
    round59_tke_*_plant.log >round59_outputs.sha256
)
printf 'ROUND59_TKE_OPERANDS_READY %s\n' "$TARGET_RUN"
