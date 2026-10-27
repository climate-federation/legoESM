#!/usr/bin/env bash
set -Eeuo pipefail

# Operator-only scalar-libm acquisition for the unresolved second-step owner.
# Preflight is read-only; only --run builds NEMO and launches the single rank.
do_run=0
finalize_existing=0
[[ $# -eq 0 ]] || [[ $# -eq 1 && $1 == --run ]] || \
  [[ $# -eq 1 && $1 == --finalize-existing ]] || {
  printf 'REFUSE_USAGE: expected [--run|--finalize-existing]\n' >&2; exit 64; }
if [[ $# -eq 1 && $1 == --run ]]; then do_run=1; fi
if [[ $# -eq 1 && $1 == --finalize-existing ]]; then finalize_existing=1; fi

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) || {
  printf 'REFUSE_NOT_WORKTREE\n' >&2; exit 65; }
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_NAME=GYRE_OMIP_L2_P3_SMW
readonly TARGET_NAME=GYRE_OMIP_L2_P3_SMW_R7STEP2
readonly SOURCE_CFG=$NEMO_ROOT/cfgs/$SOURCE_NAME
readonly TARGET_CFG=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly BASE_EXP00=$NEMO_ROOT/cfgs/GYRE_OMIP_L2_P3/EXP00
readonly PATCH=$here/gyre_p3_scalar_step2_writer.patch
readonly ZDF_BASE_PATCH=$here/../nemo_testcase_l2_gyre_round35_trazdf/trazdf_round35.patch
readonly ZDF_KT2_PATCH=$here/../nemo_testcase_l2_gyre_round38_trazdf_kt2/trazdf_round38_kt2.patch
readonly DECK_PATCH=$here/namelist_cfg_r4_kt10.patch
readonly BASE_DECK=$here/../configs/gyre_omip_l2_phase3_namelist_cfg
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round7/oracle_gyre_r4_scalar_step2}
readonly SOURCE_STP_SHA=11c6a66407eda69f3c8ceb57a37ed5f6565cefe4d8e32c0dee453ad021906fee
readonly SOURCE_STG_SHA=4d3493de0e837654e961a30ac4684801d87b4226ed9e8c226922db780d699afe
readonly PATCHED_STP_SHA=449390fe70a29298b5a1a98c45f74d0d3a927722e0573c02fcc3834c0c7f1dbe
readonly PATCHED_STG_SHA=43d5cf571b94165599cb1e681fd88827079770c4007b3bef0e87f934c8f149e3
readonly SOURCE_ZDF_SHA=cd7c446164d2813bb278130ba29cfaa103b736815d743b39c5748d4ccc7917ec
readonly PATCHED_ZDF_SHA=6f28b8ec93daa40903bcc9e1800a619de52a8d21600059c375749dc59d7deaaa
readonly SCALAR_TWIN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round6/operator_scalarmath_r4_stage
readonly IDENTITY_GATE=$here/../nemo_testcase_l2_gyre_hires_ladder_gate.py
readonly PYTHON=/home/dbalwada/legoESM/.venv/bin/python
readonly LANE_PYTHONPATH=$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src

finalize_evidence() {
  for record in \
    oracle_stage_kt00000002_s1.bin oracle_stage_kt00000002_s2.bin \
    oracle_stage_kt00000002_s3.bin oracle_zdf_entry_kt00000002.bin \
    oracle_rktracer_operands_kt00000002_s1.bin \
    oracle_rktracer_operands_kt00000002_s2.bin \
    oracle_tracer_transport_kt00000002_s3.bin \
    oracle_rktracer_stage3_kt00000002.bin oracle_qsr_stage3_kt00000002.bin \
    oracle_trazdf_matrix_kt00000001.bin \
    oracle_trazdf_matrix_kt00000002.bin; do
    [[ -s $EVIDENCE/run/$record ]] || {
      printf 'REFUSE_MISSING_RECORD: %s\n' "$record" >&2; return 70; }
  done
  mkdir -p "$EVIDENCE/manifest"
  env PYTHONPATH="$LANE_PYTHONPATH" "$PYTHON" "$IDENTITY_GATE" \
    --writer-identity-only \
    --writer-twin-root "$SCALAR_TWIN" \
    --writer-candidate-root "$EVIDENCE/run" \
    --writer-zdf-record \
      "$EVIDENCE/run/oracle_trazdf_matrix_kt00000001.bin" \
    --output "$EVIDENCE/manifest/writer_identity.json"
  if env PYTHONPATH="$LANE_PYTHONPATH" "$PYTHON" "$IDENTITY_GATE" \
      --writer-identity-only \
      --writer-twin-root "$SCALAR_TWIN" \
      --writer-candidate-root "$EVIDENCE/run" \
      --writer-zdf-record \
        "$EVIDENCE/run/oracle_trazdf_matrix_kt00000001.bin" \
      --plant-writer-active-word \
      --output "$EVIDENCE/manifest/writer_identity_active_plant.json"; then
    printf 'REFUSE_WRITER_IDENTITY_PLANT_MISSED\n' >&2
    return 71
  fi
  grep -q '"verdict": "DEBT"' \
    "$EVIDENCE/manifest/writer_identity_active_plant.json" || {
    printf 'REFUSE_WRITER_IDENTITY_PLANT_DID_NOT_FIRE\n' >&2; return 71; }
  (
    cd "$EVIDENCE/run"
    sha256sum oracle_*.bin >../manifest/oracle_records.sha256
  )
  printf 'ACQUISITION_DONE: %s (r4 scalar-math step-2 stages, tracer and tra_zdf operands)\n' \
    "$EVIDENCE"
}

for path in "$PATCH" "$ZDF_BASE_PATCH" "$ZDF_KT2_PATCH" \
  "$DECK_PATCH" "$BASE_DECK" \
  "$SOURCE_CFG/MY_SRC/stprk3.F90" "$SOURCE_CFG/MY_SRC/stprk3_stg.F90" \
  "$SOURCE_CFG/WORK/trazdf.F90" \
  "$BASE_EXP00/namelist_ref" "$IDENTITY_GATE"; do
  [[ -f $path ]] || { printf 'REFUSE_MISSING_INPUT: %s\n' "$path" >&2; exit 66; }
done
[[ $(sha256sum "$SOURCE_CFG/MY_SRC/stprk3.F90" | awk '{print $1}') == $SOURCE_STP_SHA ]]
[[ $(sha256sum "$SOURCE_CFG/MY_SRC/stprk3_stg.F90" | awk '{print $1}') == $SOURCE_STG_SHA ]]
[[ $(sha256sum "$SOURCE_CFG/WORK/trazdf.F90" | awk '{print $1}') == $SOURCE_ZDF_SHA ]]

dry=$(mktemp -d)
cp "$SOURCE_CFG/MY_SRC/stprk3.F90" "$SOURCE_CFG/MY_SRC/stprk3_stg.F90" "$dry/"
patch -s --fuzz=0 -p1 -d "$dry" <"$PATCH"
cp "$SOURCE_CFG/WORK/trazdf.F90" "$dry/trazdf.F90.shipped"
patch -s --fuzz=0 -p0 -d "$dry" <"$ZDF_BASE_PATCH"
cp "$dry/trazdf.F90.shipped" "$dry/trazdf.F90"
patch -s --fuzz=0 -p0 -d "$dry" <"$ZDF_KT2_PATCH"
[[ $(sha256sum "$dry/stprk3.F90" | awk '{print $1}') == $PATCHED_STP_SHA ]]
[[ $(sha256sum "$dry/stprk3_stg.F90" | awk '{print $1}') == $PATCHED_STG_SHA ]]
[[ $(sha256sum "$dry/trazdf.F90" | awk '{print $1}') == $PATCHED_ZDF_SHA ]]
grep -q 'oracle_zdf_entry_kt",I8.8' "$dry/stprk3.F90"
grep -q 'oracle_qsr_stage3_kt",I8.8' "$dry/stprk3_stg.F90"
grep -q 'kstp <= nit000 + 1' "$dry/stprk3_stg.F90"
grep -q 'oracle_trazdf_matrix_kt' "$dry/trazdf.F90"
grep -q 'kt <= nit000 + 1' "$dry/trazdf.F90"

printf 'PREFLIGHT: source=%s target=%s arch=conda-scalarmath ranks=1 evidence=%s\n' \
  "$SOURCE_NAME" "$TARGET_NAME" "$EVIDENCE"
if [[ $finalize_existing -eq 1 ]]; then
  [[ -d $EVIDENCE/run ]] || {
    printf 'REFUSE_MISSING_EVIDENCE: %s\n' "$EVIDENCE/run" >&2; exit 68; }
  finalize_evidence
  exit 0
fi
if [[ $do_run -ne 1 ]]; then
  printf 'WOULD_RUN: EVIDENCE=%s %s --run\nPREFLIGHT_OK\n' "$EVIDENCE" "$0"
  exit 0
fi

[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the acquisition tool first\n' >&2; exit 67; }
[[ ! -e $TARGET_CFG && ! -e $EVIDENCE ]] || {
  printf 'REFUSE_TARGET_EXISTS: %s or %s\n' "$TARGET_CFG" "$EVIDENCE" >&2; exit 68; }
readonly COMMIT=$(git -C "$repo" rev-parse HEAD)

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
cd "$NEMO_ROOT"
./makenemo -r "$SOURCE_NAME" -n "$TARGET_NAME" -d "OCE" -m conda-scalarmath -j 0
cp -r "$SOURCE_CFG/MY_SRC/." "$TARGET_CFG/MY_SRC/"
patch -s --fuzz=0 -p1 -d "$TARGET_CFG/MY_SRC" <"$PATCH"
cp "$dry/trazdf.F90" "$TARGET_CFG/MY_SRC/trazdf.F90"
readonly WORK_CFGS=$NEMO_ROOT/cfgs/work_cfgs.txt
if ! grep -qE "^${TARGET_NAME}[[:space:]]+OCE([[:space:]]|$)" "$WORK_CFGS"; then
  printf '%s OCE\n' "$TARGET_NAME" >>"$WORK_CFGS"
fi
touch "$TARGET_CFG/MY_SRC/"*.F90
./makenemo -n "$TARGET_NAME" -m conda-scalarmath -j 8
readonly WRITER=$TARGET_CFG/BLD/bin/nemo.exe
[[ -x $WRITER ]]
grep -q 'oracle_qsr_stage3_kt",I8.8' "$TARGET_CFG/BLD/ppsrc/nemo/stprk3_stg.f90"
grep -q 'oracle_zdf_entry_kt",I8.8' "$TARGET_CFG/BLD/ppsrc/nemo/stprk3.f90"
grep -q 'oracle_trazdf_matrix_kt' "$TARGET_CFG/BLD/ppsrc/nemo/trazdf.f90"
if nm -D "$WRITER" | grep -q '_ZGV'; then
  printf 'REFUSE_COMPILED_MATH: vector symbol in scalar writer\n' >&2; exit 69
fi

mkdir -p "$EVIDENCE/manifest" "$EVIDENCE/run"
cp "$BASE_DECK" "$EVIDENCE/run/namelist_cfg"
patch -s --fuzz=0 "$EVIDENCE/run/namelist_cfg" <"$DECK_PATCH"
. "$here/deck.sh"
gyre_hires_verify_deck r4 "$EVIDENCE/run/namelist_cfg"
cp "$BASE_EXP00/namelist_ref" "$EVIDENCE/run/"
for xml in "$BASE_EXP00"/*.xml; do [[ ! -e $xml ]] || cp "$xml" "$EVIDENCE/run/"; done
cp "$WRITER" "$EVIDENCE/run/nemo"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
printf 'scalar\n' >"$EVIDENCE/manifest/math_flavour"
gyre_hires_write_resolved_namelist "$EVIDENCE/run/namelist_cfg" \
  "$EVIDENCE/manifest/resolved_namelist_lines.txt"
sha256sum "$EVIDENCE/run/nemo" "$EVIDENCE/run/namelist_cfg" \
  "$EVIDENCE/run/namelist_ref" "$PATCH" "$ZDF_BASE_PATCH" \
  "$ZDF_KT2_PATCH" >"$EVIDENCE/manifest/run.sha256"
(
  cd "$EVIDENCE/run"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  mpirun -np 1 ./nemo >run.user.log 2>&1
)

finalize_evidence
