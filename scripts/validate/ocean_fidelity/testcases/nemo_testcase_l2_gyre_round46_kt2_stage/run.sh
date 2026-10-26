#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED ACQUISITION.  The agent writes this script and stops; only the
# operator may invoke makenemo/mpirun.  It never edits NEMO's canonical src/.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R41ADVSP
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R46KT2
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round41/oracle_dynadv_split
readonly ROUND46=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46
readonly TARGET_RUN=$ROUND46/oracle_kt2_stage
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly R40_KT1=$SOURCE_RUN/oracle_rkstage3_terms_kt00000001.bin
readonly R41_KT1=$SOURCE_RUN/oracle_dynadv_split_kt00000001_s3.bin

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly GATE=$here/../nemo_testcase_l2_gyre_round46_kt2_stage_gate.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_phase3_round46.md
readonly PREREG_JSON=$here/../manifests/nemo_testcase_l2_gyre_round46_preregister.json
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly STP2_PATCH=$here/stp2d_round46.patch
readonly STG_PATCH=$here/stprk3_stg_round46.patch
readonly ADV_PATCH=$here/dynadv_round46.patch
readonly TKE_PATCH=$here/zdftke_round46.patch
readonly WRITER=$here/l2_r46_stage.F90

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63;
}
readonly COMMIT=$(git rev-parse HEAD)
git merge-base --is-ancestor 14df593e10fe "$COMMIT"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$GATE" "$ADMISSION" "$PREREG" "$PREREG_JSON" "$STP2_PATCH" "$STG_PATCH" \
            "$ADV_PATCH" "$TKE_PATCH" "$WRITER" "$R40_KT1" "$R41_KT1"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 64; }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]]
[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2; exit 64;
}
grep -Eq 'nn_itend[[:space:]]*=[[:space:]]*([2-9]|[1-9][0-9]+)' "$SOURCE_ROOT/EXP00/namelist_cfg" || {
  printf 'REFUSE: source namelist does not run through kt=2\n' >&2; exit 65;
}
grep -q 'NEMO_L2_RKTS3_1' "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90"
grep -q 'NEMO_L2_ADVSP_1' "$SOURCE_ROOT/MY_SRC/dynadv.F90"
grep -q 'Surface wave (forced or coupled).*ln_wave.*=  F' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: wsd_effective=0 is not resolved by this run\n' >&2; exit 65;
}

# Verify all deltas before creating the config.  The final MY_SRC copies must
# remove no shipped NEMO statement; replacements touch only inherited writer
# lines (kt predicate and filename), never a physical statement.
dry=$(mktemp -d /tmp/gyre-r46-source.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/stp2d.F90" "$dry/stp2d.F90"
cp "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90"
cp "$SOURCE_ROOT/MY_SRC/dynadv.F90" "$dry/dynadv.F90"
cp "$NEMO_ROOT/src/OCE/ZDF/zdftke.F90" "$dry/zdftke.F90"
patch -s "$dry/stp2d.F90" <"$STP2_PATCH"
patch -s "$dry/stprk3_stg.F90" <"$STG_PATCH"
patch -s "$dry/dynadv.F90" <"$ADV_PATCH"
patch -s "$dry/zdftke.F90" <"$TKE_PATCH"
for pair in "src/OCE/stp2d.F90:stp2d.F90" \
            "src/OCE/stprk3_stg.F90:stprk3_stg.F90" \
            "src/OCE/DYN/dynadv.F90:dynadv.F90" \
            "src/OCE/ZDF/zdftke.F90:zdftke.F90"; do
  shipped=${pair%%:*}; result=${pair##*:}
  removed=$(diff "$NEMO_ROOT/$shipped" "$dry/$result" | grep -c '^<' || true)
  if [[ "$result" == stprk3_stg.F90 ]]; then
    parent_removed=$(diff "$NEMO_ROOT/$shipped" "$SOURCE_ROOT/MY_SRC/$result" | grep -c '^<' || true)
    [[ "$removed" -eq "$parent_removed" ]] || {
      printf 'REFUSE: round46 removes a new shipped stprk3_stg line\n' >&2; exit 66;
    }
  else
    [[ "$removed" -eq 0 ]] || {
      printf 'REFUSE: final %s removes %s shipped lines\n' "$result" "$removed" >&2; exit 66;
    }
  fi
done
# The only parent lines replaced in stprk3_stg are the five round-40 writer
# predicates/name needed for kt2/per-kt output; no physical statement is one.
parent_delta=$(diff "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90" | grep '^<' || true)
[[ "$(printf '%s\n' "$parent_delta" | grep -c '^<')" -eq 5 ]]
[[ -z "$(printf '%s\n' "$parent_delta" | grep -Ev 'kstp == nit000|oracle_rkstage3_terms_kt00000001' || true)" ]] || {
  printf 'REFUSE: stprk3_stg patch replaces a non-writer parent line\n' >&2; exit 66;
}
grep -q 'kstp <= nit000 + 1' "$dry/stprk3_stg.F90"
grep -q 'oracle_dynadv_split_kt",I8.8' "$dry/dynadv.F90"

for mount in /tmp "$ROUND46" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2; exit 67;
  }
done

manifest=$(mktemp -d /tmp/gyre-r46-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$WRITER" \
  "$STP2_PATCH" "$STG_PATCH" "$ADV_PATCH" "$TKE_PATCH" "$GATE" "$PREREG" "$PREREG_JSON" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
# cp -r plus touch is intentional: preserved mtimes caused stale FCM extracts.
cp -r "$SOURCE_ROOT/EXP00/." "$TARGET_ROOT/EXP00/"
cp -r "$SOURCE_ROOT/MY_SRC/." "$TARGET_ROOT/MY_SRC/"
cp "$SOURCE_ROOT/cpp_${SOURCE_CFG}.fcm" "$TARGET_ROOT/cpp_${TARGET_CFG}.fcm"
(
  cd "$TARGET_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/copied_cfg_before_patch.sha256"
cmp "$manifest/source_cfg.sha256" "$manifest/copied_cfg_before_patch.sha256"
cp "$NEMO_ROOT/src/OCE/ZDF/zdftke.F90" "$TARGET_ROOT/MY_SRC/zdftke.F90"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l2_r46_stage.F90"
patch -s "$TARGET_ROOT/MY_SRC/stp2d.F90" <"$STP2_PATCH"
patch -s "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" <"$STG_PATCH"
patch -s "$TARGET_ROOT/MY_SRC/dynadv.F90" <"$ADV_PATCH"
patch -s "$TARGET_ROOT/MY_SRC/zdftke.F90" <"$TKE_PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
binary=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$binary" ]]
for check in \
  'stp2d.f90:r46_begin' 'stprk3_stg.f90:r46_finish' \
  'dynadv.f90:oracle_dynadv_split_kt' 'l2_r46_stage.f90:NEMO_L2_R46STG1' \
  'zdftke.f90:PUBLIC   dissl'; do
  file=${check%%:*}; needle=${check#*:}
  grep -q "$needle" "$TARGET_ROOT/BLD/ppsrc/nemo/$file" || {
    printf 'REFUSE: compiled writer check failed: %s\n' "$check" >&2; exit 68;
  }
done
if nm -D "$binary" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2; exit 68
fi
sha256sum "$binary" >"$manifest/binary.sha256"

mkdir -p "$ROUND46"
mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref \
namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done
cp "$binary" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  for name in $PREPARED; do cmp "$SOURCE_RUN/$name" "$name"; done
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } 2>>run.user.time.log
  test "${PIPESTATUS[0]}" -eq 0
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)

# Raw twin report, then the only verdict: schema-aware consumed-field
# admission.  The widened kt=1 RKTS3/ADVSP records contain whole-array ``ww``
# halos, so raw identity is informative but not gating.  The gate below also
# bit-tests kt=2 RKTS3/ADVSP shared payloads against their within-run duplicate
# named fields in oracle_momstage_kt00000002_s3.bin (28 and 29 exact pairs).
# This proves duplicate identity, not that whole-array ``ww`` halos are defined.
: >"$TARGET_RUN/round46_raw_twin_cmp.log"
for source_record in "$SOURCE_RUN"/oracle_*.bin; do
  name=${source_record##*/}
  [[ -e "$TARGET_RUN/$name" ]] || { printf 'REFUSE: missing inherited %s\n' "$name" >&2; exit 69; }
  if cmp -s "$source_record" "$TARGET_RUN/$name"; then
    printf 'RAW_IDENTICAL %s\n' "$name"
  else
    printf 'RAW_DIFFERS   %s (consumed-field admission follows)\n' "$name"
  fi
done | tee "$TARGET_RUN/round46_raw_twin_cmp.log"

readonly NEW="oracle_momstage_kt00000001_s1.bin oracle_momstage_kt00000001_s2.bin \
oracle_momstage_kt00000001_s3.bin oracle_momstage_kt00000002_s1.bin \
oracle_momstage_kt00000002_s2.bin oracle_momstage_kt00000002_s3.bin \
oracle_rkstage3_terms_kt00000002.bin oracle_dynadv_split_kt00000002_s3.bin"
"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new $NEW --output "$TARGET_RUN/round46_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new $NEW --plant-consumed \
     --output "$TARGET_RUN/round46_admission_plant.json" \
     >"$TARGET_RUN/round46_admission_plant.log" 2>&1; then
  printf 'REFUSE: consumed-field admission plant stayed green\n' >&2; exit 70
fi
grep -q '"plant_applied": true' "$TARGET_RUN/round46_admission_plant.json"

gate_common=(--root "$TARGET_RUN" --expect-commit "$COMMIT" --mode validate \
  --round40-kt1 "$R40_KT1" --round41-kt1 "$R41_KT1")
"$PY" "$GATE" "${gate_common[@]}" --output "$TARGET_RUN/round46_validation.json"
for plant in header truncation calibration twin stamp; do
  if "$PY" "$GATE" "${gate_common[@]}" --plant "$plant" \
       >"$TARGET_RUN/round46_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
  fi
done
(
  cd "$TARGET_RUN"
  sha256sum oracle_*.bin "$FINAL_RESTART" mesh_mask.nc round46_*.json \
    round46_*_plant.log round46_raw_twin_cmp.log >round46_outputs.sha256
)
printf 'ROUND46_GYRE_KT2_STAGE_READY %s\n' "$TARGET_RUN"
