#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED ACQUISITION ONLY. This script invokes makenemo/mpirun.
# It creates a new configuration and never modifies canonical NEMO src/.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R46KT2
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R63KRHS
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/oracle_kt2_stage
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round63/oracle_krhs_split

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly WRITER=$here/l2_r63_krhs.F90
readonly STP_PATCH=$here/stprk3_stg_round63.patch
readonly ADV_PATCH=$here/traadv_fct_round63.patch
readonly ZDF_PATCH=$here/trazdf_round63.patch
readonly TKE_PATCH=$here/zdftke_round63.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round54_tracer_decomposition.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round63.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly KRHS_RECORD=oracle_krhs_split_kt00000002.bin
readonly TKE_RECORD=oracle_tke_rhs_split_kt00000002.bin

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63;
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$WRITER" "$STP_PATCH" "$ADV_PATCH" "$ZDF_PATCH" "$TKE_PATCH" "$GATE" "$ADMISSION" "$PREREG"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s\n' "$path" >&2; exit 64; }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]]
[[ -f "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" ]]
[[ -f "$SOURCE_ROOT/MY_SRC/trazdf.F90" ]]
[[ -f "$SOURCE_ROOT/MY_SRC/zdftke.F90" ]]
[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
}

for pattern in 'Assimilation cycle.*nn_no *= *0' 'ln_tile *= *F' 'ln_traqsr *= *T' 'ln_bdy *= *F' 'ln_traadv_fct *= *T' 'nn_fct_h *= *2' 'nn_fct_v *= *2' 'nn_fct_imp *= *1' 'ln_zad_Aimp *= *F' 'ln_traldf_msc *= *F' 'ln_trabbc *= *F' 'ln_trabbl *= *F' 'ln_tradmp *= *F' 'ln_zdfmfc *= *F' 'ln_zdfosm *= *F' 'ln_zdfnpc *= *F'; do
  grep -Eq "$pattern" "$SOURCE_RUN/ocean.output" || {
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2
    exit 65
  }
done

# Dry-apply every additive patch on the exact source base.
dry=$(mktemp -d /tmp/gyre-r63-source.XXXXXX)
cp "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90"
cp "$SOURCE_ROOT/MY_SRC/trazdf.F90" "$dry/trazdf.F90"
cp "$SOURCE_ROOT/MY_SRC/zdftke.F90" "$dry/zdftke.F90"
cp "$NEMO_ROOT/src/OCE/TRA/traadv_fct.F90" "$dry/traadv_fct.F90"
patch -s "$dry/stprk3_stg.F90" <"$STP_PATCH"
patch -s "$dry/trazdf.F90" <"$ZDF_PATCH"
patch -s "$dry/zdftke.F90" <"$TKE_PATCH"
patch -s "$dry/traadv_fct.F90" <"$ADV_PATCH"
for pair in "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90:stprk3_stg.F90" "$SOURCE_ROOT/MY_SRC/trazdf.F90:trazdf.F90" "$SOURCE_ROOT/MY_SRC/zdftke.F90:zdftke.F90" "$NEMO_ROOT/src/OCE/TRA/traadv_fct.F90:traadv_fct.F90"; do
  base=$(printf '%s' "$pair" | cut -d: -f1)
  result=$(printf '%s' "$pair" | cut -d: -f2)
  [[ -z "$(diff "$base" "$dry/$result" | grep '^<' || true)" ]] || {
    printf 'REFUSE: %s patch removes/replaces a base line\n' "$result" >&2
    exit 66
  }
done
grep -q 'CALL r63_begin' "$dry/stprk3_stg.F90"
grep -q 'CALL r63_fct_first' "$dry/traadv_fct.F90"
grep -q 'CALL r63_content_finish' "$dry/trazdf.F90"
grep -q 'CALL r63_tke_before_row' "$dry/zdftke.F90"

# Reproduce FCM preprocessing and compile every dry-patched file.
syntax=$(mktemp -d /tmp/gyre-r63-syntax.XXXXXX)
preprocess() {
  cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$1" -o "$2"
}
preprocess "$WRITER" "$syntax/l2_r63_krhs.f90"
for name in stprk3_stg trazdf zdftke traadv_fct; do
  preprocess "$dry/$name.F90" "$syntax/$name.f90"
done
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/l2_r63_krhs.f90"
for name in stprk3_stg trazdf zdftke traadv_fct; do
  "$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/$name.f90"
done

for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2; exit 67;
  }
done

manifest=$(mktemp -d /tmp/gyre-r63-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" "$WRITER" "$STP_PATCH" "$ADV_PATCH" "$ZDF_PATCH" "$TKE_PATCH" "$GATE" "$ADMISSION" "$PREREG" "$SOURCE_RUN/ocean.output" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l2_r63_krhs.F90"
cp "$NEMO_ROOT/src/OCE/TRA/traadv_fct.F90" "$TARGET_ROOT/MY_SRC/traadv_fct.F90"
patch -s "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" <"$STP_PATCH"
patch -s "$TARGET_ROOT/MY_SRC/trazdf.F90" <"$ZDF_PATCH"
patch -s "$TARGET_ROOT/MY_SRC/zdftke.F90" <"$TKE_PATCH"
patch -s "$TARGET_ROOT/MY_SRC/traadv_fct.F90" <"$ADV_PATCH"
# The requested two-step acquisition is the sole target-card change.
sed -i -E 's/^([[:space:]]*nn_itend[[:space:]]*=[[:space:]]*)10/\12/' "$TARGET_ROOT/EXP00/namelist_cfg"
grep -Eq 'nn_itend[[:space:]]*=[[:space:]]*2([^0-9]|$)' "$TARGET_ROOT/EXP00/namelist_cfg"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]]
for check in 'l2_r63_krhs.f90:NEMO_L2_R63KRS1' 'stprk3_stg.f90:CALL r63_snapshot' 'traadv_fct.f90:CALL r63_fct_first' 'trazdf.f90:CALL r63_content_finish' 'zdftke.f90:CALL r63_tke_before_row'; do
  file=$(printf '%s' "$check" | cut -d: -f1)
  needle=$(printf '%s' "$check" | cut -d: -f2-)
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
  test "$PIPESTATUS" -eq 0
  [[ -s "$KRHS_RECORD" && -s "$TKE_RECORD" ]]
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  for record in "$KRHS_RECORD" "$TKE_RECORD"; do
    digest=$(sha256sum "$record" | awk '{print $1}')
    printf '%s %s %s\n' "$digest" "$COMMIT" "$record" >"$record.stamp"
  done
)

r63_gate() {
  "$PY" "$GATE" --mode krhs-calibrate --krhs-record "$TARGET_RUN/$KRHS_RECORD" --tke-rhs-record "$TARGET_RUN/$TKE_RECORD" --krhs-stamp "$TARGET_RUN/$KRHS_RECORD.stamp" --tke-rhs-stamp "$TARGET_RUN/$TKE_RECORD.stamp" --producer-commit "$TARGET_RUN/producer_commit.txt" --resolved-output "$TARGET_RUN/ocean.output" --expect-commit "$COMMIT" "$@"
}
r63_gate --output "$TARGET_RUN/round63_krhs_validation.json"
for plant in stamp krhs-ulp tke-ulp krhs-truncation tke-truncation; do
  if r63_gate --plant "$plant" >"$TARGET_RUN/round63_$plant-plant.log" 2>&1; then
    printf 'REFUSE: round63 %s plant stayed green\n' "$plant" >&2; exit 69
  fi
done

"$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" --twin /nonexistent --identical mesh_mask.nc --allowed-new "$KRHS_RECORD" "$TKE_RECORD" --output "$TARGET_RUN/round63_admission.json"
if "$PY" "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" --twin /nonexistent --identical mesh_mask.nc --allowed-new "$KRHS_RECORD" "$TKE_RECORD" --plant-consumed --output "$TARGET_RUN/round63_admission_plant.json" >"$TARGET_RUN/round63_admission_plant.log" 2>&1; then
  printf 'REFUSE: round63 admission plant stayed green\n' >&2; exit 69
fi

(
  cd "$TARGET_RUN"
  sha256sum "$KRHS_RECORD" "$TKE_RECORD" "$KRHS_RECORD.stamp" "$TKE_RECORD.stamp" mesh_mask.nc round63_*.json round63_*-plant.log round63_admission_plant.log >round63_outputs.sha256
)
printf 'ROUND63_KRHS_SPLIT_READY %s\n' "$TARGET_RUN"
