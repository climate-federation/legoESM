#!/usr/bin/env bash
set -Eeuo pipefail

# USER-EXECUTED ACQUISITION ONLY.  This script invokes makenemo and mpirun.
# Round 125 prepares and syntax-checks it but never runs either command.
refuse_on_error() {
  local status=$?
  printf 'REFUSE: round125 acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_YRPERT
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R125ZDFMAG
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
readonly SOURCE_RUN=$L2/year_fromrest/nemo_seed0
readonly PROCESS_RUN=$L2/round123/oracle_process_budget
readonly TARGET_RUN=$L2/round125/oracle_vertical_decomposition
readonly PROCESS_COMMIT=af3f7215060fc17c71adc6794817c710df8ee471
readonly SOURCE_BINARY_SHA=578c88f17ecaa8052276ff43e6b6c928f5be49fb218d4af33bc8718472613c4a
readonly RESTART_1080=GYRE_OMIP_L2_P3_00001080_restart.nc
readonly RESTART_1440=GYRE_OMIP_L2_P3_00001440_restart.nc
readonly SHA_1080=6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976
readonly SHA_1440=96529a98da0e0d89b328632a826a9d41593f81f0d1a917350f28f184d49b163a
readonly START_STEP=1081
readonly END_STEP=1440
readonly INTERVAL_COUNT=$((END_STEP - START_STEP + 1))
readonly EXPECTED_COUNT=$((INTERVAL_COUNT + 2))
readonly EXPECTED_SIZE=$((80 + 44 * 32 + 11 * 8 + 27 * 36 * 26 * 31 * 8 + 3 * 36 * 26 * 30 * 8 + 3 * 36 * 26 * 8))
readonly EXPECTED_INTERVAL_BYTES=$((INTERVAL_COUNT * EXPECTED_SIZE))
readonly EXPECTED_TOTAL=$((EXPECTED_COUNT * EXPECTED_SIZE))

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly TRA_PATCH=$here/trazdf_round125.patch
readonly NAMELIST_PATCH=$here/namelist_cfg_round125.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_year_owners.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round125.md
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

if [[ "$INTERVAL_COUNT" -ne 360 || "$EXPECTED_COUNT" -ne 362 \
      || "$EXPECTED_SIZE" -ne 6965416 \
      || "$EXPECTED_INTERVAL_BYTES" -ne 2507549760 \
      || "$EXPECTED_TOTAL" -ne 2521480592 ]]; then
  printf 'REFUSE: frozen layout arithmetic changed: interval=%s count=%s size=%s interval_bytes=%s total=%s\n' \
    "$INTERVAL_COUNT" "$EXPECTED_COUNT" "$EXPECTED_SIZE" \
    "$EXPECTED_INTERVAL_BYTES" "$EXPECTED_TOTAL" >&2
  exit 64
fi

for path in "$TRA_PATCH" "$NAMELIST_PATCH" "$GATE" "$PREREG" \
            "$SOURCE_ROOT/MY_SRC/trazdf.F90" \
            "$SOURCE_ROOT/BLD/ppsrc/nemo/trazdf.f90" \
            "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
            "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/ocean.output" \
            "$SOURCE_RUN/$RESTART_1080" "$SOURCE_RUN/$RESTART_1440" \
            "$PROCESS_RUN/producer_commit.txt"; do
  if [[ ! -f "$path" ]]; then
    printf 'REFUSE: missing frozen acquisition input %s\n' "$path" >&2
    exit 64
  fi
done
for path in "$SOURCE_ROOT/EXP00" "$SOURCE_ROOT/MY_SRC"; do
  if [[ ! -d "$path" ]]; then
    printf 'REFUSE: missing source-card directory %s\n' "$path" >&2
    exit 64
  fi
done
if [[ -e "$TARGET_ROOT" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: new target already exists: %s or %s\n' \
    "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
fi

digest=$(sha256sum "$SOURCE_ROOT/BLD/bin/nemo.exe" | awk '{print $1}')
if [[ "$digest" != "$SOURCE_BINARY_SHA" ]]; then
  printf 'REFUSE: source binary hash is %s, expected %s\n' \
    "$digest" "$SOURCE_BINARY_SHA" >&2
  exit 65
fi
if ! cmp -s "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo"; then
  printf 'REFUSE: admitted source run did not use the frozen source binary\n' >&2
  exit 65
fi
for pair in "$RESTART_1080:$SHA_1080" "$RESTART_1440:$SHA_1440"; do
  name=${pair%%:*}
  expected=${pair#*:}
  digest=$(sha256sum "$SOURCE_RUN/$name" | awk '{print $1}')
  if [[ "$digest" != "$expected" ]]; then
    printf 'REFUSE: source restart %s is %s, expected %s\n' \
      "$name" "$digest" "$expected" >&2
    exit 65
  fi
done
if [[ "$(<"$PROCESS_RUN/producer_commit.txt")" != "$PROCESS_COMMIT" ]]; then
  printf 'REFUSE: admitted process record has the wrong producer commit\n' >&2
  exit 65
fi
"$PY" "$GATE" --process-record "$PROCESS_RUN" \
  --expect-commit "$PROCESS_COMMIT"

for needle in "NEMO_L2_TRAZD_1" \
  "ll_l2_tra = ( lwp .AND. kt <= nit000 + 1 )" \
  "WRITE(il2_unit) 'T_Kbb_in" "WRITE(il2_unit) 'T_Krhs_in" \
  "WRITE(il2_unit) 'zwt_mix" "WRITE(il2_unit) 'zwi" \
  "WRITE(il2_unit) 'zwd" "WRITE(il2_unit) 'zws" \
  "WRITE(il2_unit) 'zwt_lu" "WRITE(il2_unit) 'rhs_T" \
  "WRITE(il2_unit) 'fwd_T" "WRITE(il2_unit) 'sol_T_pre_clamp" \
  "WRITE(il2_unit) 'sol_T_post_clamp" "WRITE(il2_unit) 'avt" \
  "WRITE(il2_unit) 'ah_wslp2" "WRITE(il2_unit) 'e3t_Kbb" \
  "WRITE(il2_unit) 'e3w_Kmm" "WRITE(il2_unit) 'r3t_Kaa"; do
  if ! grep -Fq "$needle" "$SOURCE_ROOT/MY_SRC/trazdf.F90"; then
    printf 'REFUSE: source tra_zdf writer lacks %s\n' "$needle" >&2
    exit 66
  fi
done

removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$TRA_PATCH")
if [[ "$removed" -ne 0 ]]; then
  printf 'REFUSE: vertical writer patch removes %s source lines\n' "$removed" >&2
  exit 66
fi
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$NAMELIST_PATCH")
if [[ "$removed" -ne 1 ]] || ! grep -q '^-.*nn_itend.*2160' "$NAMELIST_PATCH"; then
  printf 'REFUSE: namelist patch does not replace exactly nn_itend=2160\n' >&2
  exit 66
fi

dry=$(mktemp -d /tmp/gyre-r125-source.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$SOURCE_ROOT/MY_SRC/trazdf.F90" "$dry/trazdf.F90"
patch -s --fuzz=0 "$dry/trazdf.F90" <"$TRA_PATCH"
cp "$SOURCE_RUN/namelist_cfg" "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$NAMELIST_PATCH"
if ! grep -Fq \
  "IF( lwp .AND. kt >= 1081 .AND. kt <= 1440 )   ll_l2_tra = .TRUE." \
  "$dry/trazdf.F90"; then
  printf 'REFUSE: dry vertical source lacks the interval arm\n' >&2
  exit 66
fi
for row in 'nn_itend *= *1440' 'nn_stock *= *180' 'nn_write *= *2160' \
           'nn_pert_seed *= *0'; do
  if ! grep -Eq "^[[:space:]]*$row" "$dry/namelist_cfg"; then
    printf 'REFUSE: dry target namelist lacks row: %s\n' "$row" >&2
    exit 66
  fi
done

syntax=$(mktemp -d /tmp/gyre-r125-syntax.XXXXXX)
printf 'temporary syntax-proof directory (retained): %s\n' "$syntax"
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P \
  -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/trazdf.F90" -o "$syntax/trazdf.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/trazdf.f90"
printf 'SYNTAX_PROOF_PASS trazdf.f90\n'

for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 8388608 ]]; then
    printf 'REFUSE: %s has under 8 GB free\n' "$mount" >&2
    exit 67
  fi
done

manifest=$(mktemp -d /tmp/gyre-r125-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TRA_PATCH" "$NAMELIST_PATCH" \
  "$GATE" "$PREREG" "$SOURCE_ROOT/BLD/ppsrc/nemo/trazdf.f90" \
  "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/ocean.output" \
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
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/trazdf.F90" <"$TRA_PATCH"
if ! cmp -s "$dry/trazdf.F90" "$TARGET_ROOT/MY_SRC/trazdf.F90"; then
  printf 'REFUSE: target writer differs from syntax-proved source\n' >&2
  exit 68
fi
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED=$TARGET_ROOT/BLD/ppsrc/nemo/trazdf.f90
if [[ ! -x "$BINARY" || ! -f "$COMPILED" ]]; then
  printf 'REFUSE: target binary or compiled tra_zdf source is missing\n' >&2
  exit 68
fi
if ! cmp -s "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
     "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"; then
  printf 'REFUSE: target preprocessor keys differ from source card\n' >&2
  exit 68
fi
for needle in \
  "IF( lwp .AND. kt >= 1081 .AND. kt <= 1440 )   ll_l2_tra = .TRUE." \
  "NEMO_L2_TRAZD_1" "WRITE(il2_unit) 'zwt_mix" \
  "WRITE(il2_unit) 'zwi" "WRITE(il2_unit) 'zwd" \
  "WRITE(il2_unit) 'zws" "WRITE(il2_unit) 'zwt_lu" \
  "WRITE(il2_unit) 'rhs_T" "WRITE(il2_unit) 'fwd_T" \
  "WRITE(il2_unit) 'sol_T_pre_clamp" "WRITE(il2_unit) 'avt" \
  "WRITE(il2_unit) 'ah_wslp2"; do
  if ! grep -Fq "$needle" "$COMPILED"; then
    printf 'REFUSE: compiled tra_zdf source lacks %s\n' "$needle" >&2
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
  if [[ ! -f "$SOURCE_RUN/$name" ]]; then
    printf 'REFUSE: admitted prepared input is missing: %s\n' "$name" >&2
    exit 68
  fi
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
patch -s --fuzz=0 "$TARGET_RUN/namelist_cfg" <"$NAMELIST_PATCH"
if ! cmp -s "$dry/namelist_cfg" "$TARGET_RUN/namelist_cfg"; then
  printf 'REFUSE: staged namelist differs from preregistered truncation\n' >&2
  exit 68
fi
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
  if ! grep -Fxq 'STOP 0' run.user.stdout.log; then
    printf 'REFUSE: NEMO run did not terminate with STOP 0\n' >&2
    exit 69
  fi
  printf 'NEMO_FINISHED_UTC=%s\nNEMO_DONE\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)

count=$(find "$TARGET_RUN" -maxdepth 1 -type f \
  -name 'oracle_trazdf_matrix_kt*.bin' | wc -l)
if [[ "$count" -ne "$EXPECTED_COUNT" ]]; then
  printf 'REFUSE: NEMO wrote %s vertical frames, expected %s\n' \
    "$count" "$EXPECTED_COUNT" >&2
  exit 70
fi
total=0
for step in 1 2 $(seq "$START_STEP" "$END_STEP"); do
  name=$(printf 'oracle_trazdf_matrix_kt%08d.bin' "$step")
  path=$TARGET_RUN/$name
  if [[ ! -f "$path" ]]; then
    printf 'REFUSE: NEMO omitted vertical frame %s\n' "$name" >&2
    exit 70
  fi
  bytes=$(stat -c %s "$path")
  if [[ "$bytes" -ne "$EXPECTED_SIZE" ]]; then
    printf 'REFUSE: vertical frame %s is %s bytes, expected %s\n' \
      "$name" "$bytes" "$EXPECTED_SIZE" >&2
    exit 70
  fi
  total=$((total + bytes))
done
if [[ "$total" -ne "$EXPECTED_TOTAL" ]]; then
  printf 'REFUSE: vertical frames total %s bytes, expected %s\n' \
    "$total" "$EXPECTED_TOTAL" >&2
  exit 70
fi

for pair in "$RESTART_1080:$SHA_1080" "$RESTART_1440:$SHA_1440"; do
  name=${pair%%:*}
  expected=${pair#*:}
  if [[ ! -f "$TARGET_RUN/$name" ]]; then
    printf 'REFUSE: target run omitted passive-control restart %s\n' "$name" >&2
    exit 71
  fi
  digest=$(sha256sum "$TARGET_RUN/$name" | awk '{print $1}')
  if [[ "$digest" != "$expected" ]]; then
    printf 'REFUSE: instrument perturbed %s: %s, expected %s\n' \
      "$name" "$digest" "$expected" >&2
    exit 71
  fi
  if ! cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; then
    printf 'REFUSE: instrumented %s is not byte-identical to source run\n' \
      "$name" >&2
    exit 71
  fi
done

(
  cd "$TARGET_RUN"
  sha256sum oracle_trazdf_matrix_kt*.bin >vertical_records.sha256
  manifest_digest=$(sha256sum vertical_records.sha256 | awk '{print $1}')
  printf '%s %s %s\n' "$manifest_digest" "$COMMIT" \
    vertical_records.sha256 >vertical_records.stamp
)

r125_gate() {
  "$PY" "$GATE" --vertical-record "$TARGET_RUN" \
    --vertical-process-root "$PROCESS_RUN" --expect-commit "$COMMIT" "$@"
}
r125_gate --json "$TARGET_RUN/round125_vertical_record_validation.json"
for plant in vertical-stamp vertical-truncation vertical-matrix-ulp \
             vertical-trajectory-ulp; do
  safe_name=${plant//-/_}
  if r125_gate --plant "$plant" \
       --json "$TARGET_RUN/round125_${safe_name}_plant.json" \
       >"$TARGET_RUN/round125_${safe_name}_plant.log" 2>&1; then
    printf 'REFUSE: round125 %s plant stayed green\n' "$plant" >&2
    exit 72
  fi
  if ! grep -Fq "STATUS PLANT-FIRED: $plant" \
       "$TARGET_RUN/round125_${safe_name}_plant.log"; then
    printf 'REFUSE: round125 %s exited nonzero without its plant marker\n' \
      "$plant" >&2
    exit 72
  fi
done

(
  cd "$TARGET_RUN"
  sha256sum vertical_records.sha256 vertical_records.stamp \
    "$RESTART_1080" "$RESTART_1440" \
    round125_vertical_record_validation.json round125_*_plant.json \
    round125_*_plant.log >round125_outputs.sha256
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
printf 'ROUND125_VERTICAL_RECORD_READY %s\n' "$TARGET_RUN"
