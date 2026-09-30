#!/usr/bin/env bash
set -Eeuo pipefail

# OPERATOR-EXECUTED ACQUISITION ONLY.  Round 185 measured that the largest
# surviving day-240 owner is the state already present at day 180.  This run
# extends the admitted Round-123 compiled-order process stream backwards over
# steps 1..1080 without changing its field list or byte layout.
refuse_on_error() {
  local status=$?
  printf 'REFUSE: round185 pre-day180 acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_YRPERT
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R185PREPROC
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/nemo_seed0
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round185/oracle_pre180_process
readonly START_STEP=1
readonly END_STEP=1080
readonly EXPECTED_COUNT=1080
readonly EXPECTED_SIZE=$((16 + 11 * 4 + 8 + (6 * 36 * 26 * 31 + 3 * 36 * 26) * 8))
readonly EXPECTED_TOTAL=$((EXPECTED_COUNT * EXPECTED_SIZE))
readonly RESTART_0180=GYRE_OMIP_L2_P3_00000180_restart.nc
readonly RESTART_1080=GYRE_OMIP_L2_P3_00001080_restart.nc
readonly SHA_0180=853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6
readonly SHA_1080=6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976
readonly SOURCE_BINARY_SHA=578c88f17ecaa8052276ff43e6b6c928f5be49fb218d4af33bc8718472613c4a

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_PATCH=$here/stprk3_stg_round185.patch
readonly NAMELIST_PATCH=$here/namelist_cfg_round185.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_year_owners.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round185.md
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

if [[ "$EXPECTED_SIZE" -ne 1415300 || "$EXPECTED_TOTAL" -ne 1528524000 ]]; then
  printf 'REFUSE: frozen layout changed: count=%s size=%s total=%s\n' \
    "$EXPECTED_COUNT" "$EXPECTED_SIZE" "$EXPECTED_TOTAL" >&2
  exit 64
fi
for path in "$SOURCE_PATCH" "$NAMELIST_PATCH" "$GATE" "$PREREG" \
  "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
  "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/ocean.output" \
  "$SOURCE_RUN/$RESTART_0180" "$SOURCE_RUN/$RESTART_1080"; do
  if [[ ! -f "$path" ]]; then
    printf 'REFUSE: missing source-card input %s\n' "$path" >&2
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
  printf 'REFUSE: admitted year run did not use the frozen source binary\n' >&2
  exit 65
fi
for pair in "$RESTART_0180:$SHA_0180" "$RESTART_1080:$SHA_1080"; do
  name=${pair%%:*}
  expected=${pair#*:}
  digest=$(sha256sum "$SOURCE_RUN/$name" | awk '{print $1}')
  if [[ "$digest" != "$expected" ]]; then
    printf 'REFUSE: source restart %s is %s, expected %s\n' \
      "$name" "$digest" "$expected" >&2
    exit 65
  fi
done
for pattern in 'number of the last time step.*nn_itend *= *2160' \
  'ocean time step.*rn_Dt *= *14400' \
  'Tiling \(T\) or not \(F\).*ln_tile *= *F' \
  'Light penetration in temperature Eq.*ln_traqsr *= *T' \
  'open boundaries not used \(ln_bdy = F\)' \
  'geothermal heating at ocean bottom.*ln_trabbc *= *F' \
  'bottom boundary layer flag.*ln_trabbl *= *F' \
  'Apply relaxation.*ln_tradmp *= *F' \
  'convection mass flux \(mfc\).*ln_zdfmfc *= *F' \
  'OSMOSIS-OBL closure \(OSM\).*ln_zdfosm *= *F' \
  'non-penetrative convection \(npc\).*ln_zdfnpc *= *F'; do
  if ! grep -Eq "$pattern" "$SOURCE_RUN/ocean.output"; then
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2
    exit 65
  fi
done

removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$SOURCE_PATCH")
if [[ "$removed" -ne 0 ]]; then
  printf 'REFUSE: process writer patch removes %s source lines\n' "$removed" >&2
  exit 66
fi
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$NAMELIST_PATCH")
if [[ "$removed" -ne 1 ]] || ! grep -q '^-.*nn_itend.*2160' "$NAMELIST_PATCH"; then
  printf 'REFUSE: namelist patch does not replace exactly nn_itend=2160\n' >&2
  exit 66
fi

dry=$(mktemp -d /tmp/gyre-r185-pre180-source.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$SOURCE_ROOT/MY_SRC/stprk3_stg.F90" "$dry/stprk3_stg.F90"
patch -s --fuzz=0 "$dry/stprk3_stg.F90" <"$SOURCE_PATCH"
cp "$SOURCE_RUN/namelist_cfg" "$dry/namelist_cfg"
patch -s --fuzz=0 "$dry/namelist_cfg" <"$NAMELIST_PATCH"
for needle in "r123_magic = 'NEMO_L2_R123PROC'" \
  'kstp >= 1 .AND. kstp <= 1080' \
  'WRITE(r123_unit) ts(:,:,:,jp_tem,Kbb)' \
  'WRITE(r123_unit) r3t(:,:,Kbb), r3t(:,:,Kmm), r3t(:,:,Kaa)' \
  'CALL tra_adv' 'CALL tra_sbc_RK3' 'CALL tra_qsr' 'CALL tra_ldf' \
  'CALL tra_zdf' 'WRITE(r123_unit) ts(:,:,:,jp_tem,Kaa)'; do
  if ! grep -Fq "$needle" "$dry/stprk3_stg.F90"; then
    printf 'REFUSE: dry process source lacks %s\n' "$needle" >&2
    exit 66
  fi
done
if [[ "$(grep -Fc 'WRITE(r123_unit)' "$dry/stprk3_stg.F90")" -ne 9 ]]; then
  printf 'REFUSE: dry process source does not contain exactly nine writes\n' >&2
  exit 66
fi
for row in 'nn_itend *= *1080' 'nn_stock *= *180' \
           'nn_write *= *2160' 'nn_pert_seed *= *0'; do
  if ! grep -Eq "^[[:space:]]*$row" "$dry/namelist_cfg"; then
    printf 'REFUSE: dry target namelist lacks row: %s\n' "$row" >&2
    exit 66
  fi
done

syntax=$(mktemp -d /tmp/gyre-r185-pre180-syntax.XXXXXX)
printf 'temporary syntax-proof directory (retained): %s\n' "$syntax"
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P \
  -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/stprk3_stg.F90" -o "$syntax/stprk3_stg.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/stprk3_stg.f90"
printf 'SYNTAX_PROOF_PASS stprk3_stg.f90\n'

for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 6291456 ]]; then
    printf 'REFUSE: %s has under 6 GB free\n' "$mount" >&2
    exit 67
  fi
done

manifest=$(mktemp -d /tmp/gyre-r185-pre180-manifest.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$SOURCE_PATCH" "$NAMELIST_PATCH" \
  "$GATE" "$PREREG" "$SOURCE_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90" \
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
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/stprk3_stg.F90" <"$SOURCE_PATCH"
if ! cmp -s "$dry/stprk3_stg.F90" "$TARGET_ROOT/MY_SRC/stprk3_stg.F90"; then
  printf 'REFUSE: target writer differs from the syntax-proved source\n' >&2
  exit 68
fi
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED=$TARGET_ROOT/BLD/ppsrc/nemo/stprk3_stg.f90
if [[ ! -x "$BINARY" || ! -f "$COMPILED" ]]; then
  printf 'REFUSE: target binary or compiled process source is missing\n' >&2
  exit 68
fi
for needle in "r123_magic = 'NEMO_L2_R123PROC'" \
  'kstp >= 1 .AND. kstp <= 1080' \
  'WRITE(r123_unit) ts(:,:,:,jp_tem,Kbb)' \
  'WRITE(r123_unit) r3t(:,:,Kbb), r3t(:,:,Kmm), r3t(:,:,Kaa)' \
  'WRITE(r123_unit) ts(:,:,:,jp_tem,Kaa)' \
  'CALL tra_adv' 'CALL tra_sbc_RK3' 'CALL tra_qsr' 'CALL tra_ldf' 'CALL tra_zdf'; do
  if ! grep -Fq "$needle" "$COMPILED"; then
    printf 'REFUSE: compiled process source lacks %s\n' "$needle" >&2
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
  printf 'REFUSE: staged namelist differs from the syntax-proved namelist\n' >&2
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
  -name 'oracle_process_budget_kt*.bin' | wc -l)
if [[ "$count" -ne "$EXPECTED_COUNT" ]]; then
  printf 'REFUSE: NEMO wrote %s process frames, expected %s\n' \
    "$count" "$EXPECTED_COUNT" >&2
  exit 70
fi
total=0
for ((step=START_STEP; step<=END_STEP; step++)); do
  name=$(printf 'oracle_process_budget_kt%08d.bin' "$step")
  path=$TARGET_RUN/$name
  if [[ ! -f "$path" ]]; then
    printf 'REFUSE: NEMO omitted process frame %s\n' "$name" >&2
    exit 70
  fi
  bytes=$(stat -c %s "$path")
  if [[ "$bytes" -ne "$EXPECTED_SIZE" ]]; then
    printf 'REFUSE: process frame %s is %s bytes, expected %s\n' \
      "$name" "$bytes" "$EXPECTED_SIZE" >&2
    exit 70
  fi
  total=$((total + bytes))
done
if [[ "$total" -ne "$EXPECTED_TOTAL" ]]; then
  printf 'REFUSE: process frames total %s bytes, expected %s\n' \
    "$total" "$EXPECTED_TOTAL" >&2
  exit 70
fi

for pair in "$RESTART_0180:$SHA_0180" "$RESTART_1080:$SHA_1080"; do
  name=${pair%%:*}
  expected=${pair#*:}
  if [[ ! -f "$TARGET_RUN/$name" ]]; then
    printf 'REFUSE: target run omitted passive-control restart %s\n' "$name" >&2
    exit 71
  fi
  digest=$(sha256sum "$TARGET_RUN/$name" | awk '{print $1}')
  if [[ "$digest" != "$expected" ]] || ! cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; then
    printf 'REFUSE: instrument perturbed restart %s: %s, expected %s\n' \
      "$name" "$digest" "$expected" >&2
    exit 71
  fi
done

(
  cd "$TARGET_RUN"
  sha256sum oracle_process_budget_kt*.bin >process_records.sha256
  manifest_digest=$(sha256sum process_records.sha256 | awk '{print $1}')
  printf '%s %s %s\n' "$manifest_digest" "$COMMIT" \
    process_records.sha256 >process_records.stamp
)

r185_gate() {
  "$PY" "$GATE" --process-record "$TARGET_RUN" \
    --process-start-step 1 --process-end-step 1080 \
    --process-restart-sha "$RESTART_0180=$SHA_0180" \
    --process-restart-sha "$RESTART_1080=$SHA_1080" \
    --expect-commit "$COMMIT" "$@"
}
r185_gate --json "$TARGET_RUN/round185_process_record_validation.json"
for plant in process-stamp process-truncation process-sbc-ulp \
             process-sbc-effect process-trajectory-ulp; do
  if r185_gate --plant "$plant" \
       --json "$TARGET_RUN/round185_${plant}_plant.json" \
       >"$TARGET_RUN/round185_${plant}_plant.log" 2>&1; then
    printf 'REFUSE: round185 %s plant stayed green\n' "$plant" >&2
    exit 72
  fi
  if ! grep -Fq "STATUS PLANT-FIRED: $plant" \
       "$TARGET_RUN/round185_${plant}_plant.log"; then
    printf 'REFUSE: round185 %s exited nonzero without its marker\n' \
      "$plant" >&2
    exit 72
  fi
done

(
  cd "$TARGET_RUN"
  sha256sum process_records.sha256 process_records.stamp \
    "$RESTART_0180" "$RESTART_1080" \
    round185_process_record_validation.json round185_*_plant.json \
    round185_*_plant.log >round185_outputs.sha256
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
printf 'ROUND185_PRE180_PROCESS_RECORD_READY %s\n' "$TARGET_RUN"
