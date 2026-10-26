#!/usr/bin/env bash
set -Eeuo pipefail

# OPERATOR-EXECUTED ACQUISITION.  The sandbox cannot create PMIx sockets.
refuse_on_error() {
  local status=$?
  printf 'REFUSE: round153 developed-FCT acquisition failed at line %s (status %s)\n' \
    "${BASH_LINENO[0]:-${LINENO}}" "$status" >&2
  exit "$status"
}
trap refuse_on_error ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R148LDF
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R153FCTD
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round148/oracle_developed_ldf
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round153/oracle_developed_fct
readonly RECORD=oracle_developed_fct_kt00001081.bin
readonly SOURCE_BINARY_SHA=9d758bf51d85a27b7692858697ddbc5fbd9b19d724f957c983a605085c89e250
readonly MODE=${1:---run}
case "$MODE" in
  --run|--admit-existing|--preflight-only|--plant-layout) ;;
  *) printf 'REFUSE: usage: %s [--run|--admit-existing|--preflight-only|--plant-layout]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly CANONICAL_SOURCE=$NEMO_ROOT/src/OCE/TRA/traadv_fct.F90
readonly WRITER=$here/l2_r153_fct.F90
readonly PATCH=$here/traadv_fct_round153.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round153_developed_fct_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round153.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$here/..:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$WRITER" "$PATCH" "$GATE" "$PREREG" "$CANONICAL_SOURCE" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
  "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/ocean.output"; do
  if [[ ! -f "$path" ]]; then
    printf 'REFUSE: missing input %s\n' "$path" >&2
    exit 64
  fi
done
for path in "$SOURCE_ROOT/EXP00" "$SOURCE_ROOT/MY_SRC"; do
  if [[ ! -d "$path" ]]; then
    printf 'REFUSE: missing source-card directory %s\n' "$path" >&2
    exit 64
  fi
done
digest=$(sha256sum "$SOURCE_ROOT/BLD/bin/nemo.exe" | awk '{print $1}')
if [[ "$digest" != "$SOURCE_BINARY_SHA" ]] || \
   ! cmp -s "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo"; then
  printf 'REFUSE: Round-148 binary ancestry changed (%s)\n' "$digest" >&2
  exit 65
fi
for row in 'nn_itend *= *1081' 'nn_stock *= *180' \
           'nn_write *= *2160' 'nn_pert_seed *= *0'; do
  if ! grep -Eq "^[[:space:]]*$row" "$SOURCE_RUN/namelist_cfg"; then
    printf 'REFUSE: source namelist lacks row %s\n' "$row" >&2
    exit 65
  fi
done
for row in 'ln_tile *= *F' 'ln_traadv_fct *= *T' 'nn_fct_h *= *2' \
           'nn_fct_v *= *2' 'nn_fct_imp *= *1' 'ln_zad_Aimp *= *F'; do
  if ! grep -Eq "$row" "$SOURCE_RUN/ocean.output"; then
    printf 'REFUSE: source run lacks resolved FCT row %s\n' "$row" >&2
    exit 65
  fi
done

removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
if [[ "$removed" -ne 0 ]]; then
  printf 'REFUSE: developed-FCT patch removes %s source lines\n' "$removed" >&2
  exit 66
fi
dry=$(mktemp -d /tmp/gyre-r153-developed-fct-source.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$CANONICAL_SOURCE" "$dry/traadv_fct.F90"
patch -s --fuzz=0 "$dry/traadv_fct.F90" <"$PATCH"
cp "$WRITER" "$dry/l2_r153_fct.F90"
check_layout() {
  local source=$1
  [[ "$(grep -Fc 'CALL r153_begin' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r153_first_flux' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r153_midpoint' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r153_average_flux' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r153_upstream' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r153_limiter_begin' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r153_limiter_end' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r153_final' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r153_coef_' "$source")" -eq 3 ]]
}
if ! check_layout "$dry/traadv_fct.F90" || \
   [[ "$(grep -Fc "magic='NEMO_L2_R153FCT'" "$dry/l2_r153_fct.F90")" -ne 1 ]] || \
   [[ "$(grep -Fc 'STORAGE_SIZE(1._wp),61' "$dry/l2_r153_fct.F90")" -ne 1 ]]; then
  printf 'REFUSE: developed-FCT writer layout is incomplete\n' >&2
  exit 66
fi
if [[ "$MODE" == --plant-layout ]]; then
  planted=$dry/traadv_fct.planted.F90
  cp "$dry/traadv_fct.F90" "$planted"
  sed -i '/CALL r153_final/d' "$planted"
  if check_layout "$planted"; then
    printf 'REFUSE: layout plant stayed green\n' >&2
    exit 68
  fi
  printf 'STATUS PLANT-FIRED: layout\n'
  printf 'REFUSE: intentional layout-plant exit\n' >&2
  exit 69
fi

syntax=$(mktemp -d /tmp/gyre-r153-developed-fct-syntax.XXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/l2_r153_fct.F90" -o "$syntax/l2_r153_fct.f90"
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/traadv_fct.F90" -o "$syntax/traadv_fct.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$syntax" "$syntax/l2_r153_fct.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" \
  -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/traadv_fct.f90"
printf 'SYNTAX_PROOF_PASS l2_r153_fct.f90 traadv_fct.f90\n'
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ROUND153_DEVELOPED_FCT_PREFLIGHT_READY\n'
  exit 0
fi

admit_existing() {
  for path in "$TARGET_RUN/$RECORD" "$TARGET_RUN/$RECORD.stamp" \
    "$TARGET_RUN/binary.sha256" "$TARGET_RUN/source_cfg.sha256" \
    "$TARGET_RUN/producer_commit.txt" "$TARGET_RUN/nemo" "$BINARY"; do
    if [[ ! -f "$path" ]]; then
      printf 'REFUSE: existing admission lacks %s\n' "$path" >&2
      exit 70
    fi
  done
  if ! cmp -s "$TARGET_RUN/nemo" "$BINARY" || \
     ! (cd "$TARGET_RUN" && sha256sum -c binary.sha256 >/dev/null); then
    printf 'REFUSE: existing Round-153 binary changed\n' >&2
    exit 70
  fi
  if ! (cd "$SOURCE_ROOT" && sha256sum -c "$TARGET_RUN/source_cfg.sha256" >/dev/null); then
    printf 'REFUSE: Round-148 source-card ancestry changed\n' >&2
    exit 70
  fi
  local producer_commit
  producer_commit=$(tr -d '[:space:]' <"$TARGET_RUN/producer_commit.txt")
  if [[ ! "$producer_commit" =~ ^[0-9a-f]{40}$ ]]; then
    printf 'REFUSE: malformed producer commit %s\n' "$producer_commit" >&2
    exit 70
  fi
  local source_list
  local target_list
  source_list=$(mktemp /tmp/r153-source-list.XXXXXX)
  target_list=$(mktemp /tmp/r153-target-list.XXXXXX)
  find "$SOURCE_RUN" -maxdepth 1 -type f \
    \( -name 'oracle*.bin' -o -name '*_restart.nc' -o -name 'mesh_mask.nc' \) \
    -printf '%f\n' | sort >"$source_list"
  find "$TARGET_RUN" -maxdepth 1 -type f \
    \( -name 'oracle*.bin' -o -name '*_restart.nc' -o -name 'mesh_mask.nc' \) \
    ! -name "$RECORD" -printf '%f\n' | sort >"$target_list"
  if ! cmp -s "$source_list" "$target_list"; then
    printf 'REFUSE: inherited file registry differs from source run\n' >&2
    diff -u "$source_list" "$target_list" >&2 || true
    exit 71
  fi
  : >"$TARGET_RUN/round153_inherited.sha256"
  while IFS= read -r name; do
    printf '%s %s\n' "$(sha256sum "$TARGET_RUN/$name" | awk '{print $1}')" "$name" \
      >>"$TARGET_RUN/round153_inherited.sha256"
  done <"$source_list"
  gate() {
    "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" \
      --expect-commit "$producer_commit" "$@"
  }
  local plant
  for plant in stamp truncation missing-field coefficients-one inherited-byte restart-byte; do
    if gate --plant "$plant" \
      --output "$TARGET_RUN/round153_${plant}_plant.json" \
      >"$TARGET_RUN/round153_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: round153 %s plant stayed green\n' "$plant" >&2
      exit 72
    fi
    if ! grep -Fq "STATUS PLANT-FIRED: $plant" \
      "$TARGET_RUN/round153_${plant}_plant.log"; then
      printf 'REFUSE: round153 %s plant lacks marker\n' "$plant" >&2
      exit 72
    fi
  done
  gate --output "$TARGET_RUN/round153_developed_fct_admission.json"
  (
    cd "$TARGET_RUN"
    sha256sum "$RECORD" "$RECORD.stamp" round153_inherited.sha256 \
      round153_developed_fct_admission.json round153_*_plant.log \
      >round153_outputs.sha256
    printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  )
  printf 'ROUND153_DEVELOPED_FCT_READY %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  admit_existing
  exit 0
fi
if [[ -e "$TARGET_ROOT" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: new target already exists: %s or %s\n' "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 67
fi
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 4194304 ]]; then
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2
    exit 67
  fi
done

manifest=$(mktemp -d /tmp/gyre-r153-developed-fct-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$WRITER" "$PATCH" "$GATE" "$PREREG" "$CANONICAL_SOURCE" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/traadv_fct.f90" \
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
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l2_r153_fct.F90"
cp "$CANONICAL_SOURCE" "$TARGET_ROOT/MY_SRC/traadv_fct.F90"
patch -s --fuzz=0 "$TARGET_ROOT/MY_SRC/traadv_fct.F90" <"$PATCH"
if ! cmp -s "$dry/l2_r153_fct.F90" "$TARGET_ROOT/MY_SRC/l2_r153_fct.F90" || \
   ! cmp -s "$dry/traadv_fct.F90" "$TARGET_ROOT/MY_SRC/traadv_fct.F90"; then
  printf 'REFUSE: target writer differs from syntax-proved source\n' >&2
  exit 68
fi
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath
if [[ ! -x "$BINARY" ]] || \
   ! check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/traadv_fct.f90" || \
   ! grep -Fq 'NEMO_L2_R153FCT' "$TARGET_ROOT/BLD/ppsrc/nemo/l2_r153_fct.f90"; then
  printf 'REFUSE: compiled writer is missing or incomplete\n' >&2
  exit 68
fi
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2
  exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do
  if [[ ! -f "$SOURCE_RUN/$name" ]]; then
    printf 'REFUSE: missing prepared input %s\n' "$name" >&2
    exit 68
  fi
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"

(
  cd "$TARGET_RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  status=${PIPESTATUS[0]}
  printf 'wall_seconds %s\n' "$((SECONDS-started))" >>run.user.time.log
  if [[ "$status" -ne 0 ]]; then
    printf 'REFUSE: NEMO process failed\n' >&2
    exit 69
  fi
  if ! grep -Fxq 'STOP 0' run.user.stdout.log; then
    printf 'REFUSE: NEMO lacks STOP 0\n' >&2
    exit 69
  fi
)
if [[ ! -f "$TARGET_RUN/$RECORD" ]]; then
  printf 'REFUSE: developed FCT record missing\n' >&2
  exit 70
fi
printf '%s %s %s\n' "$(sha256sum "$TARGET_RUN/$RECORD" | awk '{print $1}')" \
  "$COMMIT" "$RECORD" >"$TARGET_RUN/$RECORD.stamp"

source_list=$(mktemp /tmp/r153-source-list.XXXXXX)
target_list=$(mktemp /tmp/r153-target-list.XXXXXX)
find "$SOURCE_RUN" -maxdepth 1 -type f \
  \( -name 'oracle*.bin' -o -name '*_restart.nc' -o -name 'mesh_mask.nc' \) \
  -printf '%f\n' | sort >"$source_list"
find "$TARGET_RUN" -maxdepth 1 -type f \
  \( -name 'oracle*.bin' -o -name '*_restart.nc' -o -name 'mesh_mask.nc' \) \
  ! -name "$RECORD" -printf '%f\n' | sort >"$target_list"
if ! cmp -s "$source_list" "$target_list"; then
  printf 'REFUSE: inherited file registry differs from source run\n' >&2
  diff -u "$source_list" "$target_list" >&2 || true
  exit 71
fi
admit_existing
