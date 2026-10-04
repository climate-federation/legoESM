#!/usr/bin/env bash
# Operator-executed ORCA2 rung-0 step-1..10 growth acquisition.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-144 growth acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing|--plant-layout|--plant-deck|--plant-toolchain) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing|--plant-layout|--plant-deck|--plant-toolchain]\n' "$0" >&2; exit 63 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4_R96SPG
readonly TARGET_CFG=ORCA2_OMIP_L4_R144INITIAL
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round96/acquisition/orca2_rung0_spgts_ranked_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round144/acquisition
readonly CALIBRATION_RUN=$EVIDENCE/orca2_rung0_initial_growth_calibration_10step_np2
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_initial_growth_10step_np2
readonly SOURCE_STPRK3_SHA=31f9d62f7ac06b84bc3ef6b5ec94da19695014663c671c31bfc21496e42d1e93
readonly SOURCE_TRAADV_SHA=ddd33bdec420246ba43419599da9c33e89148cf032ad7f9ba3fab3d9542625ff
readonly SOURCE_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly SOURCE_BINARY_SHA=7a65083c8f9a625a394844c9bfb0fcb927d666f34136a010de33f28685bab7f6
readonly SOURCE_NML_SHA=5192355842d9233d8356ab87b4ff8b65eac539e66dc135f77a07e26451d360e8
readonly DECK_MANIFEST_SHA=0e40688deddd7a22f8a6a7105ebc623c80e335d7bea5d88a80702bf447148312
readonly INPUT_MANIFEST_SHA=395ae3e2dad969bc7ef5c94e20f5f441875e17d8a87b73407d9cab5e0c3a2b51
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/growth_round144.patch
readonly WRITER=$here/l4_r144_growth.F90
readonly GATE=$here/check_record.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round144.md

pin() {
  local expected=$1 path=$2 label=$3 observed
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  observed=$(sha256sum "$path" | awk '{print $1}')
  [[ "$observed" == "$expected" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

check_layout() {
  local stp=$1 tra=$2 writer=$3
  [[ "$(grep -Fc 'USE l4_r144_growth, ONLY : r144_growth_entry, r144_growth_baro' "$stp")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r144_growth_entry' "$stp")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r144_growth_baro' "$stp")" -eq 1 ]] &&
  [[ "$(grep -Fc 'USE l4_r144_growth, ONLY : r144_growth_stage1' "$tra")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r144_growth_stage1' "$tra")" -eq 1 ]] &&
  [[ "$(grep -Fc "STATUS='NEW'" "$writer")" -eq 1 ]] &&
  [[ "$(grep -Fc 'kt >= 1 .AND. kt <= 10' "$writer")" -eq 1 ]] &&
  [[ "$(grep -Fc 'WRITE(record_unit) 1, kt, mpprank' "$writer")" -eq 1 ]]
}

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed producer tree\n' >&2; exit 64;
}
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

for path in "$0" "$PATCH" "$WRITER" "$GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
for path in "$SOURCE_ROOT/MY_SRC/stprk3.F90" "$SOURCE_ROOT/MY_SRC/traadv.F90" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$SOURCE_ROOT/BLD/bin/nemo.exe" \
  "$SOURCE_RUN/nemo" "$SOURCE_RUN/namelist_cfg" "$SOURCE_RUN/deck_files.sha256" \
  "$SOURCE_RUN/input_files.sha256"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing pinned source %s\n' "$path" >&2; exit 64; }
done
pin "$SOURCE_STPRK3_SHA" "$SOURCE_ROOT/MY_SRC/stprk3.F90" 'source stprk3'
pin "$SOURCE_TRAADV_SHA" "$SOURCE_ROOT/MY_SRC/traadv.F90" 'source traadv'
pin "$SOURCE_CPP_SHA" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source cpp keys'
pin "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source build binary'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'source run binary'
pin "$SOURCE_NML_SHA" "$SOURCE_RUN/namelist_cfg" 'source rung-0 namelist'
pin "$DECK_MANIFEST_SHA" "$SOURCE_RUN/deck_files.sha256" 'source deck manifest'
pin "$INPUT_MANIFEST_SHA" "$SOURCE_RUN/input_files.sha256" 'source input manifest'
(cd "$SOURCE_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)

mkdir -p "$EVIDENCE"
bash -n "$0"
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
[[ "$removed" -eq 0 ]] || { printf 'REFUSE: recorder patch removes %s source lines\n' "$removed" >&2; exit 66; }
dry=$(mktemp -d /tmp/orca2-r144-growth.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$SOURCE_ROOT/MY_SRC/stprk3.F90" "$SOURCE_ROOT/MY_SRC/traadv.F90" "$WRITER" "$dry/"
patch -s --fuzz=0 -p0 -d "$dry" <"$PATCH"

if [[ "$MODE" == --plant-layout ]]; then
  sed -i '/CALL r144_growth_stage1/d' "$dry/traadv.F90"
  if check_layout "$dry/stprk3.F90" "$dry/traadv.F90" "$dry/l4_r144_growth.F90"; then
    printf 'REFUSE: source-layout plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED source-layout\n'
  exit 69
fi
check_layout "$dry/stprk3.F90" "$dry/traadv.F90" "$dry/l4_r144_growth.F90" || {
  printf 'REFUSE: compiled recorder call order is incomplete\n' >&2; exit 66;
}

cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/l4_r144_growth.F90" -o "$dry/l4_r144_growth.f90"
"$FC" -c -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$dry" \
  "$dry/l4_r144_growth.f90" -o "$dry/l4_r144_growth.o"
for name in stprk3 traadv; do
  cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
    -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$dry/$name.F90" -o "$dry/$name.f90"
  "$FC" -fsyntax-only -ffree-line-length-none -I "$dry" -I "$SOURCE_ROOT/BLD/inc" -J "$dry" "$dry/$name.f90"
done
printf 'SYNTAX_PROOF_PASS l4_r144_growth.f90 stprk3.f90 traadv.f90\n'
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND144_INITIAL_GROWTH_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

verify_toolchain() {
  local manifest=$1 expected
  [[ -f "$manifest" && "$(wc -l <"$manifest")" -eq 5 ]] || {
    printf 'REFUSE: producer content manifest is missing or has wrong cardinality\n' >&2; exit 70;
  }
  while read -r expected path; do
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { printf 'REFUSE: malformed producer digest\n' >&2; exit 70; }
    pin "$expected" "$path" "recorded producer content $(basename "$path")"
  done <"$manifest"
}

if [[ "$MODE" == --plant-toolchain ]]; then
  plant_manifest=$dry/toolchain.sha256
  sha256sum "$0" "$PATCH" "$WRITER" "$GATE" "$PREREG" >"$plant_manifest"
  sed -i '1s/^[0-9a-f]/z/' "$plant_manifest"
  if (verify_toolchain "$plant_manifest") >/dev/null 2>&1; then
    printf 'REFUSE: producer-content plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED producer-content\n'
  exit 69
fi

make_growth_namelist() {
  local source=$1 target=$2
  cp "$source" "$target"
  cmp -s "$source" "$target"
}

if [[ "$MODE" == --plant-deck ]]; then
  make_growth_namelist "$SOURCE_RUN/namelist_cfg" "$dry/namelist_cfg"
  sed -i 's/ln_zdfevd   = .true./ln_zdfevd   = .false./' "$dry/namelist_cfg"
  expected=$dry/expected_namelist_cfg
  make_growth_namelist "$SOURCE_RUN/namelist_cfg" "$expected"
  if cmp -s "$expected" "$dry/namelist_cfg"; then
    printf 'REFUSE: hidden-deck-delta plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED hidden-deck-delta\n'
  exit 69
fi

admit() {
  verify_toolchain "$TARGET_RUN/toolchain.sha256"
  verify_toolchain "$CALIBRATION_RUN/toolchain.sha256"
  check_layout "$TARGET_RUN/compiled_stprk3.f90" "$TARGET_RUN/compiled_traadv.f90" \
    "$TARGET_ROOT/MY_SRC/l4_r144_growth.F90" || {
      printf 'REFUSE: existing compiled recorder layout moved\n' >&2; exit 70;
    }
  expected=$dry/expected_namelist_cfg
  make_growth_namelist "$SOURCE_RUN/namelist_cfg" "$expected"
  cmp -s "$expected" "$TARGET_RUN/namelist_cfg" || {
    printf 'REFUSE: growth namelist differs outside the run protocol\n' >&2; exit 70;
  }
  cmp -s "$SOURCE_RUN/namelist_cfg" "$CALIBRATION_RUN/namelist_cfg" || {
    printf 'REFUSE: calibration namelist moved\n' >&2; exit 70;
  }
  local plant
  for plant in header field-name field-rank field-dims truncation missing-step nonfinite swapped-rank restart-byte; do
    if "$PY" "$GATE" --record-root "$TARGET_RUN" --calibration-root "$CALIBRATION_RUN" \
      --baseline "$SOURCE_RUN" --plant "$plant" >"$EVIDENCE/${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$EVIDENCE/${plant}_plant.log" || {
      printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71;
    }
  done
  "$PY" "$GATE" --record-root "$TARGET_RUN" --calibration-root "$CALIBRATION_RUN" \
    --baseline "$SOURCE_RUN" --output "$EVIDENCE/initial_growth_record_admission.json"
  (cd "$EVIDENCE" && sha256sum initial_growth_record_admission.json *_plant.log >round144_initial_outputs.sha256)
  printf 'ORCA2_ROUND144_INITIAL_GROWTH_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$TARGET_ROOT/BLD/bin/nemo.exe" && -x "$TARGET_RUN/nemo" && -x "$CALIBRATION_RUN/nemo" ]] || {
    printf 'REFUSE: existing target lacks built/staged binaries\n' >&2; exit 68;
  }
  cmp -s "$TARGET_ROOT/BLD/bin/nemo.exe" "$TARGET_RUN/nemo" || { printf 'REFUSE: target binary mismatch\n' >&2; exit 68; }
  cmp -s "$TARGET_RUN/nemo" "$CALIBRATION_RUN/nemo" || { printf 'REFUSE: calibration/growth binaries differ\n' >&2; exit 68; }
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" && ! -e "$CALIBRATION_RUN" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 68;
}
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || { printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67; }
done

manifest=$(mktemp -d /tmp/orca2-r144-growth-manifest.XXXXXX)
sha256sum "$0" "$PATCH" "$WRITER" "$GATE" "$PREREG" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l4_r144_growth.F90"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$PATCH"
cmp -s "$dry/stprk3.F90" "$TARGET_ROOT/MY_SRC/stprk3.F90" || { printf 'REFUSE: staged stprk3 differs from syntax proof\n' >&2; exit 68; }
cmp -s "$dry/traadv.F90" "$TARGET_ROOT/MY_SRC/traadv.F90" || { printf 'REFUSE: staged traadv differs from syntax proof\n' >&2; exit 68; }
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 69; }
if nm -D "$BINARY" | grep -q '_ZGV'; then printf 'REFUSE: vector-math symbol present\n' >&2; exit 69; fi

stage_run() {
  local destination=$1 mode=$2 name
  mkdir "$destination"
  while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$destination/$name"; done <"$SOURCE_RUN/deck_files.sha256"
  while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$destination/$name"; done <"$SOURCE_RUN/input_files.sha256"
  cp "$SOURCE_RUN/input_files.sha256" "$destination/"
  if [[ "$mode" == growth ]]; then
    make_growth_namelist "$SOURCE_RUN/namelist_cfg" "$destination/namelist_cfg" || {
      printf 'REFUSE: could not preserve the 10-step run protocol\n' >&2; exit 69;
    }
  fi
  (cd "$destination" && while read -r _ name; do sha256sum "$name"; done <"$SOURCE_RUN/deck_files.sha256" >deck_files.sha256)
  cp "$BINARY" "$destination/nemo"
  cp "$TARGET_ROOT/BLD/ppsrc/nemo/stprk3.f90" "$destination/compiled_stprk3.f90"
  cp "$TARGET_ROOT/BLD/ppsrc/nemo/traadv.f90" "$destination/compiled_traadv.f90"
  cp "$manifest/toolchain.sha256" "$destination/"
  (cd "$destination" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
  (
    cd "$destination"
    export ORCA2_R144_GROWTH_DIR=$destination
    [[ "$ORCA2_R144_GROWTH_DIR" == /* && -d "$ORCA2_R144_GROWTH_DIR" ]] || {
      printf 'REFUSE: recorder directory is not absolute/pre-created\n' >&2; exit 69;
    }
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
    set +e
    mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
    pipe_rc=("${PIPESTATUS[@]}")
    set -e
    [[ "${pipe_rc[0]}" -eq 0 && "${pipe_rc[1]}" -eq 0 ]] || {
      printf 'REFUSE: NEMO/mpirun pipeline failed: %s %s\n' "${pipe_rc[0]}" "${pipe_rc[1]}" >&2; exit 72;
    }
    grep -Fxq 'STOP 0' run.user.stdout.log || { printf 'REFUSE: NEMO lacks STOP 0\n' >&2; exit 72; }
  )
}

stage_run "$CALIBRATION_RUN" calibration
stage_run "$TARGET_RUN" growth
admit
