#!/usr/bin/env bash
# Operator-only passive acquisition. Default mode is read-only preflight; this
# agent never invokes --run because the round contract forbids running NEMO.
set -Eeuo pipefail

readonly MODE=${1:---preflight-only}
case "$MODE" in
  --preflight-only|--run|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R326HMI
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R328ZDF
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_dynzdf_kt1_np2
readonly SOURCE_DYNZDF=$NEMO_ROOT/src/OCE/DYN/dynzdf.F90
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly MEM_MIN_GB=150
readonly RUN_TIMEOUT_S=900
readonly SOURCE_DYNZDF_SHA=4489773bb4cafe46123cfd658b1561188215c9ce0db11f6859fb27cb019bb174
readonly SOURCE_BINARY_SHA=20376f40a78cb1a5491838ff112bfbe738178e4da8da1e43d5bd5c3884c72934

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/dynzdf_round328.patch
readonly WRITER=$here/dynzdf_round328_writer.F90
readonly GATE=$here/../nemo_testcase_l4_orca2_round328_dynzdf_gate.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round328_dynzdf_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l4_orca2_round328.md
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe

refuse() { printf 'REFUSE: %s\n' "$*" >&2; exit 65; }
plant_must_refuse() {
  local label=$1; shift
  if "$@"; then refuse "preflight plant stayed green: $label"; fi
  printf 'PREFLIGHT_PLANT_FIRED %s\n' "$label"
}
mem_ok() { (( $(free -g | awk '$1 == "Mem:" {print $7}') >= $1 )); }
free_ok() {
  local p=$1 need=${2:-8388608}
  while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done
  (( $(df -Pk "$p" | awk 'NR == 2 {print $4}') >= need ))
}
echo_has() { grep -E "(^|[[:space:]])$2[[:space:]]*=[[:space:]]*$3([[:space:]]|$)" "$1" >/dev/null; }
check_echo() {
  echo_has "$1" nn_it000 1 && echo_has "$1" nn_itend 10 && echo_has "$1" ln_tile F \
    && echo_has "$1" ln_drgimp T && echo_has "$1" ln_drgice_imp T \
    && echo_has "$1" ln_dynadv_vec T && echo_has "$1" ln_dynspg_ts T
}
check_no_removed_lines() { [[ $(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$1") -eq 0 ]]; }

cd "$REPO"
[[ -z $(git status --porcelain --untracked-files=all) ]] || refuse 'acquisition requires a clean committed tree'
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for path in "$PATCH" "$WRITER" "$GATE" "$TEST" "$PREREG" "$SOURCE_DYNZDF" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" "$SOURCE_RUN/ocean.output" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256"; do
  [[ -f "$path" ]] || refuse "missing input $path"
done
[[ $(sha256sum "$SOURCE_DYNZDF" | awk '{print $1}') == "$SOURCE_DYNZDF_SHA" ]] || refuse 'source dynzdf changed'
[[ $(sha256sum "$SOURCE_ROOT/BLD/bin/nemo.exe" | awk '{print $1}') == "$SOURCE_BINARY_SHA" ]] || refuse 'source build binary changed'
[[ $(sha256sum "$SOURCE_RUN/nemo" | awk '{print $1}') == "$SOURCE_BINARY_SHA" ]] || refuse 'source run binary changed'
check_echo "$SOURCE_RUN/ocean.output" || refuse 'source run lacks the frozen D150/vector/split-explicit selectors'
check_no_removed_lines "$PATCH" || refuse 'observer patch removes or replaces a source line'
grep -E "^$REFERENCE_CFG[[:space:]]" "$NEMO_ROOT/cfgs/ref_cfgs.txt" >/dev/null || refuse 'reference config is not registered'
free_ok "$EVIDENCE" || refuse 'under 8 GB free at evidence root'
free_ok "$NEMO_ROOT" || refuse 'under 8 GB free at NEMO root'
mem_ok "$MEM_MIN_GB" || refuse "under $MEM_MIN_GB GB available memory"

mkdir -p "$(dirname "$EVIDENCE")"
scratch=$(mktemp -d "$(dirname "$EVIDENCE")/.r328-preflight.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
cp "$SOURCE_DYNZDF" "$scratch/dynzdf.F90"
cp "$WRITER" "$scratch/dynzdf_round328_writer.F90"
patch -s --fuzz=0 -p0 -d "$scratch" <"$PATCH"
[[ $(grep -Fc 'CALL r328_' "$scratch/dynzdf.F90") -eq 18 ]] || refuse 'writer call census changed'
mkdir "$scratch/mod"
cpp -Dkey_nosignedzero -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$scratch/dynzdf_round328_writer.F90" >"$scratch/writer.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$scratch/mod" "$scratch/writer.f90"
cpp -Dkey_nosignedzero -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$scratch/dynzdf.F90" >"$scratch/dynzdf.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$scratch/mod" -I "$SOURCE_ROOT/BLD/inc" \
  -J "$scratch/mod" "$scratch/dynzdf.f90"
"$PY" -m pytest -q -p no:cacheprovider "$TEST"
plant_must_refuse memory mem_ok 999999
plant_must_refuse disk free_ok "$NEMO_ROOT" 999999999999999
if [[ "$MODE" != --admit-existing ]]; then
  [[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || refuse 'target config or evidence already exists'
fi
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND328_DYNZDF_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local source name plant
  [[ $(cat "$TARGET_RUN/producer_commit.txt") == "$COMMIT" ]] || refuse 'producer commit changed'
  while IFS= read -r -d '' source; do
    name=$(basename "$source")
    cmp -s "$source" "$TARGET_RUN/$name" || refuse "inherited output changed: $name"
  done < <(find "$SOURCE_RUN" -maxdepth 1 -type f \( -name 'oracle_*.bin' -o -name 'ORCA2_*.nc' -o -name 'mesh_mask_*.nc' \) -print0)
  for plant in header field-order truncation stamp avm bottom top solve; do
    if "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$COMMIT" --plant "$plant" \
      >"$TARGET_RUN/round328_${plant}_plant.log" 2>&1; then
      refuse "$plant plant stayed green"
    fi
  done
  "$PY" "$GATE" --record-dir "$TARGET_RUN" --expect-commit "$COMMIT" \
    --output "$TARGET_RUN/round328_dynzdf_admission.json"
  ( cd "$TARGET_RUN" && sha256sum oracle_dynzdf_*.bin round328_dynzdf_admission.json \
      round328_*_plant.log >round328_outputs.sha256 )
  printf 'ORCA2_ROUND328_DYNZDF_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$BINARY" && -x "$TARGET_RUN/nemo" ]] || refuse 'existing target lacks its binary'
  cmp -s "$BINARY" "$TARGET_RUN/nemo" || refuse 'staged binary differs'
  admit
  exit 0
fi

manifest=$scratch/manifest; mkdir "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$PATCH" "$WRITER" "$GATE" "$TEST" "$PREREG" "$SOURCE_DYNZDF" >"$manifest/toolchain.sha256"
cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$SOURCE_DYNZDF" "$TARGET_ROOT/MY_SRC/dynzdf.F90"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/dynzdf_round328_writer.F90"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || refuse 'build produced no executable'
grep 'NEMO_L4_R328ZDF' "$TARGET_ROOT/BLD/ppsrc/nemo/dynzdf_round328_writer.f90" >/dev/null || refuse 'compiled writer is absent'

# Same-program census: only dynzdf plus the new writer may differ from R326.
while IFS= read -r -d '' source; do
  name=${source#"$SOURCE_ROOT/BLD/ppsrc/"}
  case "$name" in nemo/dynzdf.f90) continue ;; esac
  cmp -s "$source" "$TARGET_ROOT/BLD/ppsrc/$name" || refuse "preprocessed unit changed: $name"
done < <(find "$SOURCE_ROOT/BLD/ppsrc" -type f -print0)
while IFS= read -r -d '' source; do
  name=${source#"$SOURCE_ROOT/BLD/obj/"}
  case "$name" in dynzdf.o) continue ;; esac
  cmp -s "$source" "$TARGET_ROOT/BLD/obj/$name" || refuse "object changed: $name"
done < <(find "$SOURCE_ROOT/BLD/obj" -type f -print0)

mkdir -p "$EVIDENCE"; mkdir "$TARGET_RUN"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
( cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null )
(
  cd "$TARGET_RUN"
  mem_ok "$MEM_MIN_GB" || refuse "under $MEM_MIN_GB GB before run"
  free -g >run.user.memory.log
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  timeout --kill-after=60 "$RUN_TIMEOUT_S" mpirun -np 2 --oversubscribe ./nemo >run.user.stdout.log 2>&1
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
grep 'STOP 0' "$TARGET_RUN/run.user.stdout.log" >/dev/null || refuse 'NEMO did not report STOP 0'
check_echo "$TARGET_RUN/ocean.output" || refuse 'run selectors changed'
admit
