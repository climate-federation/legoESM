#!/usr/bin/env bash
# ORCA2 round-62 acquisition: split stage-2 vector advection into KEG and ZAD.
# The operator runs this file; the agent does not invoke mpirun in the sandbox.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-62 vector split failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round62/acquisition_vector_split
readonly TARGET_RUN=$EVIDENCE/orca1ice_vector_advection_split_np2
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly DYN_PATCH=$here/dynadv_round62.patch
readonly STG_PATCH=$here/stprk3_stg_round62.patch
readonly WRITER=$here/dynadv_round62_writer.F90
readonly GATE=$here/../nemo_testcase_l4_orca2_round62_vector_split_admission.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round62_vector_advection.md
readonly SOURCE_DYN=$NEMO_ROOT/src/OCE/DYN/dynadv.F90
readonly SOURCE_STG=$SOURCE_ROOT/MY_SRC/stprk3_stg.F90

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2; exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 65; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 66;
  }
}

for path in "$DYN_PATCH" "$STG_PATCH" "$WRITER" "$GATE" "$PREREG" \
  "$SOURCE_DYN" "$SOURCE_STG" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing input %s\n' "$path" >&2; exit 64; }
done
pin d6802c22ce6a3b0307b556cbb20326f04f916711e2288f68e7a67dc7beb94a6a "$SOURCE_DYN" 'source dynadv'
pin 2b1636336acc5a5e3326291801c217f263f8ccc3ca875dbfec0d3b789d911e5b "$SOURCE_STG" 'source stprk3_stg'
pin 0d4360984a193ff9bce2cfbaa13a45dbd26fcd18b215e0159c95254075830630 "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source build binary'
pin 0d4360984a193ff9bce2cfbaa13a45dbd26fcd18b215e0159c95254075830630 "$SOURCE_RUN/nemo" 'source run binary'
pin 51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9 "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin 3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5 "$SOURCE_RUN/input_files.sha256" 'input manifest'
grep -Eq 'number of the last time step.*nn_itend *= *10' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source run is not ten steps\n' >&2; exit 65;
}
grep -Eq 'Vector form: 2nd order centered scheme.*ln_dynadv_vec *= *T' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source run does not execute vector advection\n' >&2; exit 65;
}
grep -Eq 'nn_dynkeg.*= *0' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source run does not execute C2 KEG\n' >&2; exit 65;
}
grep -Eq 'Tiling \(T\) or not \(F\).*ln_tile *= *F' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source run enables unsupported tiling\n' >&2; exit 65;
}

for patch in "$DYN_PATCH" "$STG_PATCH"; do
  removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$patch")
  [[ "$removed" -eq 0 ]] || { printf 'REFUSE: additions-only patch removes %s lines: %s\n' "$removed" "$patch" >&2; exit 66; }
done
dry=$(mktemp -d /tmp/orca2-r62-vector-split.XXXXXX)
cp "$SOURCE_DYN" "$dry/dynadv.F90"
cp "$SOURCE_STG" "$dry/stprk3_stg.F90"
cp "$WRITER" "$dry/dynadv_round62_writer.F90"
patch -s --fuzz=0 -p0 -d "$dry" <"$DYN_PATCH"
patch -s --fuzz=0 -p0 -d "$dry" <"$STG_PATCH"
[[ "$(grep -Fc 'CALL dyn_keg' "$dry/dynadv.F90")" -eq 1 ]] || { printf 'REFUSE: KEG call count changed\n' >&2; exit 66; }
[[ "$(grep -Fc 'CALL dyn_zad' "$dry/dynadv.F90")" -eq 1 ]] || { printf 'REFUSE: ZAD call count changed\n' >&2; exit 66; }
[[ "$(grep -Fc 'R62_3D(' "$dry/dynadv_round62_writer.F90")" -eq 15 ]] || { printf 'REFUSE: writer 3-D field census changed\n' >&2; exit 66; }
[[ "$(grep -Fc 'R62_2D(' "$dry/dynadv_round62_writer.F90")" -eq 8 ]] || { printf 'REFUSE: writer 2-D field census changed\n' >&2; exit 66; }

mkdir -p "$EVIDENCE"
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || { printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2; exit 67; }
done

syntax=$(mktemp -d /tmp/orca2-r62-vector-syntax.XXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/dynadv_round62_writer.F90" -o "$syntax/dynadv_round62_writer.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" \
  "$syntax/dynadv_round62_writer.f90"
printf 'SYNTAX_PROOF_PASS dynadv_round62_writer.f90\n'
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND62_VECTOR_SPLIT_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant
  [[ -f "$TARGET_RUN/producer_commit.txt" ]] || { printf 'REFUSE: missing producer commit\n' >&2; exit 70; }
  [[ "$(cat "$TARGET_RUN/producer_commit.txt")" == "$COMMIT" ]] || { printf 'REFUSE: producer commit changed\n' >&2; exit 70; }
  for plant in header field-order truncation restart stamp; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" \
      --expect-commit "$COMMIT" --plant "$plant" >"$TARGET_RUN/round62_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -q 'STATUS PLANT-FIRED' "$TARGET_RUN/round62_${plant}_plant.log"
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" \
    --expect-commit "$COMMIT" --output "$TARGET_RUN/round62_vector_split_admission.json"
  ( cd "$TARGET_RUN" && sha256sum oracle_vector_adv_split_*.bin ORCA2_00000010_restart*.nc \
      round62_vector_split_admission.json round62_*_plant.log >round62_outputs.sha256 )
  printf 'ORCA2_ROUND62_VECTOR_SPLIT_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$BINARY" && -x "$TARGET_RUN/nemo" ]] || { printf 'REFUSE: existing target lacks binary\n' >&2; exit 68; }
  cmp -s "$BINARY" "$TARGET_RUN/nemo" || { printf 'REFUSE: staged binary differs\n' >&2; exit 68; }
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 64;
}
manifest=$(mktemp -d /tmp/orca2-r62-vector-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$DYN_PATCH" "$STG_PATCH" "$WRITER" "$GATE" "$PREREG" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/dynadv.f90" "$SOURCE_ROOT/BLD/ppsrc/nemo/dynkeg.f90" \
  "$SOURCE_ROOT/BLD/ppsrc/nemo/dynzad.f90" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$SOURCE_DYN" "$TARGET_ROOT/MY_SRC/dynadv.F90"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/dynadv_round62_writer.F90"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$DYN_PATCH"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$STG_PATCH"
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 68; }
grep -q 'NEMO_L4_VADV_1' "$TARGET_ROOT/BLD/ppsrc/nemo/dynadv_round62_writer.f90" || {
  printf 'REFUSE: compiled writer is absent\n' >&2; exit 68;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then printf 'REFUSE: vector math symbol present\n' >&2; exit 68; fi

mkdir "$TARGET_RUN"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r digest name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
( cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null )

(
  cd "$TARGET_RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  [[ "${pipe_rc[0]}" -eq 0 ]] || { printf 'REFUSE: mpirun exited %s\n' "${pipe_rc[0]}" >&2; exit 69; }
  printf 'wall_seconds %s\nRUN_FINISHED_UTC=%s\nRUN_DONE\n' "$((SECONDS-started))" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
)
grep -q 'STOP 0' "$TARGET_RUN/run.user.stdout.log" || { printf 'REFUSE: NEMO did not report STOP 0\n' >&2; exit 69; }
for name in ORCA2_00000010_restart_0000.nc ORCA2_00000010_restart_0001.nc \
  ORCA2_00000010_restart_ice_0000.nc ORCA2_00000010_restart_ice_0001.nc \
  oracle_rkstage2_terms_kt00000001.bin; do
  cmp -s "$SOURCE_RUN/$name" "$TARGET_RUN/$name" || { printf 'REFUSE: calibration changed: %s\n' "$name" >&2; exit 69; }
done
admit
