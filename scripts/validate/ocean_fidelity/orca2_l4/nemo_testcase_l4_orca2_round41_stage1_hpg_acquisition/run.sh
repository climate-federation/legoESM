#!/usr/bin/env bash
# ORCA2 round-41 acquisition: stage-1 HPG operands and statement boundaries.
# USER-EXECUTED ONLY. The default --run mode invokes makenemo and mpirun.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-41 ORCA2 stage-1 HPG acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing|--plant-layout) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing|--plant-layout]\n' "$0" >&2; exit 64 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R41HPG1
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round40/acquisition/orca1ice_rhs_families_ranked_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round41/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_stage1_hpg_ranked_np2
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED=$TARGET_ROOT/BLD/ppsrc/nemo/dynhpg.f90
readonly EXPECTED_SOURCE_SHA256=ef20ab2187c16e4a28ec54987a3d5ae2029b99104547919c9cd805facb71a7df
readonly EXPECTED_CPP_SHA256=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly EXPECTED_SOURCE_BINARY_SHA256=7eb2c8f6173997fab27e75a1f151eda08e37f2f00db7361ac1a27c701560ff10
readonly EXPECTED_DECK_MANIFEST_SHA256=51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9
readonly EXPECTED_INPUT_MANIFEST_SHA256=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5
readonly EXPECTED_RECORD_BYTES=32119476

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/dynhpg_round41.patch
readonly GATE=$here/../nemo_testcase_l4_orca2_round41_stage1_hpg_gate.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round41.md
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
fi
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing pinned %s: %s\n' "$label" "$path" >&2; exit 65; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 66;
  }
}

for path in "$PATCH" "$GATE" "$PREREG" "$SOURCE_ROOT/MY_SRC/dynhpg.F90" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$SOURCE_ROOT/BLD/bin/nemo.exe" \
  "$SOURCE_RUN/nemo" "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing input %s\n' "$path" >&2; exit 64; }
done
pin "$EXPECTED_SOURCE_SHA256" "$SOURCE_ROOT/MY_SRC/dynhpg.F90" 'source dynhpg'
pin "$EXPECTED_CPP_SHA256" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source cpp keys'
pin "$EXPECTED_SOURCE_BINARY_SHA256" "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source build binary'
pin "$EXPECTED_SOURCE_BINARY_SHA256" "$SOURCE_RUN/nemo" 'source run binary'
pin "$EXPECTED_DECK_MANIFEST_SHA256" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin "$EXPECTED_INPUT_MANIFEST_SHA256" "$SOURCE_RUN/input_files.sha256" 'input manifest'
for pattern in 'number of the last time step.*nn_itend *= *10' 'Tiling \(T\) or not \(F\).*ln_tile *= *F' 'Vector form: 2nd order centered scheme.*ln_dynadv_vec *= *T'; do
  grep -Eq "$pattern" "$SOURCE_RUN/ocean.output" || {
    printf 'REFUSE: source run lacks resolved row: %s\n' "$pattern" >&2; exit 65;
  }
done

removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
if [[ "$removed" -ne 0 ]]; then
  printf 'REFUSE: writer patch removes %s source lines\n' "$removed" >&2
  exit 66
fi
dry=$(mktemp -d /tmp/orca2-r41-stage1-hpg-source.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$SOURCE_ROOT/MY_SRC/dynhpg.F90" "$dry/dynhpg.F90"
patch -s --fuzz=0 -p0 -d "$dry" <"$PATCH"

check_layout() {
  local source=$1
  [[ "$(grep -Fc "r41_magic = 'NEMO_L4_R41HPG1'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "Kmm == 1 .AND. Krhs == 3" "$source")" -eq 4 ]] &&
  [[ "$(grep -Fc "l4_canon_3d(r41_zhpi_u,'U')" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "l4_canon_3d(r41_zuap_u,'U')" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "l4_canon_3d(r41_sum_u,'U')" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "ORCA2_R41_STAGE1_HPG_DUMP" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "L2_RK_STAGE2_HPG_LITERAL_DUMP" "$source")" -eq 1 ]]
}
if ! check_layout "$dry/dynhpg.F90"; then
  printf 'REFUSE: writer layout is incomplete\n' >&2; exit 66
fi
reader_size=$("$PY" -c "import runpy; print(runpy.run_path('$GATE')['EXPECTED_SIZE'])")
if [[ "$reader_size" -ne "$EXPECTED_RECORD_BYTES" ]]; then
  printf 'REFUSE: writer/reader sizes disagree: %s versus %s\n' "$EXPECTED_RECORD_BYTES" "$reader_size" >&2; exit 66
fi
if [[ "$MODE" == --plant-layout ]]; then
  planted=$dry/dynhpg.planted.F90
  cp "$dry/dynhpg.F90" "$planted"
  sed -i '0,/l4_canon_3d(r41_zhpi_u/d' "$planted"
  if check_layout "$planted"; then
    printf 'REFUSE: layout plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED: writer layout\n'
  exit 69
fi

syntax=$(mktemp -d /tmp/orca2-r41-stage1-hpg-syntax.XXXXXX)
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$dry/dynhpg.F90" -o "$syntax/dynhpg.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/dynhpg.f90"
printf 'SYNTAX_PROOF_PASS dynhpg.f90\n'
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND41_STAGE1_HPG_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local rank record plant
  for rank in 0000 0001; do
    record=$TARGET_RUN/oracle_stage1_hpg_ranked_kt00000001_r${rank}.bin
    [[ -f "$record" && "$(stat -c %s "$record")" -eq "$EXPECTED_RECORD_BYTES" ]] || {
      printf 'REFUSE: rank %s HPG stream has wrong or absent byte count\n' "$rank" >&2; exit 70;
    }
  done
  for name in ORCA2_00000010_restart_0000.nc ORCA2_00000010_restart_0001.nc \
    ORCA2_00000010_restart_ice_0000.nc ORCA2_00000010_restart_ice_0001.nc \
    oracle_rhs_families_ranked_kt00000001_r0000.bin \
    oracle_rhs_families_ranked_kt00000001_r0001.bin \
    oracle_rkstage2_hpg_literal_kt00000001.bin; do
    [[ -f "$TARGET_RUN/$name" ]] || { printf 'REFUSE: missing calibration artifact %s\n' "$name" >&2; exit 70; }
  done
  [[ -f "$TARGET_RUN/producer_commit.txt" ]] || { printf 'REFUSE: missing producer commit\n' >&2; exit 70; }
  [[ "$(cat "$TARGET_RUN/producer_commit.txt")" == "$COMMIT" ]] || {
    printf 'REFUSE: producer commit differs from current clean tree\n' >&2; exit 70;
  }
  cmp -s "$TARGET_RUN/oracle_rkstage2_hpg_literal_kt00000001.bin" \
    "$SOURCE_RUN/oracle_rkstage2_hpg_literal_kt00000001.bin" || {
    printf 'REFUSE: inherited stage-2 HPG literal stream moved\n' >&2; exit 70;
  }
  gate() {
    "$PY" "$GATE" --root "$TARGET_RUN" --baseline-root "$SOURCE_RUN" \
      --repo "$REPO" --expect-commit "$COMMIT" "$@"
  }
  for plant in header truncation swapped-rank parent-byte restart-byte; do
    if gate --plant "$plant" --output "$TARGET_RUN/round41_${plant}_plant.json" \
      >"$TARGET_RUN/round41_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round41_${plant}_plant.log" || {
      printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71;
    }
  done
  gate --output "$TARGET_RUN/round41_stage1_hpg_admission.json"
  for rank in 0000 0001; do
    record=oracle_stage1_hpg_ranked_kt00000001_r${rank}.bin
    printf '%s %s %s\n' "$(sha256sum "$TARGET_RUN/$record" | awk '{print $1}')" \
      "$COMMIT" "$record" >"$TARGET_RUN/$record.stamp"
  done
  (
    cd "$TARGET_RUN"
    sha256sum oracle_stage1_hpg_ranked_kt00000001_r*.bin \
      oracle_stage1_hpg_ranked_kt00000001_r*.bin.stamp \
      oracle_rhs_families_ranked_kt00000001_r*.bin \
      oracle_rkstage2_hpg_literal_kt00000001.bin ORCA2_00000010_restart*.nc \
      round41_*_plant.json round41_*_plant.log round41_stage1_hpg_admission.json \
      >round41_outputs.sha256
  )
  printf 'ORCA2_ROUND41_STAGE1_HPG_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$BINARY" && -x "$TARGET_RUN/nemo" && -f "$COMPILED" ]] || {
    printf 'REFUSE: existing target lacks binary or compiled source\n' >&2; exit 68;
  }
  cmp -s "$BINARY" "$TARGET_RUN/nemo" || {
    printf 'REFUSE: existing staged binary differs from target build\n' >&2; exit 68;
  }
  check_layout "$COMPILED" || { printf 'REFUSE: existing compiled writer changed\n' >&2; exit 68; }
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 64;
}
for mount in /tmp "$(dirname "$EVIDENCE")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2; exit 67;
  }
done

manifest=$(mktemp -d /tmp/orca2-r41-stage1-hpg-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$manifest/source_cfg.sha256"
sha256sum "$PATCH" "$GATE" "$PREREG" "$SOURCE_ROOT/BLD/ppsrc/nemo/dynhpg.f90" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$PATCH"
cmp -s "$dry/dynhpg.F90" "$TARGET_ROOT/MY_SRC/dynhpg.F90" || {
  printf 'REFUSE: target writer differs from syntax-proved source\n' >&2; exit 68;
}
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
[[ -x "$BINARY" ]] || { printf 'REFUSE: target build produced no executable\n' >&2; exit 68; }
check_layout "$COMPILED" || { printf 'REFUSE: compiled writer is incomplete\n' >&2; exit 68; }
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol is present in acquisition binary\n' >&2; exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir -p "$EVIDENCE"
mkdir "$TARGET_RUN"
while read -r digest name; do
  [[ -f "$SOURCE_RUN/$name" || -L "$SOURCE_RUN/$name" ]] || {
    printf 'REFUSE: deck manifest source absent: %s\n' "$name" >&2; exit 68;
  }
  cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done <"$SOURCE_RUN/deck_files.sha256"
while read -r digest name; do
  [[ -f "$SOURCE_RUN/$name" || -L "$SOURCE_RUN/$name" ]] || {
    printf 'REFUSE: input manifest source absent: %s\n' "$name" >&2; exit 68;
  }
  cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$COMPILED" "$TARGET_RUN/compiled_dynhpg.f90"
cp "$manifest"/* "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  sha256sum -c deck_files.sha256 >/dev/null
  sha256sum -c input_files.sha256 >/dev/null
)

(
  cd "$TARGET_RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  mpi_rc=${pipe_rc[0]}
  tee_rc=${pipe_rc[1]:-0}
  printf 'MPIRUN_RC=%d\nWALL_SECONDS=%d\nRUN_FINISHED_UTC=%s\n' "$mpi_rc" \
    "$((SECONDS-started))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  [[ "$mpi_rc" -eq 0 && "$tee_rc" -eq 0 ]] || {
    printf 'REFUSE: NEMO run failed (mpi=%s tee=%s)\n' "$mpi_rc" "$tee_rc" >&2; exit 69;
  }
  grep -Fxq 'STOP 0' run.user.stdout.log || {
    printf 'REFUSE: completed run lacks STOP 0\n' >&2; exit 69;
  }
  printf 'RUN DONE\n' >>run.user.time.log
)
admit
