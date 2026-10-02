#!/usr/bin/env bash
# Operator-executed ORCA2 hierarchy rung-0 slow-forcing acquisition.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-93 slow acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing|--plant-layout) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing|--plant-layout]\n' "$0" >&2; exit 63 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4_R92RHS
readonly TARGET_CFG=ORCA2_OMIP_L4_R93SLOW
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round92/acquisition/orca2_rung0_rhs_ranked_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round93/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_slow_ranked_10step_np2
readonly SOURCE_STP_SHA=8a39f2aa68738519e849286860b3b1faba1eb81d8a722902975a02297a4a5b2d
readonly SOURCE_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly SOURCE_BINARY_SHA=86117fc6afa5916208cc1f6de3bd154b9c92a2176c6f761526f7eebb1871ca44
readonly DECK_MANIFEST_SHA=92f2a73eeb3b9989b6f390519e3f0cab900b122194259f2988cad672e1819677
readonly INPUT_MANIFEST_SHA=395ae3e2dad969bc7ef5c94e20f5f441875e17d8a87b73407d9cab5e0c3a2b51
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/stp2d_round93.patch
readonly WRITER=$here/l4_r93_slow_frames.F90
readonly GATE=$here/check_record.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round93.md

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed producer tree\n' >&2; exit 64;
}
readonly COMMIT=$(git rev-parse HEAD)
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

for path in "$0" "$PATCH" "$WRITER" "$GATE" "$PREREG"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
for path in "$SOURCE_ROOT/MY_SRC/stp2d.F90" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$SOURCE_RUN/ocean.output"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing pinned input %s\n' "$path" >&2; exit 64; }
done
pin "$SOURCE_STP_SHA" "$SOURCE_ROOT/MY_SRC/stp2d.F90" 'source stp2d'
pin "$SOURCE_CPP_SHA" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source cpp keys'
pin "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source build binary'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'source run binary'
pin "$DECK_MANIFEST_SHA" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin "$INPUT_MANIFEST_SHA" "$SOURCE_RUN/input_files.sha256" 'input manifest'
(cd "$SOURCE_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
grep -Eq 'Vector form: 2nd order centered scheme.*ln_dynadv_vec *= *T' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source deck does not resolve vector-invariant C2\n' >&2; exit 65;
}
grep -Eq 'Free surface with time splitting.*ln_dynspg_ts *= *T' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source deck does not resolve split-explicit free surface\n' >&2; exit 65;
}
grep -Eq 'Patm gradient added in ocean.*ln_apr_dyn *= *F' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: atmospheric-pressure branch is not off\n' >&2; exit 65;
}
grep -Eq 'ice embedded into ocean.*ln_ice_embd *= *F' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: embedded-ice branch is not off\n' >&2; exit 65;
}

mkdir -p "$EVIDENCE"
bash -n "$0"
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
[[ "$removed" -eq 0 ]] || { printf 'REFUSE: writer patch removes %s source lines\n' "$removed" >&2; exit 66; }
dry=$(mktemp -d /tmp/orca2-r93-slow.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$dry"
cp "$SOURCE_ROOT/MY_SRC/stp2d.F90" "$dry/stp2d.F90"
cp "$WRITER" "$dry/l4_r93_slow_frames.F90"
patch -s --fuzz=0 -p0 -d "$dry" <"$PATCH"

check_layout() {
  local source=$1
  [[ "$(grep -Fc 'CALL r93_slow_start( kt, Kbb, Kaa, Krhs )' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r93_slow_put_pair( kt, 'depth'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r93_slow_put_pair( kt, 'drag'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r93_slow_put_pair( kt, 'wind'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r93_slow_put_pair( kt, 'final'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r93_slow_finish( kt )' "$source")" -eq 1 ]]
}
check_layout "$dry/stp2d.F90" || { printf 'REFUSE: slow writer layout is incomplete\n' >&2; exit 66; }
if [[ "$MODE" == --plant-layout ]]; then
  sed -i "/CALL r93_slow_put_pair( kt, 'depth'/d" "$dry/stp2d.F90"
  if check_layout "$dry/stp2d.F90"; then printf 'REFUSE: layout plant stayed green\n' >&2; exit 69; fi
  printf 'STATUS PLANT-FIRED layout\n'
  exit 69
fi

cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$dry/l4_r93_slow_frames.F90" -o "$dry/l4_r93_slow_frames.f90"
"$FC" -c -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$dry" \
  "$dry/l4_r93_slow_frames.f90" -o "$dry/l4_r93_slow_frames.o"
cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" "$dry/stp2d.F90" -o "$dry/stp2d.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$dry" -I "$SOURCE_ROOT/BLD/inc" -J "$dry" "$dry/stp2d.f90"
printf 'SYNTAX_PROOF_PASS l4_r93_slow_frames.f90 stp2d.f90\n'
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND93_RUNG0_SLOW_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant record
  [[ "$(cat "$TARGET_RUN/producer_commit.txt")" == "$COMMIT" ]] || {
    printf 'REFUSE: producer commit differs from current tree\n' >&2; exit 70;
  }
  [[ "$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r93_slow_rank????_kt00000001.bin' | wc -l)" -eq 2 ]] || {
    printf 'REFUSE: expected exactly two rank slow records\n' >&2; exit 70;
  }
  for record in "$TARGET_RUN"/oracle_r93_slow_rank????_kt00000001.bin; do
    printf '%s %s %s\n' "$(sha256sum "$record" | awk '{print $1}')" "$COMMIT" "$(basename "$record")" >"$record.stamp"
  done
  for plant in header field-name truncation swapped-rank restart-byte; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" --plant "$plant" \
      >"$TARGET_RUN/round93_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round93_${plant}_plant.log" || {
      printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71;
    }
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" \
    --output "$TARGET_RUN/round93_slow_admission.json"
  (cd "$TARGET_RUN" && sha256sum oracle_r93_slow_rank*.bin oracle_r93_slow_rank*.bin.stamp \
    ORCA2_000000??_restart_????.nc round93_*_plant.log round93_slow_admission.json \
    >round93_outputs.sha256)
  printf 'ORCA2_ROUND93_RUNG0_SLOW_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$TARGET_ROOT/BLD/bin/nemo.exe" && -x "$TARGET_RUN/nemo" ]] || {
    printf 'REFUSE: existing target lacks binary\n' >&2; exit 68;
  }
  cmp -s "$TARGET_ROOT/BLD/bin/nemo.exe" "$TARGET_RUN/nemo" || {
    printf 'REFUSE: staged and built binaries differ\n' >&2; exit 68;
  }
  check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/stp2d.f90" || {
    printf 'REFUSE: compiled writer layout moved\n' >&2; exit 68;
  }
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 68;
}
mkdir -p "$EVIDENCE"
for mount in /tmp "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || { printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67; }
done

manifest=$(mktemp -d /tmp/orca2-r93-slow-manifest.XXXXXX)
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
sha256sum "$PATCH" "$WRITER" "$GATE" "$PREREG" "$SOURCE_ROOT/BLD/ppsrc/nemo/stp2d.f90" >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l4_r93_slow_frames.F90"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$PATCH"
cmp -s "$dry/stp2d.F90" "$TARGET_ROOT/MY_SRC/stp2d.F90" || {
  printf 'REFUSE: target source differs from syntax-proved source\n' >&2; exit 68;
}
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 69; }
check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/stp2d.f90" || { printf 'REFUSE: compiled writer layout is incomplete\n' >&2; exit 69; }
if nm -D "$BINARY" | grep -q '_ZGV'; then printf 'REFUSE: vector-math symbol present\n' >&2; exit 69; fi

mkdir "$TARGET_RUN"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/stp2d.f90" "$TARGET_RUN/compiled_stp2d.f90"
cp "$manifest"/* "$TARGET_RUN/"
(cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
(
  cd "$TARGET_RUN"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  started=$SECONDS
  set +e
  mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  mpi_rc=${pipe_rc[0]}; tee_rc=${pipe_rc[1]:-0}
  printf 'MPIRUN_RC=%d\nWALL_SECONDS=%d\nRUN_FINISHED_UTC=%s\n' "$mpi_rc" "$((SECONDS-started))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  [[ "$mpi_rc" -eq 0 && "$tee_rc" -eq 0 ]] || { printf 'REFUSE: NEMO run failed (mpi=%s tee=%s)\n' "$mpi_rc" "$tee_rc" >&2; exit 69; }
  grep -Fxq 'STOP 0' run.user.stdout.log || { printf 'REFUSE: completed run lacks STOP 0\n' >&2; exit 69; }
  printf 'RUN DONE\n' >>run.user.time.log
)
admit
