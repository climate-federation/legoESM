#!/usr/bin/env bash
# Operator-executed ORCA2 rung-0 kt=8 slow-forcing operand acquisition.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-169 slow8 acquisition failed at line %s (exit %s)\n' \
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
readonly SOURCE_CFG=ORCA2_OMIP_L4_R166SPG8
readonly TARGET_CFG=ORCA2_OMIP_L4_R169SLOW8
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round166/acquisition/orca2_rung0_spgts_kt8_ranked_10step_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round169/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_rung0_slow8_ranked_10step_np2
readonly SOURCE_STP_SHA=71ac8a0fec29fe5d7738b8de93f517dda5602eb497836f621d2d5e3ce866534c
readonly SOURCE_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly SOURCE_BINARY_SHA=23b20aa74dff4713ce225a3a8891a0b41a8110db79342a86db8fddff38337237
readonly DECK_MANIFEST_SHA=0e40688deddd7a22f8a6a7105ebc623c80e335d7bea5d88a80702bf447148312
readonly INPUT_MANIFEST_SHA=395ae3e2dad969bc7ef5c94e20f5f441875e17d8a87b73407d9cab5e0c3a2b51
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PATCH=$here/stp2d_round169.patch
readonly WRITER=$here/l4_r169_slow8.F90
readonly GATE=$here/check_record.py
readonly RUN_SH=$here/run.sh
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round169.md
readonly ARTIFACTS=("$PATCH" "$WRITER" "$GATE" "$RUN_SH" "$PREREG")

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
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

for path in "${ARTIFACTS[@]}"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
for path in "$SOURCE_ROOT/MY_SRC/stp2d.F90" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
  "$SOURCE_ROOT/BLD/bin/nemo.exe" "$SOURCE_RUN/nemo" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" \
  "$SOURCE_RUN/ocean.output"; do
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
  printf 'REFUSE: source deck is not vector-invariant C2\n' >&2; exit 65;
}
grep -Eq 'Free surface with time splitting.*ln_dynspg_ts *= *T' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source deck does not use split-explicit free surface\n' >&2; exit 65;
}

mkdir -p "$EVIDENCE"
bash -n "$RUN_SH"
removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
[[ "$removed" -eq 0 ]] || {
  printf 'REFUSE: writer patch removes %s source lines\n' "$removed" >&2; exit 66;
}
scratch=$(mktemp -d /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/orca2-r169-slow8.XXXXXX)
printf 'temporary source proof directory (retained): %s\n' "$scratch"
cp "$SOURCE_ROOT/MY_SRC/stp2d.F90" "$scratch/stp2d.F90"
cp "$WRITER" "$scratch/l4_r169_slow8.F90"
( cd "$scratch" && git apply --recount "$PATCH" )

check_layout() {
  local source=$1
  [[ "$(grep -Fc 'CALL r169_start( kt, Kbb, Kaa, Krhs )' "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r169_put3( kt, 'rhs_u'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r169_put2( kt, 'depth_u'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r169_put2( kt, 'drag_u'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r169_put2( kt, 'wind_u'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc "CALL r169_put2( kt, 'final_u'" "$source")" -eq 1 ]] &&
  [[ "$(grep -Fc 'CALL r169_finish( kt )' "$source")" -eq 1 ]]
}
check_layout "$scratch/stp2d.F90" || {
  printf 'REFUSE: slow8 writer layout is incomplete\n' >&2; exit 66;
}
if [[ "$MODE" == --plant-layout ]]; then
  sed -i "/CALL r169_put2( kt, 'depth_u'/d" "$scratch/stp2d.F90"
  if check_layout "$scratch/stp2d.F90"; then
    printf 'REFUSE: layout plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED layout\n'
  exit 69
fi

cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$scratch/l4_r169_slow8.F90" -o "$scratch/l4_r169_slow8.f90"
"$FC" -c -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" -J "$scratch" \
  "$scratch/l4_r169_slow8.f90" -o "$scratch/l4_r169_slow8.o"
cpp -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$scratch/stp2d.F90" -o "$scratch/stp2d.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$scratch" \
  -I "$SOURCE_ROOT/BLD/inc" -J "$scratch" "$scratch/stp2d.f90"
printf 'SYNTAX_PROOF_PASS l4_r169_slow8.f90 stp2d.f90\n'
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND169_SLOW8_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

verify_content_manifest() {
  [[ -f "$TARGET_RUN/toolchain.sha256" ]] || {
    printf 'REFUSE: existing target lacks toolchain.sha256\n' >&2; exit 70;
  }
  sha256sum -c "$TARGET_RUN/toolchain.sha256" >/dev/null || {
    printf 'REFUSE: acquisition content pin moved\n' >&2; exit 70;
  }
}

admit() {
  local plant record expected_stamp toolchain_digest
  verify_content_manifest
  toolchain_digest=$(sha256sum "$TARGET_RUN/toolchain.sha256" | awk '{print $1}')
  [[ "$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r169_slow8_rank????_kt00000008.bin' | wc -l)" -eq 2 ]] || {
    printf 'REFUSE: expected exactly two rank-complete slow8 records\n' >&2; exit 70;
  }
  for record in "$TARGET_RUN"/oracle_r169_slow8_rank????_kt00000008.bin; do
    expected_stamp="$(sha256sum "$record" | awk '{print $1}') $toolchain_digest $(basename "$record")"
    [[ "$(cat "$record.stamp")" == "$expected_stamp" ]] || {
      printf 'REFUSE: record stamp moved: %s\n' "$record" >&2; exit 70;
    }
  done
  for plant in header field-name field-dims truncation swapped-rank restart-byte; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" --plant "$plant" \
      >"$TARGET_RUN/round169_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round169_${plant}_plant.log" || {
      printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71;
    }
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" \
    --output "$TARGET_RUN/round169_slow8_admission.json"
  (cd "$TARGET_RUN" && sha256sum oracle_r169_slow8_rank*.bin \
    oracle_r169_slow8_rank*.bin.stamp ORCA2_000000??_restart_????.nc \
    round169_*_plant.log round169_slow8_admission.json >round169_outputs.sha256)
  printf 'ORCA2_ROUND169_SLOW8_ACQUISITION_PASS %s\n' "$TARGET_RUN"
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
for mount in /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67;
  }
done

manifest=$(mktemp -d /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/orca2-r169-manifest.XXXXXX)
sha256sum "${ARTIFACTS[@]}" "$SOURCE_ROOT/MY_SRC/stp2d.F90" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$SOURCE_ROOT/BLD/bin/nemo.exe" \
  "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l4_r169_slow8.F90"
( cd "$TARGET_ROOT/MY_SRC" && git apply --recount "$PATCH" )
cmp -s "$scratch/stp2d.F90" "$TARGET_ROOT/MY_SRC/stp2d.F90" || {
  printf 'REFUSE: target source differs from syntax-proved source\n' >&2; exit 68;
}
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 69; }
check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/stp2d.f90" || {
  printf 'REFUSE: compiled writer layout is incomplete\n' >&2; exit 69;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2; exit 69
fi

mkdir "$TARGET_RUN"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/stp2d.f90" "$TARGET_RUN/compiled_stp2d.f90"
cp "$manifest/toolchain.sha256" "$TARGET_RUN/"
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
  [[ "$mpi_rc" -eq 0 && "$tee_rc" -eq 0 ]] || {
    printf 'REFUSE: NEMO run failed (mpi=%s tee=%s)\n' "$mpi_rc" "$tee_rc" >&2; exit 69;
  }
  grep -Fxq 'STOP 0' run.user.stdout.log || {
    printf 'REFUSE: completed run lacks STOP 0\n' >&2; exit 69;
  }
  printf 'RUN DONE\n' >>run.user.time.log
  toolchain_digest=$(sha256sum toolchain.sha256 | awk '{print $1}')
  for record in oracle_r169_slow8_rank????_kt00000008.bin; do
    printf '%s %s %s\n' "$(sha256sum "$record" | awk '{print $1}')" \
      "$toolchain_digest" "$record" >"$record.stamp"
  done
)
admit
