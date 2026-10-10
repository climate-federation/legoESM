#!/usr/bin/env bash
# Operator-executed recovery of the OMT-1 pre-LBC record's field dimensions.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-214 vector-header acquisition failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 63 ;;
esac

export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4_R213VECPRE
readonly TARGET_CFG=ORCA2_OMIP_L4_R214VECPREV2
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round213/acquisition/orca2_omt1_vector_pre_lbc_8step_np2
readonly BASELINE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round211/acquisition/orca2_omt1_frames_8step_a_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round214/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_omt1_vector_pre_lbc_headerfix_8step_np2
readonly WORK_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work
readonly SOURCE_DYNSPG_SHA=c5745bfff2e99e521e4786f87f1c2d3c4d9a5c4a283a10290ff08a5b3311ffff
readonly SOURCE_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly SOURCE_BINARY_SHA=61ca1ac3832a9bec5057993318d9333f538884263516a1208f435d8c32cb8479
readonly DECK_MANIFEST_SHA=42ba42807d38b1e41a06e8f5589be16d56a1e1b426ab851a2d7f29076dd66463
readonly INPUT_MANIFEST_SHA=395ae3e2dad969bc7ef5c94e20f5f441875e17d8a87b73407d9cab5e0c3a2b51
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly RUNNER=$here/run.sh
readonly PATCH=$here/dynspg_ts_round214.patch
readonly GATE=$REPO/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round213_vector_pre_lbc_acquisition/check_record.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round214.md
readonly SOURCE_MANIFEST=$REPO/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round209_omt1_frames_acquisition/source_files.sha256

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

check_layout() {
  local source=$1
  [[ "$(grep -Fc "r213_name = 'zv_frc'" "$source")" -eq 1 ]] || return 1
  [[ "$(grep -Fc 'WRITE(r213_unit) r213_name, 2, SIZE(zv_frc,1), SIZE(zv_frc,2), 1' "$source")" -eq 1 ]] || return 1
  [[ "$(grep -Fc "r213_name = 'va_pre_lbc'" "$source")" -eq 1 ]] || return 1
}

check_producer_manifest() {
  local path digest base
  [[ -f "$TARGET_RUN/producer_content.sha256" ]] || return 1
  for path in "$RUNNER" "$PATCH" "$GATE" "$PREREG" "$SOURCE_MANIFEST"; do
    digest=$(sha256sum "$path" | awk '{print $1}')
    base=$(basename "$path")
    grep -Fxq "$digest  $base" "$TARGET_RUN/producer_content.sha256" || return 1
  done
}

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed producer tree\n' >&2; exit 64;
}
export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

for path in "$RUNNER" "$PATCH" "$GATE" "$PREREG" "$SOURCE_MANIFEST"; do
  git ls-files --error-unmatch "${path#"$REPO"/}" >/dev/null || {
    printf 'REFUSE: acquisition artifact is not committed: %s\n' "$path" >&2; exit 64;
  }
done
pin "$SOURCE_DYNSPG_SHA" "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" 'round-213 writer source'
pin "$SOURCE_CPP_SHA" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source cpp keys'
pin "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source binary'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'source run binary'
pin "$DECK_MANIFEST_SHA" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin "$INPUT_MANIFEST_SHA" "$SOURCE_RUN/input_files.sha256" 'input manifest'
(cd "$SOURCE_ROOT/MY_SRC" && grep -v '  dynspg_ts.F90$' "$SOURCE_MANIFEST" | sha256sum -c - >/dev/null)
(cd "$SOURCE_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
grep -Fxq 'STOP 0' "$SOURCE_RUN/run.user.stdout.log" || {
  printf 'REFUSE: source record run did not complete\n' >&2; exit 65;
}

mkdir -p "$EVIDENCE" "$WORK_ROOT"
bash -n "$RUNNER"
"$PY" -m py_compile "$GATE"
scratch=$(mktemp -d "$WORK_ROOT/orca2-r214-vector.XXXXXX")
cp "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$scratch/dynspg_ts.F90"
patch -s --fuzz=0 -p0 -d "$scratch" <"$PATCH"
check_layout "$scratch/dynspg_ts.F90" || {
  printf 'REFUSE: repaired writer layout is incomplete\n' >&2; exit 66;
}
cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$scratch/dynspg_ts.F90" -o "$scratch/dynspg_ts.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$scratch" "$scratch/dynspg_ts.f90"
printf 'SYNTAX_PROOF_PASS round214 dynspg_ts.f90\n'
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND214_VECTOR_HEADER_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant
  check_producer_manifest || { printf 'REFUSE: producer content manifest moved\n' >&2; exit 70; }
  for plant in header rk-level field-name field-dims truncation swapped-rank nonfinite frame-byte restart-byte; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$BASELINE_RUN" --plant "$plant" \
      >"$TARGET_RUN/round214_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round214_${plant}_plant.log" || {
      printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71;
    }
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$BASELINE_RUN" \
    --output "$TARGET_RUN/round214_vector_pre_lbc_admission.json"
  (cd "$TARGET_RUN" && sha256sum oracle_r213_vector_rank*.bin \
    ORCA2_00000008_restart_????.nc oracle_r84_frame_*.bin producer_content.sha256 \
    round214_*_plant.log round214_vector_pre_lbc_admission.json >round214_outputs.sha256)
  printf 'ORCA2_ROUND214_VECTOR_HEADER_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$TARGET_ROOT/BLD/bin/nemo.exe" && -x "$TARGET_RUN/nemo" ]] || {
    printf 'REFUSE: existing target lacks binary\n' >&2; exit 68;
  }
  cmp -s "$TARGET_ROOT/BLD/bin/nemo.exe" "$TARGET_RUN/nemo" || {
    printf 'REFUSE: staged and built binaries differ\n' >&2; exit 68;
  }
  check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" || {
    printf 'REFUSE: compiled repaired writer moved\n' >&2; exit 68;
  }
  admit
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: target config or run directory already exists\n' >&2; exit 68;
}
for mount in "$WORK_ROOT" "$EVIDENCE" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GiB free\n' "$mount" >&2; exit 67;
  }
done

cd "$NEMO_ROOT"
./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
while IFS= read -r -d '' source; do cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"; done \
  < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$PATCH"
cmp -s "$scratch/dynspg_ts.F90" "$TARGET_ROOT/MY_SRC/dynspg_ts.F90" || {
  printf 'REFUSE: staged writer differs from syntax-proved source\n' >&2; exit 68;
}
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 69; }
check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" || {
  printf 'REFUSE: compiled repaired writer moved\n' >&2; exit 69;
}

mkdir "$TARGET_RUN"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" "$TARGET_RUN/compiled_dynspg_ts.f90"
for path in "$RUNNER" "$PATCH" "$GATE" "$PREREG" "$SOURCE_MANIFEST"; do
  printf '%s  %s\n' "$(sha256sum "$path" | awk '{print $1}')" "$(basename "$path")"
done >"$TARGET_RUN/producer_content.sha256"
(cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)

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
