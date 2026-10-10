#!/usr/bin/env bash
# Operator-executed OMT-1 substep-1 vector pre-LBC acquisition.
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: round-213 vector acquisition failed at line %s (exit %s)\n' \
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
readonly SOURCE_CFG=ORCA2_OMIP_L4_R210OMT1_P3
readonly TARGET_CFG=ORCA2_OMIP_L4_R213VECPRE
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round211/acquisition/orca2_omt1_frames_8step_a_np2
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round213/acquisition
readonly TARGET_RUN=$EVIDENCE/orca2_omt1_vector_pre_lbc_8step_np2
readonly WORK_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work
readonly SOURCE_DYNSPG_SHA=c1542a517a40627aeea099c0955728cda85eb49b3ec5aa9d98068f371dbcb74c
readonly SOURCE_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly SOURCE_BINARY_SHA=5b82a3254c40f71186af159b93cba419709ccf49cf4172ad3d44440b8fb1d895
readonly DECK_MANIFEST_SHA=42ba42807d38b1e41a06e8f5589be16d56a1e1b426ab851a2d7f29076dd66463
readonly INPUT_MANIFEST_SHA=395ae3e2dad969bc7ef5c94e20f5f441875e17d8a87b73407d9cab5e0c3a2b51
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly RUNNER=$here/run.sh
readonly PATCH=$here/dynspg_ts_round213.patch
readonly GATE=$here/check_record.py
readonly PREREG=$REPO/docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round213.md
readonly SOURCE_MANIFEST=$REPO/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round209_omt1_frames_acquisition/source_files.sha256

pin() {
  local digest=$1 path=$2 label=$3
  [[ -f "$path" ]] || { printf 'REFUSE: missing %s: %s\n' "$label" "$path" >&2; exit 64; }
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]] || {
    printf 'REFUSE: %s SHA-256 changed: %s\n' "$label" "$path" >&2; exit 65;
  }
}

artifact_digest() {
  sha256sum "$RUNNER" "$PATCH" "$GATE" "$PREREG" "$SOURCE_MANIFEST" | \
    sha256sum | awk '{print $1}'
}

check_layout() {
  local source=$1 update_line writer_line drag_line
  [[ "$(grep -Fc "r213_magic = 'NEMO_L4_R213VV1'" "$source")" -eq 1 ]] || return 1
  [[ "$(grep -Fc 'ORCA2_R213_VECTOR_PRE_LBC_DUMP' "$source")" -eq 1 ]] || return 1
  [[ "$(grep -Fc "r213_name = 'zv_frc'" "$source")" -eq 1 ]] || return 1
  [[ "$(grep -Fc "r213_name = 'ssvmask'" "$source")" -eq 1 ]] || return 1
  [[ "$(grep -Fc "r213_name = 'va_pre_lbc'" "$source")" -eq 1 ]] || return 1
  update_line=$(grep -nF 'va_e(ji,jj) = (                                 vn_e(ji,jj)' "$source" | head -1 | cut -d: -f1)
  writer_line=$(grep -nF "r213_magic = 'NEMO_L4_R213VV1'" "$source" | cut -d: -f1)
  drag_line=$(grep -nF '!jth implicit bottom friction:' "$source" | cut -d: -f1)
  [[ -n "$update_line" && -n "$writer_line" && -n "$drag_line" ]] || return 1
  [[ "$update_line" -lt "$writer_line" && "$writer_line" -lt "$drag_line" ]]
}

validate_source_inventory() {
  local expected=$1 actual
  actual=$(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 -type f -printf '%f\n' | sort)
  [[ "$actual" == "$expected" ]] || {
    printf 'REFUSE: admitted record source inventory changed\n' >&2
    return 1
  }
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
  git diff --quiet HEAD -- "$path" || {
    printf 'REFUSE: acquisition artifact differs from HEAD: %s\n' "$path" >&2; exit 64;
  }
done
for path in "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$SOURCE_ROOT/BLD/bin/nemo.exe" \
  "$SOURCE_RUN/nemo" "$SOURCE_RUN/deck_files.sha256" \
  "$SOURCE_RUN/input_files.sha256" "$SOURCE_RUN/ocean.output"; do
  [[ -f "$path" ]] || { printf 'REFUSE: missing pinned input %s\n' "$path" >&2; exit 64; }
done
pin "$SOURCE_DYNSPG_SHA" "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" 'source dynspg_ts'
pin "$SOURCE_CPP_SHA" "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" 'source cpp keys'
pin "$SOURCE_BINARY_SHA" "$SOURCE_ROOT/BLD/bin/nemo.exe" 'source build binary'
pin "$SOURCE_BINARY_SHA" "$SOURCE_RUN/nemo" 'source run binary'
pin "$DECK_MANIFEST_SHA" "$SOURCE_RUN/deck_files.sha256" 'deck manifest'
pin "$INPUT_MANIFEST_SHA" "$SOURCE_RUN/input_files.sha256" 'input manifest'
expected_sources=$(awk '{print $2}' "$SOURCE_MANIFEST" | sort)
validate_source_inventory "$expected_sources"
(cd "$SOURCE_ROOT/MY_SRC" && sha256sum -c "$SOURCE_MANIFEST" >/dev/null)
(cd "$SOURCE_RUN" && sha256sum -c deck_files.sha256 >/dev/null && sha256sum -c input_files.sha256 >/dev/null)
grep -Eq 'Vector form: 2nd order centered scheme.*ln_dynadv_vec *= *T' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source deck does not resolve vector momentum advection\n' >&2; exit 65;
}
grep -Eq 'Free surface with time splitting.*ln_dynspg_ts *= *T' "$SOURCE_RUN/ocean.output" || {
  printf 'REFUSE: source deck does not resolve split-explicit SSH\n' >&2; exit 65;
}
grep -Fxq 'STOP 0' "$SOURCE_RUN/run.user.stdout.log" || {
  printf 'REFUSE: admitted OMT-1 source run did not complete\n' >&2; exit 65;
}

removed=$(awk '/^--- / {next} /^-/ {n++} END {print n+0}' "$PATCH")
[[ "$removed" -eq 0 ]] || {
  printf 'REFUSE: writer patch removes %s source lines\n' "$removed" >&2; exit 66;
}
mkdir -p "$EVIDENCE" "$WORK_ROOT"
bash -n "$RUNNER"
"$PY" -m py_compile "$GATE"
scratch=$(mktemp -d "$WORK_ROOT/orca2-r213-vector.XXXXXX")
printf 'temporary source proof directory (retained): %s\n' "$scratch"
cp "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" "$scratch/dynspg_ts.F90"
patch -s --fuzz=0 -p0 -d "$scratch" <"$PATCH"
check_layout "$scratch/dynspg_ts.F90" || {
  printf 'REFUSE: patched writer layout is incomplete or misplaced\n' >&2; exit 66;
}
if [[ "$MODE" == --plant-layout ]]; then
  sed -i "/r213_name = 'va_pre_lbc'/d" "$scratch/dynspg_ts.F90"
  if check_layout "$scratch/dynspg_ts.F90"; then
    printf 'REFUSE: layout plant stayed green\n' >&2; exit 69
  fi
  printf 'STATUS PLANT-FIRED layout\n'
  exit 69
fi

cpp -Dkey_nosignedzero -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
  -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
  "$scratch/dynspg_ts.F90" -o "$scratch/dynspg_ts.f90"
"$FC" -fsyntax-only -ffree-line-length-none -I "$SOURCE_ROOT/BLD/inc" \
  -J "$scratch" "$scratch/dynspg_ts.f90"
printf 'SYNTAX_PROOF_PASS dynspg_ts.f90\n'
if validate_source_inventory "${expected_sources}"$'\n__source_inventory_plant__.F90' \
  >"$EVIDENCE/source_inventory_plant.log" 2>&1; then
  printf 'REFUSE: source-inventory plant stayed green\n' >&2; exit 69
fi
grep -Fq 'REFUSE: admitted record source inventory changed' \
  "$EVIDENCE/source_inventory_plant.log"
printf 'STATUS PLANT-FIRED source-inventory\n' >>"$EVIDENCE/source_inventory_plant.log"
if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND213_VECTOR_PRE_LBC_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

admit() {
  local plant record expected_digest
  expected_digest=$(artifact_digest)
  [[ -f "$TARGET_RUN/producer_content.sha256" ]] || {
    printf 'REFUSE: target lacks producer content stamp\n' >&2; exit 70;
  }
  [[ "$(cat "$TARGET_RUN/producer_content.sha256")" == "$expected_digest" ]] || {
    printf 'REFUSE: producer artifact content differs from current tree\n' >&2; exit 70;
  }
  [[ "$(find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_r213_vector_rank????_kt00000001_jn001.bin' | wc -l)" -eq 2 ]] || {
    printf 'REFUSE: expected exactly two rank vector records\n' >&2; exit 70;
  }
  for record in "$TARGET_RUN"/oracle_r213_vector_rank????_kt00000001_jn001.bin; do
    printf '%s %s\n' "$(sha256sum "$record" | awk '{print $1}')" \
      "$(basename "$record")" >"$record.stamp"
  done
  for plant in header field-name field-dims truncation swapped-rank nonfinite \
    frame-byte restart-byte; do
    if "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" --plant "$plant" \
      >"$TARGET_RUN/round213_${plant}_plant.log" 2>&1; then
      printf 'REFUSE: %s plant stayed green\n' "$plant" >&2; exit 71
    fi
    grep -Fq 'STATUS PLANT-FIRED' "$TARGET_RUN/round213_${plant}_plant.log" || {
      printf 'REFUSE: %s plant lacks firing marker\n' "$plant" >&2; exit 71;
    }
  done
  "$PY" "$GATE" --root "$TARGET_RUN" --baseline "$SOURCE_RUN" \
    --output "$TARGET_RUN/round213_vector_pre_lbc_admission.json"
  (cd "$TARGET_RUN" && sha256sum oracle_r213_vector_rank*.bin \
    oracle_r213_vector_rank*.bin.stamp ORCA2_00000008_restart_????.nc \
    oracle_r84_frame_*.bin producer_content.sha256 round213_*_plant.log \
    round213_vector_pre_lbc_admission.json >round213_outputs.sha256)
  printf 'ORCA2_ROUND213_VECTOR_PRE_LBC_ACQUISITION_PASS %s\n' "$TARGET_RUN"
}

if [[ "$MODE" == --admit-existing ]]; then
  [[ -x "$TARGET_ROOT/BLD/bin/nemo.exe" && -x "$TARGET_RUN/nemo" ]] || {
    printf 'REFUSE: existing target lacks binary\n' >&2; exit 68;
  }
  cmp -s "$TARGET_ROOT/BLD/bin/nemo.exe" "$TARGET_RUN/nemo" || {
    printf 'REFUSE: staged and built binaries differ\n' >&2; exit 68;
  }
  check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" || {
    printf 'REFUSE: compiled vector writer moved\n' >&2; exit 68;
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

manifest=$(mktemp -d "$WORK_ROOT/orca2-r213-vector-manifest.XXXXXX")
artifact_digest >"$manifest/producer_content.sha256"
sha256sum "$RUNNER" "$PATCH" "$GATE" "$PREREG" "$SOURCE_MANIFEST" \
  "$SOURCE_ROOT/MY_SRC/dynspg_ts.F90" >"$manifest/toolchain.sha256"

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
cmp -s "$scratch/dynspg_ts.F90" "$TARGET_ROOT/MY_SRC/dynspg_ts.F90" || {
  printf 'REFUSE: staged writer differs from syntax-proved source\n' >&2; exit 68;
}
touch "$TARGET_ROOT/MY_SRC/"*.F90
./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
[[ -x "$BINARY" ]] || { printf 'REFUSE: build produced no executable\n' >&2; exit 69; }
check_layout "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" || {
  printf 'REFUSE: compiled vector writer is incomplete or misplaced\n' >&2; exit 69;
}
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present\n' >&2; exit 69
fi

mkdir "$TARGET_RUN"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/deck_files.sha256"
while read -r _ name; do cp -a "$SOURCE_RUN/$name" "$TARGET_RUN/$name"; done <"$SOURCE_RUN/input_files.sha256"
cp "$SOURCE_RUN/deck_files.sha256" "$SOURCE_RUN/input_files.sha256" "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$TARGET_ROOT/BLD/ppsrc/nemo/dynspg_ts.f90" "$TARGET_RUN/compiled_dynspg_ts.f90"
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
