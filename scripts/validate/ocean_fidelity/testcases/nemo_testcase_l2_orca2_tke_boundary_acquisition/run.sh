#!/usr/bin/env bash
set -Eeuo pipefail

refuse_unexpected() {
  local status=$?
  printf 'REFUSE: ORCA2 TKE boundary acquisition failed unexpectedly at line %s (exit %s)\n' \
    "${BASH_LINENO[0]:-unknown}" "$status" >&2
  exit "$status"
}
trap refuse_unexpected ERR

verify_exp00_copy() {
  local source_root=$1
  local target_root=$2
  local source target name

  for source in "$source_root"/*; do
    name=$(basename "$source")
    target=$target_root/$name
    if [[ -L "$source" ]]; then
      if [[ ! -L "$target" ]] || \
          [[ "$(readlink "$source")" != "$(readlink "$target")" ]]; then
        printf 'REFUSE: file-by-file EXP00 copy differs for %s\n' "$name" >&2
        exit 68
      fi
    elif [[ -f "$source" ]]; then
      if [[ ! -f "$target" || -L "$target" ]] || ! cmp -s "$source" "$target"; then
        printf 'REFUSE: file-by-file EXP00 copy differs for %s\n' "$name" >&2
        exit 68
      fi
    else
      printf 'REFUSE: file-by-file EXP00 copy differs for %s\n' "$name" >&2
      exit 68
    fi
  done
}

# USER-EXECUTED ACQUISITION ONLY. --run invokes makenemo and mpirun.
# The construction follows the round-56/59/64/101 source-card clone pattern:
# clone the Phase-2v reference, copy its source card file by file, apply only a
# WRITE-only instrument, prove the resulting Fortran syntax, and admit output
# only when every inherited native stream remains byte-identical.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH
readonly NEMO_ROOT=/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2
readonly REFERENCE_CFG=ORCA2_ICE_PISCES
readonly SOURCE_CFG=ORCA2_OMIP_L4
readonly FAILED_TARGET_CFG=ORCA2_OMIP_L4_P2VBND
readonly TARGET_CFG=ORCA2_OMIP_L4_P2VBND_R2
readonly BASELINE_RUN=/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2v_tke_a_10step_np2
readonly TARGET_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/parallel/orca2/oracle_phase2v_tke_boundary_np2
readonly RECORD=oracle_tke_boundary_kt00000002.bin
readonly EXPECTED_BASELINE_STREAMS=101
readonly EXPECTED_TARGET_STREAMS=102
readonly EXPECTED_SIZE=$((16 + 15 * 4 + 3 * 4 + 94 * 152 * 31 * 8))
readonly PHASE2V_REVISION=b7ce08cc8afa5cf377922abf198cf1794fab8a73
readonly PHASE2V_PATCH_PATH=scripts/validate/ocean_fidelity/orca2_l4/phase2v_tke_walk_writer.patch
readonly PHASE2V_PATCH_SHA256=e317142ae3e23c626dd00a7aac53ca3ff81eddee9aa3df0d8d1e3249b105db54
readonly EXPECTED_DECK_MANIFEST_SHA256=e2cb4c552360491fa9dcea0649661e5f44a9d972769d40a4ecfc2aa70a097059
readonly EXPECTED_INPUT_MANIFEST_SHA256=3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5

readonly MODE=${1:---run}
case "$MODE" in
  --run|--preflight-only) ;;
  *)
    printf 'REFUSE: usage: %s [--run|--preflight-only]\n' "$0" >&2
    exit 64
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly SOURCE_ROOT=$NEMO_ROOT/cfgs/$SOURCE_CFG
readonly TARGET_ROOT=$NEMO_ROOT/cfgs/$TARGET_CFG
readonly FAILED_TARGET_ROOT=$NEMO_ROOT/cfgs/$FAILED_TARGET_CFG
readonly CANONICAL=$NEMO_ROOT/src/OCE/ZDF/zdftke.F90
readonly WRITER=$here/l2_orca2_tke_boundary.F90
readonly BOUNDARY_PATCH=$here/zdftke_orca2_boundary.patch
readonly FC=/home/dbalwada/miniconda3/envs/nemo-build/bin/gfortran
readonly BINARY=$TARGET_ROOT/BLD/bin/nemo.exe
readonly COMPILED_ZDF=$TARGET_ROOT/BLD/ppsrc/nemo/zdftke.f90
readonly COMPILED_WRITER=$TARGET_ROOT/BLD/ppsrc/nemo/l2_orca2_tke_boundary.f90

cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE: acquisition requires a clean committed tree\n' >&2
  exit 63
}
readonly COMMIT=$(git rev-parse HEAD)

for path in "$CANONICAL" "$WRITER" "$BOUNDARY_PATCH" \
    "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$BASELINE_RUN/deck_files.sha256" \
    "$BASELINE_RUN/input_files.sha256" "$BASELINE_RUN/ocean.output" \
    "$BASELINE_RUN/run.user.time.log" "$BASELINE_RUN/time.step"; do
  [[ -f "$path" ]] || {
    printf 'REFUSE: missing required acquisition input %s\n' "$path" >&2
    exit 64
  }
done
[[ -d "$SOURCE_ROOT/EXP00" && -d "$SOURCE_ROOT/MY_SRC" ]] || {
  printf 'REFUSE: Phase-2v source configuration is incomplete: %s\n' "$SOURCE_ROOT" >&2
  exit 64
}
[[ -d "$NEMO_ROOT/cfgs/$REFERENCE_CFG" ]] || {
  printf 'REFUSE: Phase-2v reference configuration is absent: %s\n' "$REFERENCE_CFG" >&2
  exit 64
}
[[ "$EXPECTED_SIZE" -eq 3543512 ]] || {
  printf 'REFUSE: native boundary-frame byte arithmetic moved from 3543512\n' >&2
  exit 66
}

if ! printf '%s  %s\n' "$EXPECTED_DECK_MANIFEST_SHA256" \
    "$BASELINE_RUN/deck_files.sha256" | sha256sum -c - >/dev/null; then
  printf 'REFUSE: Phase-2v A deck manifest digest changed\n' >&2
  exit 65
fi
if ! printf '%s  %s\n' "$EXPECTED_INPUT_MANIFEST_SHA256" \
    "$BASELINE_RUN/input_files.sha256" | sha256sum -c - >/dev/null; then
  printf 'REFUSE: Phase-2v A input manifest digest changed\n' >&2
  exit 65
fi
if ! (cd "$BASELINE_RUN" && sha256sum -c deck_files.sha256 >/dev/null); then
  printf 'REFUSE: Phase-2v A deck files no longer verify\n' >&2
  exit 65
fi
if ! (cd "$BASELINE_RUN" && sha256sum -c input_files.sha256 >/dev/null); then
  printf 'REFUSE: Phase-2v A input files no longer verify\n' >&2
  exit 65
fi
grep -Fxq 'MPIRUN_RC=0' "$BASELINE_RUN/run.user.time.log" || {
  printf 'REFUSE: Phase-2v A lacks MPIRUN_RC=0\n' >&2
  exit 65
}
grep -Fxq 'RUN DONE' "$BASELINE_RUN/run.user.time.log" || {
  printf 'REFUSE: Phase-2v A lacks RUN DONE\n' >&2
  exit 65
}
[[ "$(tr -d '[:space:]' <"$BASELINE_RUN/time.step")" == 10 ]] || {
  printf 'REFUSE: Phase-2v A did not finish time step 10\n' >&2
  exit 65
}
mapfile -t baseline_streams < <(
  find "$BASELINE_RUN" -maxdepth 1 -type f -name 'oracle_*.bin' \
    -printf '%f\n' | sort
)
[[ "${#baseline_streams[@]}" -eq "$EXPECTED_BASELINE_STREAMS" ]] || {
  printf 'REFUSE: Phase-2v A has %s native streams, expected %s\n' \
    "${#baseline_streams[@]}" "$EXPECTED_BASELINE_STREAMS" >&2
  exit 65
}
for pattern in \
    'jpni *= *2' \
    'jpnj *= *1' \
    'number of the first time step.*nn_it000 *= *1' \
    'number of the last time step.*nn_itend *= *10' \
    'Tiling.*ln_tile *= *F' \
    'prandl number flag.*nn_pdl *= *1' \
    'mixing length type.*nn_mxl *= *3' \
    'Langmuir cells parametrization.*ln_lc *= *T' \
    'test param. to add tke induced by wind.*nn_etau *= *1' \
    'langmuir & surface wave breaking under ice.*nn_eice *= *1'; do
  grep -Eq "$pattern" "$BASELINE_RUN/ocean.output" || {
    printf 'REFUSE: Phase-2v A lacks resolved configuration row %s\n' "$pattern" >&2
    exit 65
  }
done

dry=$(mktemp -d /tmp/orca2-tke-boundary-source.XXXXXXXX)
readonly DRY_PHASE2V=$dry/zdftke.phase2v.F90
readonly DRY_BOUNDARY=$dry/zdftke.boundary.F90
readonly HISTORICAL_PATCH=$dry/phase2v_tke_walk_writer.patch
if ! git show "$PHASE2V_REVISION:$PHASE2V_PATCH_PATH" >"$HISTORICAL_PATCH"; then
  printf 'REFUSE: cannot extract the registered Phase-2v source patch\n' >&2
  exit 66
fi
[[ "$(sha256sum "$HISTORICAL_PATCH" | awk '{print $1}')" == \
    "$PHASE2V_PATCH_SHA256" ]] || {
  printf 'REFUSE: registered Phase-2v source patch digest changed\n' >&2
  exit 66
}
cp "$CANONICAL" "$DRY_PHASE2V"
if ! patch -s "$DRY_PHASE2V" <"$HISTORICAL_PATCH"; then
  printf 'REFUSE: Phase-2v patch no longer applies to canonical zdftke.F90\n' >&2
  exit 66
fi
cp "$DRY_PHASE2V" "$DRY_BOUNDARY"
if ! patch -s "$DRY_BOUNDARY" <"$BOUNDARY_PATCH"; then
  printf 'REFUSE: native boundary patch no longer applies to Phase-2v zdftke.F90\n' >&2
  exit 66
fi
removed_lines=$(diff "$DRY_PHASE2V" "$DRY_BOUNDARY" | grep '^<' || true)
if [[ -n "$removed_lines" ]]; then
  printf 'REFUSE: native boundary patch removes or replaces a Phase-2v source line\n' >&2
  exit 66
fi
if grep -Eq '^-[^-]' "$BOUNDARY_PATCH"; then
  printf 'REFUSE: native boundary patch contains a removed source line\n' >&2
  exit 66
fi
for call in o2b_begin o2b_after_boundaries_row o2b_finish; do
  [[ "$(grep -c "CALL $call" "$DRY_BOUNDARY")" -eq 1 ]] || {
    printf 'REFUSE: patched zdftke has the wrong call count for %s\n' "$call" >&2
    exit 66
  }
done
if ! python3 - "$DRY_BOUNDARY" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
capture = "CALL o2b_after_boundaries_row( jj, en )"
langmuir = "IF( ln_lc ) THEN"
if source.index(capture) > source.index(langmuir):
    raise SystemExit(1)
between = source[source.index(capture) + len(capture) : source.index(langmuir)]
if between.strip():
    raise SystemExit(1)
PY
then
  printf 'REFUSE: native frame is not immediately after boundaries and before Langmuir\n' >&2
  exit 66
fi
grep -Fq "CHARACTER(LEN=16), PARAMETER :: o2b_magic = 'NEMO_L4_TKEB_1'" \
    "$WRITER" || {
  printf 'REFUSE: boundary writer magic/layout declaration moved\n' >&2
  exit 66
}
grep -Fq 'INTENT(in) :: p_en' "$WRITER" || {
  printf 'REFUSE: boundary writer no longer treats model TKE as read-only\n' >&2
  exit 66
}

syntax=$(mktemp -d /tmp/orca2-tke-boundary-syntax.XXXXXXXX)
preprocess() {
  cpp -Dkey_nosignedzero -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 \
    -P -traditional -I "$SOURCE_ROOT/WORK" -I "$SOURCE_ROOT/BLD/inc" \
    "$1" -o "$2"
}
if ! preprocess "$WRITER" "$syntax/l2_orca2_tke_boundary.f90"; then
  printf 'REFUSE: preprocessing the native boundary writer failed\n' >&2
  exit 66
fi
if ! preprocess "$DRY_BOUNDARY" "$syntax/zdftke.f90"; then
  printf 'REFUSE: preprocessing patched zdftke failed\n' >&2
  exit 66
fi
if ! "$FC" -fsyntax-only -ffree-line-length-none \
    -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" \
    "$syntax/l2_orca2_tke_boundary.f90"; then
  printf 'REFUSE: native boundary writer failed gfortran -fsyntax-only\n' >&2
  exit 66
fi
if ! "$FC" -fsyntax-only -ffree-line-length-none -I "$syntax" \
    -I "$SOURCE_ROOT/BLD/inc" -J "$syntax" "$syntax/zdftke.f90"; then
  printf 'REFUSE: patched zdftke failed gfortran -fsyntax-only\n' >&2
  exit 66
fi

if [[ "$MODE" == --preflight-only ]]; then
  if [[ -d "$FAILED_TARGET_ROOT/EXP00" ]]; then
    # The failed first attempt copied this card after building its reference
    # clone. Verify the corrected link-aware comparison against that copy, but
    # never reuse its pre-instrumentation executable.
    verify_exp00_copy "$SOURCE_ROOT/EXP00" "$FAILED_TARGET_ROOT/EXP00"
    printf 'ORCA2_TKE_BOUNDARY_EXISTING_EXP00_COPY_READY %s\n' \
      "$FAILED_TARGET_ROOT"
  fi
  printf 'ORCA2_TKE_BOUNDARY_PREFLIGHT_READY %s\n' "$BASELINE_RUN"
  exit 0
fi

[[ ! -e "$TARGET_ROOT" && ! -e "$TARGET_RUN" ]] || {
  printf 'REFUSE: new target already exists: %s or %s\n' \
    "$TARGET_ROOT" "$TARGET_RUN" >&2
  exit 64
}
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  [[ "$free_kb" -ge 4194304 ]] || {
    printf 'REFUSE: %s has under 4 GB free\n' "$mount" >&2
    exit 67
  }
done

manifest=$(mktemp -d /tmp/orca2-tke-boundary-manifest.XXXXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$manifest"
printf '%s\n' "$COMMIT" >"$manifest/producer_commit.txt"
(
  cd "$SOURCE_ROOT"
  find EXP00 MY_SRC -maxdepth 2 \( -type f -o -type l \) -print0 \
    | sort -z | xargs -0 sha256sum
) >"$manifest/source_card.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$CANONICAL" "$HISTORICAL_PATCH" \
  "$WRITER" "$BOUNDARY_PATCH" "$BASELINE_RUN/ocean.output" \
  "$BASELINE_RUN/deck_files.sha256" "$BASELINE_RUN/input_files.sha256" \
  >"$manifest/toolchain.sha256"

cd "$NEMO_ROOT"
if ! ./makenemo -r "$REFERENCE_CFG" -n "$TARGET_CFG" -m conda-scalarmath \
    del_key 'key_xios'; then
  printf 'REFUSE: makenemo could not clone the Phase-2v reference configuration\n' >&2
  exit 68
fi
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"
done < <(find "$SOURCE_ROOT/EXP00" -maxdepth 1 \( -type f -o -type l \) \
  -print0 | sort -z)
while IFS= read -r -d '' source; do
  cp -a "$source" "$TARGET_ROOT/MY_SRC/$(basename "$source")"
done < <(find "$SOURCE_ROOT/MY_SRC" -maxdepth 1 \( -type f -o -type l \) \
  -print0 | sort -z)
cp "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"
cp "$DRY_BOUNDARY" "$TARGET_ROOT/MY_SRC/zdftke.F90"
cp "$WRITER" "$TARGET_ROOT/MY_SRC/l2_orca2_tke_boundary.F90"
verify_exp00_copy "$SOURCE_ROOT/EXP00" "$TARGET_ROOT/EXP00"
for source in "$SOURCE_ROOT"/MY_SRC/*; do
  name=$(basename "$source")
  if ! cmp -s "$source" "$TARGET_ROOT/MY_SRC/$name"; then
    printf 'REFUSE: file-by-file MY_SRC copy differs for %s\n' "$name" >&2
    exit 68
  fi
done
if ! cmp -s "$SOURCE_ROOT/cpp_$SOURCE_CFG.fcm" \
    "$TARGET_ROOT/cpp_$TARGET_CFG.fcm"; then
  printf 'REFUSE: copied Phase-2v CPP source card differs\n' >&2
  exit 68
fi
touch "$TARGET_ROOT/MY_SRC/"*.F90
if ! ./makenemo -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'; then
  printf 'REFUSE: makenemo could not build the new ORCA2 boundary target\n' >&2
  exit 68
fi
[[ -x "$BINARY" ]] || {
  printf 'REFUSE: new ORCA2 boundary target produced no executable\n' >&2
  exit 68
}
for pair in \
    "$COMPILED_ZDF:CALL o2b_begin" \
    "$COMPILED_ZDF:CALL o2b_after_boundaries_row" \
    "$COMPILED_ZDF:CALL o2b_finish" \
    "$COMPILED_WRITER:NEMO_L4_TKEB_1"; do
  file=${pair%%:*}
  needle=${pair#*:}
  grep -Fq "$needle" "$file" || {
    printf 'REFUSE: compiled source lacks registered write-only marker %s\n' "$needle" >&2
    exit 68
  }
done
if nm -D "$BINARY" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol is present in the acquisition binary\n' >&2
  exit 68
fi
sha256sum "$BINARY" >"$manifest/binary.sha256"

mkdir "$TARGET_RUN"
while read -r _ name; do
  [[ -f "$BASELINE_RUN/$name" || -L "$BASELINE_RUN/$name" ]] || {
    printf 'REFUSE: Phase-2v A deck manifest names missing file %s\n' "$name" >&2
    exit 65
  }
  cp -a "$BASELINE_RUN/$name" "$TARGET_RUN/$name"
done <"$BASELINE_RUN/deck_files.sha256"
while read -r _ name; do
  [[ -f "$BASELINE_RUN/$name" || -L "$BASELINE_RUN/$name" ]] || {
    printf 'REFUSE: Phase-2v A input manifest names missing file %s\n' "$name" >&2
    exit 65
  }
  cp -a "$BASELINE_RUN/$name" "$TARGET_RUN/$name"
done <"$BASELINE_RUN/input_files.sha256"
cp "$BASELINE_RUN/deck_files.sha256" "$BASELINE_RUN/input_files.sha256" \
  "$TARGET_RUN/"
cp "$BINARY" "$TARGET_RUN/nemo"
cp "$manifest"/* "$TARGET_RUN/"
if ! (cd "$TARGET_RUN" && sha256sum -c deck_files.sha256 >/dev/null); then
  printf 'REFUSE: staged candidate deck is not byte-identical to Phase-2v A\n' >&2
  exit 68
fi
if ! (cd "$TARGET_RUN" && sha256sum -c input_files.sha256 >/dev/null); then
  printf 'REFUSE: staged candidate inputs are not byte-identical to Phase-2v A\n' >&2
  exit 68
fi

(
  cd "$TARGET_RUN"
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  set +e
  { time mpirun -np 2 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } \
    2>>run.user.time.log
  pipe_rc=("${PIPESTATUS[@]}")
  set -e
  mpi_rc=${pipe_rc[0]}
  tee_rc=${pipe_rc[1]:-0}
  printf 'MPIRUN_RC=%d\nRUN_FINISHED_UTC=%s\n' \
    "$mpi_rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.time.log
  if [[ "$mpi_rc" -ne 0 ]]; then
    printf 'RUN FAILED\n' >>run.user.time.log
    printf 'REFUSE: np2 NEMO acquisition exited %s\n' "$mpi_rc" >&2
    exit "$mpi_rc"
  fi
  if [[ "$tee_rc" -ne 0 ]]; then
    printf 'RUN FAILED\n' >>run.user.time.log
    printf 'REFUSE: acquisition stdout capture exited %s\n' "$tee_rc" >&2
    exit "$tee_rc"
  fi
  printf 'RUN DONE\n' >>run.user.time.log
)

[[ -s "$TARGET_RUN/$RECORD" ]] || {
  printf 'REFUSE: NEMO produced no native en_after_boundaries frame\n' >&2
  exit 66
}
[[ "$(stat -c %s "$TARGET_RUN/$RECORD")" -eq "$EXPECTED_SIZE" ]] || {
  printf 'REFUSE: native boundary frame does not have %s bytes\n' "$EXPECTED_SIZE" >&2
  exit 66
}
if ! python3 - "$TARGET_RUN/$RECORD" <<'PY'
import pathlib
import struct
import sys

record = pathlib.Path(sys.argv[1])
with record.open("rb") as handle:
    magic = handle.read(16).decode("ascii").rstrip()
    header = struct.unpack("=15i", handle.read(15 * 4))
    extents = struct.unpack("=3i", handle.read(3 * 4))
expected_header = (1, 2, 3, 3, 94, 152, 31, 64, 1, 0, 94 * 152 * 31, 1, 1, 3, 1)
if magic != "NEMO_L4_TKEB_1" or header != expected_header:
    raise SystemExit(1)
if extents != (94, 152, 31) or record.stat().st_size != 3543512:
    raise SystemExit(1)
PY
then
  printf 'REFUSE: native en_after_boundaries frame failed exact schema/EOF validation\n' >&2
  exit 66
fi
grep -Fq 'ORCA2_TKE_BOUNDARY_DUMP' "$TARGET_RUN/run.user.stdout.log" || {
  printf 'REFUSE: candidate stdout lacks the native boundary write marker\n' >&2
  exit 66
}
grep -Fxq 'MPIRUN_RC=0' "$TARGET_RUN/run.user.time.log" || {
  printf 'REFUSE: candidate run lacks MPIRUN_RC=0\n' >&2
  exit 66
}
grep -Fxq 'RUN DONE' "$TARGET_RUN/run.user.time.log" || {
  printf 'REFUSE: candidate run lacks RUN DONE\n' >&2
  exit 66
}
[[ "$(tr -d '[:space:]' <"$TARGET_RUN/time.step")" == 10 ]] || {
  printf 'REFUSE: candidate run did not finish time step 10\n' >&2
  exit 66
}
mapfile -t target_streams < <(
  find "$TARGET_RUN" -maxdepth 1 -type f -name 'oracle_*.bin' \
    -printf '%f\n' | sort
)
[[ "${#target_streams[@]}" -eq "$EXPECTED_TARGET_STREAMS" ]] || {
  printf 'REFUSE: candidate has %s native streams, expected %s\n' \
    "${#target_streams[@]}" "$EXPECTED_TARGET_STREAMS" >&2
  exit 66
}
for name in "${baseline_streams[@]}"; do
  [[ -f "$TARGET_RUN/$name" ]] || {
    printf 'REFUSE: candidate omitted inherited Phase-2v stream %s\n' "$name" >&2
    exit 66
  }
  if ! cmp -s "$BASELINE_RUN/$name" "$TARGET_RUN/$name"; then
    printf 'REFUSE: write-only passivity failed; inherited stream differs: %s\n' \
      "$name" >&2
    exit 69
  fi
done
for name in "${target_streams[@]}"; do
  if [[ "$name" != "$RECORD" && ! -f "$BASELINE_RUN/$name" ]]; then
    printf 'REFUSE: candidate produced unregistered extra native stream %s\n' \
      "$name" >&2
    exit 66
  fi
done
sha256sum "$TARGET_RUN/$RECORD" >"$TARGET_RUN/$RECORD.sha256"
printf 'ORCA2_TKE_BOUNDARY_READY %s\n' "$TARGET_RUN"
