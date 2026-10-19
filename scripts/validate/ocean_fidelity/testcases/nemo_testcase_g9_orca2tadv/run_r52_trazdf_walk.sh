#!/usr/bin/env bash
# Grid ladder G9 round 7 (gridladder_r52) acquisition -- OPERATOR-EXECUTED ONLY.
# Default is read-only preflight.  --run builds one additive, WRITE-only NEMO
# observer and runs the committed G9 deck for two steps.  --admit-existing
# validates a completed operator run without building or running NEMO.
#
# The observer records the kt-1 stage-3 trazdf content recurrence, matrix
# diagonals, forward recurrence, and final solution.  No captured value is read
# back into NEMO.  Admission requires all 42 round-46 records to remain
# byte-identical and every frame of the new record to be finite and complete.
#
# The agent that wrote this runs neither makenemo nor NEMO.
set -Eeuo pipefail

fail() { printf 'REFUSE: G9 r52 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly BASE_NAME=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly NAME=ORCA2G9_OMIP_L4_R52TRAZDF
readonly BASE=$NEMO_ROOT/cfgs/$BASE_NAME TARGET=$NEMO_ROOT/cfgs/$NAME
readonly BASE_BIN_SHA=41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275
readonly WANT_KEYS='key_RK3 key_qco key_si3 key_vco_1d3d'
readonly ROUNDS=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds
readonly INPUT_ROOT=$ROUNDS/round21/oracle_g4_r21/p6_a
readonly R46=$ROUNDS/round46/oracle_g9_r46/twin_a
readonly EVIDENCE=${EVIDENCE:-$ROUNDS/round52/oracle_g9_r52}
readonly RUN=$EVIDENCE/twin_walk
readonly RECORD=oracle_g9_trazdf_zrhs_kt00000001.bin
readonly TRA_SRC=$NEMO_ROOT/src/OCE/TRA/trazdf.F90
readonly INPUT_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc
  ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc q_10.15JUNE2009_fill.nc
  ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc
  sss_data.nc runoff_core_monthly.nc namelist_ref)

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in worktree'
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] \
  || fail 'commit the exact acquisition tool before running it'
readonly COMMIT=$(git -C "$repo" rev-parse HEAD)
readonly DECK=$here/namelist_cfg ICE_DECK=$here/namelist_ice_cfg
readonly ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref

declare -a scratch_dirs=()
cleanup() {
  local path
  for path in "${scratch_dirs[@]}"; do
    [[ ! -e "$path" ]] || rm -r -- "$path"
  done
}
trap cleanup EXIT

new_scratch() {
  local path
  path=$(mktemp -d "$repo/.g9-r52-preflight.XXXXXX")
  scratch_dirs+=("$path")
  printf '%s\n' "$path"
}

sha() { sha256sum "$1" | awk '{print $1}'; }
sorted_keys() { sed -E 's/.*fppkeys *//' "$1" | xargs -n1 | LC_ALL=C sort | xargs; }

# Add declarations, captures, and one terminal stream write.  Existing NEMO
# statements are not removed or replaced; check_additive enforces that.
patch_trazdf() { # target trazdf.F90
  python3 -I - "$1" <<'PY'
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

def add_after(old: str, addition: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(
            f"{p}: non-unique after anchor ({count}): {old[:80]!r}")
    s = s.replace(old, old + addition)

add_after(
    "   PUBLIC   tra_zdf_imp   ! called by trczdf.F90\n",
    "\n"
    "   INTEGER, SAVE :: n_r52_kt = -1   ! G9 r52 WRITE-only observer clock\n")
add_after(
    "      !\n      !                                      !* compute lateral mixing trend and add it to the general trend\n",
    "      n_r52_kt = kt\n")
add_after(
    "      REAL(wp), DIMENSION(T1Di(0),jpk) ::  zwi, zwt, zwd, zws\n",
    "      LOGICAL :: l_r52_capture\n"
    "      INTEGER :: i_r52_ios, n_r52_unit\n"
    "      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE :: zr52_content, zr52_matrix\n"
    "      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE :: zr52_forward, zr52_solution\n")
add_after(
    "      !!---------------------------------------------------------------------\n      !\n",
    "      l_r52_capture = lwp .AND. .NOT.l_istiled .AND. cdtype == 'TRA' .AND. &\n"
    "         & n_r52_kt == nit000 .AND. Kbb == 1 .AND. Kmm == 2 .AND. &\n"
    "         & Krhs == 3 .AND. Kaa == 3 .AND. kjpt >= 2\n"
    "      IF( l_r52_capture ) THEN\n"
    "         ALLOCATE( zr52_content(jpi,jpj,jpk,2), zr52_matrix(jpi,jpj,jpk,4), &\n"
    "            & zr52_forward(jpi,jpj,jpk,2), zr52_solution(jpi,jpj,jpk,2) )\n"
    "         zr52_content = 0._wp ; zr52_matrix = 0._wp\n"
    "         zr52_forward = 0._wp ; zr52_solution = 0._wp\n"
    "      ENDIF\n")
add_after(
    "               DO_2Dik( 0, 0,   2, jpkm1, 1 )\n"
    "                  zwt(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwt(ji,jk-1)\n"
    "               END_2D\n",
    "               IF( l_r52_capture .AND. jn == jp_tem ) THEN\n"
    "                  DO_2Dik( 0, 0,   1, jpkm1, 1 )\n"
    "                     zr52_matrix(ji,jj,jk,1) = zwi(ji,jk)\n"
    "                     zr52_matrix(ji,jj,jk,2) = zwd(ji,jk)\n"
    "                     zr52_matrix(ji,jj,jk,3) = zws(ji,jk)\n"
    "                     zr52_matrix(ji,jj,jk,4) = zwt(ji,jk)\n"
    "                  END_2D\n"
    "               ENDIF\n")
add_after(
    "               pt(ji,jj,1,jn,Kaa) =       e3t(ji,jj,1,Kbb) * pt(ji,jj,1,jn,Kbb )    &\n"
    "                  &               + p2dt * e3t(ji,jj,1,Kmm) * pt(ji,jj,1,jn,Krhs)\n",
    "               IF( l_r52_capture .AND. jn <= 2 ) THEN\n"
    "                  zr52_content(ji,jj,1,jn) = pt(ji,jj,1,jn,Kaa)\n"
    "                  zr52_forward(ji,jj,1,jn) = pt(ji,jj,1,jn,Kaa)\n"
    "               ENDIF\n")
add_after(
    "               zrhs =       e3t(ji,jj,jk,Kbb) * pt(ji,jj,jk,jn,Kbb )   &\n"
    "                  & + p2dt * e3t(ji,jj,jk,Kmm) * pt(ji,jj,jk,jn,Krhs)   ! zrhs=right hand side\n",
    "               IF( l_r52_capture .AND. jn <= 2 ) zr52_content(ji,jj,jk,jn) = zrhs\n")
add_after(
    "               pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk) / zwt(ji,jk-1) * pt(ji,jj,jk-1,jn,Kaa)\n",
    "               IF( l_r52_capture .AND. jn <= 2 ) zr52_forward(ji,jj,jk,jn) = pt(ji,jj,jk,jn,Kaa)\n")
add_after(
    "            DO_2Dik( 0, 0,   jpk-2, 1, -1 )\n"
    "               pt(ji,jj,jk,jn,Kaa) = ( pt(ji,jj,jk,jn,Kaa) - zws(ji,jk) * pt(ji,jj,jk+1,jn,Kaa) )   &\n"
    "                  &             / zwt(ji,jk) * tmask(ji,jj,jk)\n"
    "            END_2D\n",
    "            IF( l_r52_capture .AND. jn <= 2 ) THEN\n"
    "               DO_2Dik( 0, 0,   1, jpkm1, 1 )\n"
    "                  zr52_solution(ji,jj,jk,jn) = pt(ji,jj,jk,jn,Kaa)\n"
    "               END_2D\n"
    "            ENDIF\n")
add_after(
    "      END_1D                                          !  i-k slices loop  !\n"
    "      !                                               ! ================= !\n",
    "      IF( l_r52_capture ) THEN\n"
    "         OPEN( NEWUNIT=n_r52_unit, FILE='oracle_g9_trazdf_zrhs_kt00000001.bin', &\n"
    "            & ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE', &\n"
    "            & ACTION='WRITE', IOSTAT=i_r52_ios )\n"
    "         IF( i_r52_ios /= 0 ) CALL ctl_stop( 'cannot open G9 r52 trazdf walk' )\n"
    "         WRITE(n_r52_unit) 'NEMO_G9_ZRHS_1  ', 1, n_r52_kt, Kbb, Kmm, Krhs, Kaa, &\n"
    "            & jpi, jpj, jpk, nn_hls, STORAGE_SIZE(1._wp), 4, 10\n"
    "         WRITE(n_r52_unit) 'zrhs_content    ', zr52_content\n"
    "         WRITE(n_r52_unit) 'matrix_diags    ', zr52_matrix\n"
    "         WRITE(n_r52_unit) 'forward_state   ', zr52_forward\n"
    "         WRITE(n_r52_unit) 'solution        ', zr52_solution\n"
    "         CLOSE(n_r52_unit)\n"
    "         DEALLOCATE( zr52_content, zr52_matrix, zr52_forward, zr52_solution )\n"
    "      ENDIF\n")
p.write_text(s)
PY
}

check_additive() { # base source, patched source
  python3 -I - "$1" "$2" <<'PY'
import difflib
import pathlib
import sys

before = pathlib.Path(sys.argv[1]).read_text().splitlines()
after = pathlib.Path(sys.argv[2]).read_text().splitlines()
removed = [line for line in difflib.ndiff(before, after) if line.startswith('- ')]
if removed:
    raise SystemExit(f"observer is not additive: {removed[:3]}")
print('TRAZDF_OBSERVER_ADDITIVE_OK')
PY
}

check_record() { # record, synthetic|production
  python3 -I - "$1" "$2" <<'PY'
import pathlib
import struct
import sys

import numpy as np

path = pathlib.Path(sys.argv[1])
profile = sys.argv[2]
frames = (
    ('zrhs_content', 2),
    ('matrix_diags', 4),
    ('forward_state', 2),
    ('solution', 2),
)
with path.open('rb') as stream:
    magic = stream.read(16).decode('ascii')
    raw_header = stream.read(52)
    if len(raw_header) != 52:
        raise SystemExit(f'truncated header: {path}')
    header = struct.unpack('=13i', raw_header)
    version, kt, kbb, kmm, krhs, kaa, nx, ny, nz, halo, bits, nframes, narrays = header
    expected_shape = (2, 3, 2, 1) if profile == 'synthetic' else (184, 152, 31, 2)
    if profile not in {'synthetic', 'production'}:
        raise SystemExit(f'unknown reader profile: {profile}')
    if magic != 'NEMO_G9_ZRHS_1  ':
        raise SystemExit(f'bad magic/width: {magic!r}, len={len(magic)}')
    if (version, kt, kbb, kmm, krhs, kaa, bits, nframes, narrays) != (
            1, 1, 1, 2, 3, 3, 64, 4, 10):
        raise SystemExit(f'bad header values: {header}')
    if (nx, ny, nz, halo) != expected_shape:
        raise SystemExit(f'bad dimensions: {(nx, ny, nz, halo)} != {expected_shape}')
    n3 = nx * ny * nz
    finite_arrays = 0
    for expected_label, count in frames:
        raw_label = stream.read(16)
        if len(raw_label) != 16:
            raise SystemExit(f'missing frame {expected_label}: got {len(raw_label)} tag bytes')
        label = raw_label.decode('ascii').rstrip()
        raw_values = stream.read(count * n3 * 8)
        if len(raw_values) != count * n3 * 8:
            raise SystemExit(
                f'truncated frame {expected_label}: {len(raw_values)} != {count*n3*8}')
        values = np.frombuffer(raw_values, dtype=np.float64)
        if label != expected_label:
            raise SystemExit(f'bad frame label: {label!r} != {expected_label!r}')
        if values.size != count * n3 or not np.isfinite(values).all():
            raise SystemExit(f'nonfinite/incomplete frame: {expected_label}')
        finite_arrays += count
    trailing = stream.read(1)
    if trailing:
        raise SystemExit(f'unread trailing payload in {path.name}')
print(
    f'G9R52_TRAZDF_RECORD_OK profile={profile} frames={len(frames)} '
    f'finite_arrays={finite_arrays} bytes={path.stat().st_size}')
PY
}

layout_selftest() {
  local root writer record missing bytes
  root=$(new_scratch)
  writer=$root/write_layout.f90
  record=$root/$RECORD
  cat >"$writer" <<'F90'
PROGRAM write_layout
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp=8, nx=2, ny=3, nz=2
   INTEGER :: unit
   REAL(wp) :: content(nx,ny,nz,2), matrix(nx,ny,nz,4)
   REAL(wp) :: forward(nx,ny,nz,2), solution(nx,ny,nz,2)
   content = 1._wp ; matrix = 2._wp ; forward = 3._wp ; solution = 4._wp
   OPEN( NEWUNIT=unit, FILE='oracle_g9_trazdf_zrhs_kt00000001.bin', &
      & ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE' )
   WRITE(unit) 'NEMO_G9_ZRHS_1  ', 1, 1, 1, 2, 3, 3, nx, ny, nz, 1, STORAGE_SIZE(1._wp), 4, 10
   WRITE(unit) 'zrhs_content    ', content
   WRITE(unit) 'matrix_diags    ', matrix
   WRITE(unit) 'forward_state   ', forward
   WRITE(unit) 'solution        ', solution
   CLOSE(unit)
END PROGRAM write_layout
F90
  (cd "$root" && gfortran -o write_layout write_layout.f90 && ./write_layout)
  check_record "$record" synthetic
  missing=$root/missing_frame.bin
  cp "$record" "$missing"
  bytes=$(stat -c %s "$missing")
  truncate -s $((bytes - 16 - 2 * 2 * 3 * 2 * 8)) "$missing"
  if check_record "$missing" synthetic >"$root/missing.log" 2>&1; then
    fail 'missing-frame plant stayed green'
  fi
  grep -Fq 'missing frame solution' "$root/missing.log" \
    || fail 'missing-frame plant failed without the expected refusal'
  printf 'G9R52_MISSING_FRAME_PLANT_REFUSED\n'
  printf 'G9R52_LAYOUT_SELFTEST_OK magic_bytes=16 frame_tag_bytes=16 arrays=10\n'
}

preflight() {
  local f dry planted
  [[ -x "$BASE/BLD/bin/nemo.exe" && "$(sha "$BASE/BLD/bin/nemo.exe")" == "$BASE_BIN_SHA" ]] \
    || fail 'round-46 record binary base is missing or changed'
  [[ "$(sorted_keys "$BASE/cpp_$BASE_NAME.fcm")" == "$WANT_KEYS" ]] \
    || fail 'base key set changed'
  [[ "$(sha "$DECK")" == 570db996c34e98ccdabf3c367455c1ee6d0b4f18df9f3bdd7264cf053f1f0720 ]] \
    || fail 'committed G9 deck changed'
  [[ "$(sha "$ICE_DECK")" == 9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8 ]] \
    || fail 'committed G9 ice deck changed'
  for f in "${INPUT_FILES[@]}"; do [[ -f "$INPUT_ROOT/$f" ]] || fail "missing input $f"; done
  [[ "$(find "$R46" -maxdepth 1 -type f -name 'oracle_*kt0000000[12]*.bin' | wc -l)" -eq 42 ]] \
    || fail 'round-46 passivity set is not exactly 42 records'
  grep -Fq 'SUBROUTINE tra_zdf_imp' "$TRA_SRC" || fail 'NEMO trazdf source moved'
  dry=$(new_scratch)
  cp "$TRA_SRC" "$dry/trazdf.F90"
  patch_trazdf "$dry/trazdf.F90"
  check_additive "$TRA_SRC" "$dry/trazdf.F90"
  grep -Fq "WRITE(n_r52_unit) 'NEMO_G9_ZRHS_1  '" "$dry/trazdf.F90" \
    || fail 'record header writer absent'
  grep -Fq "WRITE(n_r52_unit) 'solution        ', zr52_solution" "$dry/trazdf.F90" \
    || fail 'solution frame writer absent'
  planted=$dry/trazdf_nonadditive.F90
  cp "$dry/trazdf.F90" "$planted"
  sed -i '0,/CALL tra_zdf_imp/s//CALL tra_zdf_imp_REMOVED/' "$planted"
  if check_additive "$TRA_SRC" "$planted" >"$dry/nonadditive.log" 2>&1; then
    fail 'non-additive patch plant stayed green'
  fi
  grep -Fq 'observer is not additive' "$dry/nonadditive.log" \
    || fail 'non-additive plant failed without the expected refusal'
  printf 'G9R52_NONADDITIVE_PATCH_PLANT_REFUSED\n'
  layout_selftest
  printf 'G9R52_PREFLIGHT_OK base=%s target=%s commit=%s\n' "$BASE_NAME" "$NAME" "$COMMIT"
}

build() {
  local f
  [[ ! -e "$TARGET" ]] || fail "target config exists: $TARGET"
  (cd "$NEMO_ROOT" && ./makenemo -r ORCA2_ICE_PISCES -n "$NAME" -m conda-scalarmath del_key key_xios)
  rm -f "$TARGET/MY_SRC/"*
  cp -a "$BASE/MY_SRC/." "$TARGET/MY_SRC/"
  cp "$BASE/cpp_$BASE_NAME.fcm" "$TARGET/cpp_$NAME.fcm"
  cp "$TRA_SRC" "$TARGET/MY_SRC/trazdf.F90"
  patch_trazdf "$TARGET/MY_SRC/trazdf.F90"
  touch "$TARGET/MY_SRC/"*.F90
  (cd "$NEMO_ROOT" && ./makenemo -n "$NAME" -m conda-scalarmath -y --clean)
  (cd "$NEMO_ROOT" && ./makenemo -n "$NAME" -m conda-scalarmath del_key key_xios)
  [[ -x "$TARGET/BLD/bin/nemo.exe" ]] || fail 'build produced no executable'
  [[ "$(sorted_keys "$TARGET/cpp_$NAME.fcm")" == "$WANT_KEYS" ]] \
    || fail 'target key set changed'
  check_additive "$BASE/BLD/ppsrc/nemo/trazdf.f90" "$TARGET/BLD/ppsrc/nemo/trazdf.f90"
  for f in $(cd "$BASE/BLD/ppsrc/nemo" && ls *.f90); do
    [[ "$f" == trazdf.f90 ]] && continue
    cmp -s "$BASE/BLD/ppsrc/nemo/$f" "$TARGET/BLD/ppsrc/nemo/$f" \
      || fail "ppsrc $f differs"
  done
}

run_walk() {
  local f rc=0
  [[ ! -e "$RUN" ]] || fail "run directory exists: $RUN"
  mkdir -p "$RUN"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$RUN/$f"; done
  cp "$ICE_REF" "$RUN/namelist_ice_ref"
  cp "$ICE_DECK" "$RUN/namelist_ice_cfg"
  sed -E -e 's/^(   nn_itend    = ).*/\1    2/' -e 's/^(   nn_stock    = ).*/\1    2/' \
    -e 's/^(   nn_stocklist = ).*/\12,0,0,0,0,0,0,0,0,0/' "$DECK" >"$RUN/namelist_cfg"
  cp "$TARGET/BLD/bin/nemo.exe" "$RUN/nemo"
  (cd "$RUN"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$RUN/run.exit"
  [[ "$rc" -eq 0 && "$(tr -d '[:space:]' <"$RUN/time.step")" == 2 ]] \
    || fail 'observer run did not complete two steps'
}

admit() {
  local f n=0
  [[ -f "$RUN/run.exit" && "$(tr -d '[:space:]' <"$RUN/run.exit")" == 0 ]] \
    || fail 'candidate run has no successful exit stamp'
  [[ -f "$RUN/time.step" && "$(tr -d '[:space:]' <"$RUN/time.step")" == 2 ]] \
    || fail 'candidate run did not complete two steps'
  while IFS= read -r f; do
    cmp -s "$R46/$f" "$RUN/$f" || fail "pre-existing record changed: $f"
    n=$((n + 1))
  done < <(find "$R46" -maxdepth 1 -type f -name 'oracle_*kt0000000[12]*.bin' \
    -printf '%f\n' | LC_ALL=C sort)
  [[ "$n" -eq 42 ]] || fail "passivity compared $n records, expected 42"
  [[ "$(find "$RUN" -maxdepth 1 -type f -name "$RECORD" | wc -l)" -eq 1 ]] \
    || fail "expected exactly one $RECORD"
  check_record "$RUN/$RECORD" production
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$TARGET/BLD/bin/nemo.exe" "$TARGET/MY_SRC/trazdf.F90" \
    "$TARGET/BLD/ppsrc/nemo/trazdf.f90" "$DECK" "$ICE_DECK" "$RUN/$RECORD" \
    >"$EVIDENCE/manifest/producers.sha256"
  printf 'G9R52_ACQUISITION_OK passivity_records=%s record=%s evidence=%s\n' \
    "$n" "$RECORD" "$EVIDENCE"
}

preflight
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD_BUILD %s from %s; run two G9 steps; write %s in %s\n' \
    "$NAME" "$BASE_NAME" "$RECORD" "$RUN"
  exit 0
fi
if [[ "$mode" == --admit-existing ]]; then admit; exit 0; fi
mkdir -p "$EVIDENCE"
build
run_walk
admit
