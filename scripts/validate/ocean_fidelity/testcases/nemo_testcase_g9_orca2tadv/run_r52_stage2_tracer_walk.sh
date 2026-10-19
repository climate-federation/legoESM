#!/usr/bin/env bash
# Grid ladder G9 round 7 follow-up acquisition -- OPERATOR-EXECUTED ONLY.
# Default is read-only preflight.  --run builds an additive, WRITE-only NEMO
# observer and runs the committed G9 deck for two steps.  --admit-existing
# validates a completed operator run without building or running NEMO.
#
# The observer records kt-2 stage-2 tracer transport, stored T ratios,
# after-advection RHS, after-SBC RHS, and the stage update.  Captured values
# are never read back into NEMO.  Admission also requires all 42 round-46
# kt-1/kt-2 records to remain byte-identical.
set -Eeuo pipefail

fail() { printf 'REFUSE: G9 r52 stage-2 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly BASE_NAME=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly NAME=ORCA2G9_OMIP_L4_R52STG2
readonly BASE=$NEMO_ROOT/cfgs/$BASE_NAME TARGET=$NEMO_ROOT/cfgs/$NAME
readonly BASE_BIN_SHA=41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275
readonly WANT_KEYS='key_RK3 key_qco key_si3 key_vco_1d3d'
readonly ROUNDS=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds
readonly INPUT_ROOT=$ROUNDS/round21/oracle_g4_r21/p6_a
readonly R46=$ROUNDS/round46/oracle_g9_r46/twin_a
readonly EVIDENCE=${EVIDENCE:-$ROUNDS/round52/oracle_g9_r52_stage2}
readonly RUN=$EVIDENCE/twin_walk
readonly RECORD=oracle_g9_stage2_tracer_kt00000002.bin
readonly STG_SRC=$BASE/MY_SRC/stprk3_stg.F90
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
scratch_path=
cleanup() {
  local path
  for path in "${scratch_dirs[@]}"; do [[ ! -e "$path" ]] || rm -r -- "$path"; done
}
trap cleanup EXIT
new_scratch() {
  scratch_path=$(mktemp -d "$repo/.g9-r52-stage2-preflight.XXXXXX")
  scratch_dirs+=("$scratch_path")
}

sha() { sha256sum "$1" | awk '{print $1}'; }
sorted_keys() { sed -E 's/.*fppkeys *//' "$1" | xargs -n1 | LC_ALL=C sort | xargs; }

patch_stage2() { # target stprk3_stg.F90
  python3 -I - "$1" <<'PY'
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

def add_after(old: str, addition: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"{p}: non-unique after anchor ({count}): {old[:80]!r}")
    s = s.replace(old, old + addition)

add_after(
    "      INTEGER  ::   l1_unit, l1_ios, l2_tracer_unit, l2_tracer_ios\n",
    "      INTEGER  ::   n_r52s2_unit, n_r52s2_ios\n")
add_after(
    "         ENDIF\n         ! Phase-3 read-only transport instrument.  This is exactly the zF\n",
    "         IF( lwp .AND. .NOT.ln_tile .AND. kstp == nit000+1 .AND. kstg == 2 ) THEN\n"
    "            OPEN( NEWUNIT=n_r52s2_unit, FILE='oracle_g9_stage2_tracer_kt00000002.bin', &\n"
    "               & ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE', &\n"
    "               & ACTION='WRITE', IOSTAT=n_r52s2_ios )\n"
    "            IF( n_r52s2_ios /= 0 ) CALL ctl_stop( 'cannot open G9 r52 stage-2 walk' )\n"
    "            WRITE(n_r52s2_unit) 'NEMO_G9_STG2_1  ', 1, kstp, kstg, Kbb, Kmm, Krhs, Kaa, &\n"
    "               & jpi, jpj, jpk, nn_hls, STORAGE_SIZE(1._wp), 12\n"
    "            WRITE(n_r52s2_unit) 'transport       ', l4_canon_3d(zFu,'U'), &\n"
    "               & l4_canon_3d(zFv,'V'), l4_canon_3d(zFw,'T')\n"
    "            WRITE(n_r52s2_unit) 'ratios          ', l4_canon_2d(r3t(:,:,Kbb),'T'), &\n"
    "               & l4_canon_2d(r3t(:,:,Kmm),'T'), l4_canon_2d(r3t(:,:,Kaa),'T')\n"
    "         ENDIF\n")
add_after(
    "         CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )   ! horizontal & vertical advection\n",
    "         IF( lwp .AND. .NOT.ln_tile .AND. kstp == nit000+1 .AND. kstg == 2 ) &\n"
    "            & WRITE(n_r52s2_unit) 'after_adv       ', &\n"
    "            & l4_canon_3d(ts(:,:,:,jp_tem,Krhs),'T'), &\n"
    "            & l4_canon_3d(ts(:,:,:,jp_sal,Krhs),'T')\n")
add_after(
    "         CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )   ! surface boundary condition\n",
    "         IF( lwp .AND. .NOT.ln_tile .AND. kstp == nit000+1 .AND. kstg == 2 ) &\n"
    "            & WRITE(n_r52s2_unit) 'after_sbc       ', &\n"
    "            & l4_canon_3d(ts(:,:,:,jp_tem,Krhs),'T'), &\n"
    "            & l4_canon_3d(ts(:,:,:,jp_sal,Krhs),'T')\n")
add_after(
    "         END DO\n         !\n         IF(sn_cfctl%l_prtctl)   CALL prt_ctl( tab3d_1=ts(:,:,:,jp_tem,Kaa), clinfo1='stp stg   - Ta: ', mask1=tmask,   &\n",
    "         IF( lwp .AND. .NOT.ln_tile .AND. kstp == nit000+1 .AND. kstg == 2 ) THEN\n"
    "            WRITE(n_r52s2_unit) 'stage_update    ', &\n"
    "               & l4_canon_3d(ts(:,:,:,jp_tem,Kaa),'T'), &\n"
    "               & l4_canon_3d(ts(:,:,:,jp_sal,Kaa),'T')\n"
    "            CLOSE(n_r52s2_unit)\n"
    "         ENDIF\n")
p.write_text(s)
PY
}

check_additive() { # base source, patched source
  python3 -I - "$1" "$2" <<'PY'
import difflib, pathlib, sys
before = pathlib.Path(sys.argv[1]).read_text().splitlines()
after = pathlib.Path(sys.argv[2]).read_text().splitlines()
removed = [line for line in difflib.ndiff(before, after) if line.startswith('- ')]
if removed:
    raise SystemExit(f"observer is not additive: {removed[:3]}")
print('STAGE2_OBSERVER_ADDITIVE_OK')
PY
}

check_record() { # record, synthetic|production
  python3 -I - "$1" "$2" <<'PY'
import pathlib, struct, sys
import numpy as np
path, profile = pathlib.Path(sys.argv[1]), sys.argv[2]
frames = (('transport', 3, '3d'), ('ratios', 3, '2d'),
          ('after_adv', 2, '3d'), ('after_sbc', 2, '3d'),
          ('stage_update', 2, '3d'))
with path.open('rb') as stream:
    magic = stream.read(16).decode('ascii')
    raw = stream.read(52)
    if len(raw) != 52: raise SystemExit(f'truncated header: {path}')
    header = struct.unpack('=13i', raw)
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, halo, bits, narrays = header
    shape = (2, 3, 2, 1) if profile == 'synthetic' else (184, 152, 31, 2)
    if profile not in {'synthetic', 'production'}: raise SystemExit(f'bad profile {profile}')
    if magic != 'NEMO_G9_STG2_1  ': raise SystemExit(f'bad magic {magic!r}')
    if (version, kt, stage, kbb, kmm, krhs, kaa, bits, narrays) != (
            1, 2, 2, 3, 1, 2, 2, 64, 12):
        raise SystemExit(f'bad header values: {header}')
    if (nx, ny, nz, halo) != shape: raise SystemExit(f'bad dimensions: {header}')
    n3, n2, finite = nx * ny * nz, nx * ny, 0
    for label, count, rank in frames:
        tag = stream.read(16)
        if len(tag) != 16: raise SystemExit(f'missing frame {label}')
        got = tag.decode('ascii').rstrip()
        n = count * (n3 if rank == '3d' else n2)
        payload = stream.read(n * 8)
        if len(payload) != n * 8: raise SystemExit(f'truncated frame {label}')
        values = np.frombuffer(payload, dtype=np.float64)
        if got != label or values.size != n or not np.isfinite(values).all():
            raise SystemExit(f'bad frame {label}: got {got!r}')
        finite += count
    if stream.read(1): raise SystemExit(f'unread trailing payload: {path}')
print(f'G9R52_STAGE2_RECORD_OK profile={profile} frames=5 finite_arrays={finite} bytes={path.stat().st_size}')
PY
}

layout_selftest() {
  local root writer record missing bytes
  new_scratch; root=$scratch_path; writer=$root/write_layout.f90; record=$root/$RECORD
  cat >"$writer" <<'F90'
PROGRAM write_layout
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp=8, nx=2, ny=3, nz=2
   INTEGER :: unit
   REAL(wp) :: a3(nx,ny,nz,3), a2(nx,ny,3), b(nx,ny,nz,2)
   a3=1._wp ; a2=2._wp ; b=3._wp
   OPEN( NEWUNIT=unit, FILE='oracle_g9_stage2_tracer_kt00000002.bin', ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE' )
   WRITE(unit) 'NEMO_G9_STG2_1  ', 1, 2, 2, 3, 1, 2, 2, nx, ny, nz, 1, STORAGE_SIZE(1._wp), 12
   WRITE(unit) 'transport       ', a3
   WRITE(unit) 'ratios          ', a2
   WRITE(unit) 'after_adv       ', b
   WRITE(unit) 'after_sbc       ', b
   WRITE(unit) 'stage_update    ', b
   CLOSE(unit)
END PROGRAM write_layout
F90
  (cd "$root" && gfortran -ffree-line-length-none -o write_layout write_layout.f90 && ./write_layout)
  check_record "$record" synthetic
  missing=$root/missing_frame.bin; cp "$record" "$missing"; bytes=$(stat -c %s "$missing")
  truncate -s $((bytes - 16 - 2 * 2 * 3 * 2 * 8)) "$missing"
  if check_record "$missing" synthetic >"$root/missing.log" 2>&1; then fail 'missing-frame plant stayed green'; fi
  grep -Fq 'missing frame stage_update' "$root/missing.log" \
    || fail 'missing-frame plant failed without expected refusal'
  printf 'G9R52_STAGE2_MISSING_FRAME_PLANT_REFUSED\n'
  printf 'G9R52_STAGE2_LAYOUT_SELFTEST_OK magic_bytes=16 frame_tag_bytes=16 arrays=12\n'
}

preflight() {
  local f dry planted
  [[ -x "$BASE/BLD/bin/nemo.exe" && "$(sha "$BASE/BLD/bin/nemo.exe")" == "$BASE_BIN_SHA" ]] \
    || fail 'round-46 record binary base is missing or changed'
  [[ "$(sorted_keys "$BASE/cpp_$BASE_NAME.fcm")" == "$WANT_KEYS" ]] || fail 'base key set changed'
  [[ "$(sha "$DECK")" == 570db996c34e98ccdabf3c367455c1ee6d0b4f18df9f3bdd7264cf053f1f0720 ]] || fail 'G9 deck changed'
  [[ "$(sha "$ICE_DECK")" == 9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8 ]] || fail 'G9 ice deck changed'
  for f in "${INPUT_FILES[@]}"; do [[ -f "$INPUT_ROOT/$f" ]] || fail "missing input $f"; done
  [[ "$(find "$R46" -maxdepth 1 -type f -name 'oracle_*kt0000000[12]*.bin' | wc -l)" -eq 42 ]] || fail 'round-46 passivity set is not 42 records'
  [[ -f "$STG_SRC" ]] || fail 'instrumented base stprk3 source missing'
  new_scratch; dry=$scratch_path; cp "$STG_SRC" "$dry/stprk3_stg.F90"
  patch_stage2 "$dry/stprk3_stg.F90"; check_additive "$STG_SRC" "$dry/stprk3_stg.F90"
  grep -Fq "WRITE(n_r52s2_unit) 'stage_update    '" "$dry/stprk3_stg.F90" || fail 'terminal frame absent'
  planted=$dry/nonadditive.F90; cp "$dry/stprk3_stg.F90" "$planted"
  sed -i '0,/CALL tra_adv    (/s//CALL tra_adv_REMOVED(/' "$planted"
  if check_additive "$STG_SRC" "$planted" >"$dry/nonadditive.log" 2>&1; then fail 'non-additive plant stayed green'; fi
  grep -Fq 'observer is not additive' "$dry/nonadditive.log" || fail 'non-additive plant lacked refusal'
  printf 'G9R52_STAGE2_NONADDITIVE_PATCH_PLANT_REFUSED\n'
  layout_selftest
  printf 'G9R52_STAGE2_PREFLIGHT_OK base=%s target=%s commit=%s\n' "$BASE_NAME" "$NAME" "$COMMIT"
}

build() {
  local f
  [[ ! -e "$TARGET" ]] || fail "target config exists: $TARGET"
  (cd "$NEMO_ROOT" && ./makenemo -r ORCA2_ICE_PISCES -n "$NAME" -m conda-scalarmath del_key key_xios)
  rm -f "$TARGET/MY_SRC/"*; cp -a "$BASE/MY_SRC/." "$TARGET/MY_SRC/"
  cp "$BASE/cpp_$BASE_NAME.fcm" "$TARGET/cpp_$NAME.fcm"
  patch_stage2 "$TARGET/MY_SRC/stprk3_stg.F90"; touch "$TARGET/MY_SRC/"*.F90
  (cd "$NEMO_ROOT" && ./makenemo -n "$NAME" -m conda-scalarmath -y --clean)
  (cd "$NEMO_ROOT" && ./makenemo -n "$NAME" -m conda-scalarmath del_key key_xios)
  [[ -x "$TARGET/BLD/bin/nemo.exe" ]] || fail 'build produced no executable'
  [[ "$(sorted_keys "$TARGET/cpp_$NAME.fcm")" == "$WANT_KEYS" ]] || fail 'target key set changed'
  check_additive "$BASE/BLD/ppsrc/nemo/stprk3_stg.f90" "$TARGET/BLD/ppsrc/nemo/stprk3_stg.f90"
  for f in $(cd "$BASE/BLD/ppsrc/nemo" && ls *.f90); do
    [[ "$f" == stprk3_stg.f90 ]] && continue
    cmp -s "$BASE/BLD/ppsrc/nemo/$f" "$TARGET/BLD/ppsrc/nemo/$f" || fail "ppsrc $f differs"
  done
}

run_walk() {
  local f rc=0
  [[ ! -e "$RUN" ]] || fail "run directory exists: $RUN"; mkdir -p "$RUN"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$RUN/$f"; done
  cp "$ICE_REF" "$RUN/namelist_ice_ref"; cp "$ICE_DECK" "$RUN/namelist_ice_cfg"
  sed -E -e 's/^(   nn_itend    = ).*/\1    2/' -e 's/^(   nn_stock    = ).*/\1    2/' \
    -e 's/^(   nn_stocklist = ).*/\12,0,0,0,0,0,0,0,0,0/' "$DECK" >"$RUN/namelist_cfg"
  cp "$TARGET/BLD/bin/nemo.exe" "$RUN/nemo"
  (cd "$RUN"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$RUN/run.exit"
  [[ "$rc" -eq 0 && "$(tr -d '[:space:]' <"$RUN/time.step")" == 2 ]] || fail 'run did not complete two steps'
}

admit() {
  local f n=0
  [[ -f "$RUN/run.exit" && "$(tr -d '[:space:]' <"$RUN/run.exit")" == 0 ]] || fail 'no successful exit stamp'
  [[ -f "$RUN/time.step" && "$(tr -d '[:space:]' <"$RUN/time.step")" == 2 ]] || fail 'run did not complete two steps'
  while IFS= read -r f; do cmp -s "$R46/$f" "$RUN/$f" || fail "pre-existing record changed: $f"; n=$((n+1)); done \
    < <(find "$R46" -maxdepth 1 -type f -name 'oracle_*kt0000000[12]*.bin' -printf '%f\n' | LC_ALL=C sort)
  [[ "$n" -eq 42 ]] || fail "passivity compared $n records, expected 42"
  [[ "$(find "$RUN" -maxdepth 1 -type f -name "$RECORD" | wc -l)" -eq 1 ]] || fail "expected exactly one $RECORD"
  check_record "$RUN/$RECORD" production
  mkdir -p "$EVIDENCE/manifest"; printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$TARGET/BLD/bin/nemo.exe" "$TARGET/MY_SRC/stprk3_stg.F90" \
    "$TARGET/BLD/ppsrc/nemo/stprk3_stg.f90" "$DECK" "$ICE_DECK" "$RUN/$RECORD" \
    >"$EVIDENCE/manifest/producers.sha256"
  printf 'G9R52_STAGE2_ACQUISITION_OK passivity_records=%s record=%s evidence=%s\n' "$n" "$RECORD" "$EVIDENCE"
}

preflight
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD_BUILD %s from %s; run two G9 steps; write %s in %s\n' "$NAME" "$BASE_NAME" "$RECORD" "$RUN"
  exit 0
fi
if [[ "$mode" == --admit-existing ]]; then admit; exit 0; fi
mkdir -p "$EVIDENCE"; build; run_walk; admit
