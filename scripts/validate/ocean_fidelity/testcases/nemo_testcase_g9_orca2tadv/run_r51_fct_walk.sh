#!/usr/bin/env bash
# Grid ladder G9 round 6 (gridladder_r51) acquisition -- OPERATOR-EXECUTED ONLY.
# Default is read-only preflight.  --run builds one additive, WRITE-only NEMO
# observer and runs the committed G9 deck for two steps.
#
# Round 46 records the exact kt-1 stage-3 FCT entry operands and the final
# post-advection RHS, but no NEMO value at the first divergence, midpoint,
# averaged-flux divergence, pre/post-limiter faces, or final correction.  The
# existing record therefore cannot identify the owner of the remaining at-bar
# 21 T / 17 S cells.  This observer writes those boundaries without reading
# any captured value back into NEMO.  Admission requires all pre-existing kt-1
# and kt-2 records to remain byte-identical to round 46.
#
# The agent that wrote this ran neither makenemo nor NEMO.
set -Eeuo pipefail

fail() { printf 'REFUSE: G9 r51 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly BASE_NAME=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly NAME=ORCA2G9_OMIP_L4_R51FCTWALK
readonly BASE=$NEMO_ROOT/cfgs/$BASE_NAME TARGET=$NEMO_ROOT/cfgs/$NAME
readonly BASE_BIN_SHA=41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275
readonly WANT_KEYS='key_RK3 key_qco key_si3 key_vco_1d3d'
readonly ROUNDS=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds
readonly INPUT_ROOT=$ROUNDS/round21/oracle_g4_r21/p6_a
readonly R46=$ROUNDS/round46/oracle_g9_r46/twin_a
readonly EVIDENCE=${EVIDENCE:-$ROUNDS/round51/oracle_g9_r51}
readonly RUN=$EVIDENCE/twin_walk
readonly TRA_SRC=$NEMO_ROOT/src/OCE/TRA/traadv_fct.F90
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

sha() { sha256sum "$1" | awk '{print $1}'; }
sorted_keys() { sed -E 's/.*fppkeys *//' "$1" | xargs -n1 | LC_ALL=C sort | xargs; }

# Add only observer declarations and writes.  No existing NEMO statement is
# removed or replaced; the compiled-source check below enforces that property.
patch_fct() { # target traadv_fct.F90
  python3 -I - "$1" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
def add_after(old, addition):
    global s
    if s.count(old) != 1:
        raise SystemExit(f"{p}: non-unique after anchor ({s.count(old)}): {old[:80]!r}")
    s = s.replace(old, old + addition)

add_after(
"   REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   ztmp, zwdia, zwinf, zwsup\n",
"   LOGICAL , SAVE ::   l_r51_fct = .FALSE.   ! G9 r51 WRITE-only observer\n"
"   INTEGER , SAVE ::   n_r51_fct = -1\n")
add_after(
"      INTEGER  ::   ji, jj, jk, jn                 ! dummy loop indices\n",
"      INTEGER  ::   i_r51_ios\n"
"      CHARACTER(LEN=256) :: c_r51_file\n")
add_after(
"      DO jn = 1, kjpt            !==  loop over the tracers  ==!\n         !\n",
"         l_r51_fct = lwp .AND. .NOT.l_istiled .AND. cdtype == 'TRA' .AND. &\n"
"            & kt == kit000 .AND. Kbb == 1 .AND. Kmm == 2 .AND. Kaa == 3\n"
"         IF( l_r51_fct ) THEN\n"
"            WRITE(c_r51_file,'(\"oracle_g9_fct_walk_kt\",I8.8,\"_tr\",I1,\".bin\")') kt, jn\n"
"            OPEN( NEWUNIT=n_r51_fct, FILE=TRIM(c_r51_file), ACCESS='STREAM', &\n"
"               & FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=i_r51_ios )\n"
"            IF( i_r51_ios /= 0 ) CALL ctl_stop( 'cannot open G9 r51 FCT walk' )\n"
"            WRITE(n_r51_fct) 'NEMO_G9_FCTW_1 ', 1, kt, jn, Kbb, Kmm, Kaa, Krhs, &\n"
"               & jpi, jpj, jpk, nn_hls, STORAGE_SIZE(1._wp), 20\n"
"            WRITE(n_r51_fct) 'entry           ', pU, pV, pW, pt(:,:,:,jn,Kbb), pt(:,:,:,jn,Krhs)\n"
"         ENDIF\n")
add_after(
"         CALL fct_up1_2stp( Kbb, Kmm, Kaa, p2dt, pt(:,:,:,jn,Kbb), pU, pV, pW, ztFu, ztFv, ztFw, zta_up1, pt(:,:,:,jn,Krhs) )\n",
"")
add_after(
"         ! -- Flux limiter\n         ! ---------------\n",
"         IF( l_r51_fct ) WRITE(n_r51_fct) 'anti_pre        ', ztFu, ztFv, ztFw\n")
add_after(
"         CALL nonosc( Kaa, pt(:,:,:,jn,Kbb), ztFu, ztFv, ztFw, zta_up1, p2dt )\n",
"         IF( l_r51_fct ) WRITE(n_r51_fct) 'anti_post       ', ztFu, ztFv, ztFw\n")
add_after(
"         END_3D\n         !\n         IF ( ll_zAimp2 ) THEN\n",
"         IF( l_r51_fct ) THEN\n"
"            WRITE(n_r51_fct) 'rhs_final       ', pt(:,:,:,jn,Krhs)\n"
"            CLOSE(n_r51_fct)\n"
"            n_r51_fct = -1 ; l_r51_fct = .FALSE.\n"
"         ENDIF\n")
add_after(
"         pt_up1(ji,jj,jk) = ( e3t(ji,jj,jk,Kbb) * pt_b(ji,jj,jk) + zDt * ztra ) / e3t(ji,jj,jk,Kmm) * tmask(ji,jj,jk) !!clem\n      END_3D\n",
"      IF( l_r51_fct ) WRITE(n_r51_fct) 'first_midpoint  ', ptFu, ptFv, ptFw, pt_up1\n")
add_after(
"         pt_up1(ji,jj,jk) = ( e3t(ji,jj,jk,Kbb) * pt_b(ji,jj,jk) + zDt * ztra ) / e3t(ji,jj,jk,Kaa) * tmask(ji,jj,jk) !!clem\n      END_3D\n",
"      IF( l_r51_fct ) WRITE(n_r51_fct) 'average_rhs     ', ptFu, ptFv, ptFw, pt_rhs\n")
p.write_text(s)
PY
}

check_additive() { # base source, patched source
  python3 -I - "$1" "$2" <<'PY'
import difflib, pathlib, sys
a = pathlib.Path(sys.argv[1]).read_text().splitlines()
b = pathlib.Path(sys.argv[2]).read_text().splitlines()
removed = [line for line in difflib.ndiff(a, b) if line.startswith('- ')]
if removed:
    raise SystemExit(f"observer is not additive: {removed[:3]}")
print('FCT_OBSERVER_ADDITIVE_OK')
PY
}

preflight() {
  local f dry
  [[ -x "$BASE/BLD/bin/nemo.exe" && "$(sha "$BASE/BLD/bin/nemo.exe")" == "$BASE_BIN_SHA" ]] \
    || fail 'round-46 record binary base is missing or changed'
  [[ "$(sorted_keys "$BASE/cpp_$BASE_NAME.fcm")" == "$WANT_KEYS" ]] || fail 'base key set changed'
  [[ "$(sha "$DECK")" == 570db996c34e98ccdabf3c367455c1ee6d0b4f18df9f3bdd7264cf053f1f0720 ]] \
    || fail 'committed G9 deck changed'
  [[ "$(sha "$ICE_DECK")" == 9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8 ]] \
    || fail 'committed G9 ice deck changed'
  for f in "${INPUT_FILES[@]}"; do [[ -f "$INPUT_ROOT/$f" ]] || fail "missing input $f"; done
  for f in oracle_rktracer_stage3_kt00000001.bin oracle_tracer_transport_kt00000001_s3.bin \
           oracle_stage_kt00000002_s3.bin oracle_step_entry_kt00000002.bin; do
    [[ -f "$R46/$f" ]] || fail "round-46 record lacks $f"
  done
  grep -Fq 'SUBROUTINE fct_up1_2stp' "$TRA_SRC" || fail 'NEMO FCT source moved'
  dry=$(mktemp -d /tmp/g9-r51-fct.XXXXXX)
  cp "$TRA_SRC" "$dry/traadv_fct.F90"
  patch_fct "$dry/traadv_fct.F90"
  check_additive "$TRA_SRC" "$dry/traadv_fct.F90"
  grep -Fq "'first_midpoint  '" "$dry/traadv_fct.F90" || fail 'first boundary absent'
  grep -Fq "'average_rhs     '" "$dry/traadv_fct.F90" || fail 'second boundary absent'
  grep -Fq "'anti_post       '" "$dry/traadv_fct.F90" || fail 'limiter boundary absent'
  rm -r "$dry"
  printf 'G9R51_PREFLIGHT_OK base=%s target=%s commit=%s\n' "$BASE_NAME" "$NAME" "$COMMIT"
}

build() {
  [[ ! -e "$TARGET" ]] || fail "target config exists: $TARGET"
  (cd "$NEMO_ROOT" && ./makenemo -r ORCA2_ICE_PISCES -n "$NAME" -m conda-scalarmath del_key key_xios)
  rm -f "$TARGET/MY_SRC/"*
  cp -a "$BASE/MY_SRC/." "$TARGET/MY_SRC/"
  cp "$BASE/cpp_$BASE_NAME.fcm" "$TARGET/cpp_$NAME.fcm"
  cp "$TRA_SRC" "$TARGET/MY_SRC/traadv_fct.F90"
  patch_fct "$TARGET/MY_SRC/traadv_fct.F90"
  touch "$TARGET/MY_SRC/"*.F90
  (cd "$NEMO_ROOT" && ./makenemo -n "$NAME" -m conda-scalarmath -y --clean)
  (cd "$NEMO_ROOT" && ./makenemo -n "$NAME" -m conda-scalarmath del_key key_xios)
  [[ -x "$TARGET/BLD/bin/nemo.exe" ]] || fail 'build produced no executable'
  [[ "$(sorted_keys "$TARGET/cpp_$NAME.fcm")" == "$WANT_KEYS" ]] || fail 'target key set changed'
  check_additive "$BASE/BLD/ppsrc/nemo/traadv_fct.f90" "$TARGET/BLD/ppsrc/nemo/traadv_fct.f90"
  for f in $(cd "$BASE/BLD/ppsrc/nemo" && ls *.f90); do
    [[ "$f" == traadv_fct.f90 ]] && continue
    cmp -s "$BASE/BLD/ppsrc/nemo/$f" "$TARGET/BLD/ppsrc/nemo/$f" || fail "ppsrc $f differs"
  done
}

run_walk() {
  local f rc=0
  [[ ! -e "$RUN" ]] || fail "run directory exists: $RUN"
  mkdir -p "$RUN"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$RUN/$f"; done
  cp "$ICE_REF" "$RUN/namelist_ice_ref"; cp "$ICE_DECK" "$RUN/namelist_ice_cfg"
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
  for f in $(cd "$R46" && ls oracle_*kt0000000[12]*.bin); do
    cmp -s "$R46/$f" "$RUN/$f" || fail "pre-existing record changed: $f"
    n=$((n + 1))
  done
  ((n >= 10)) || fail "only $n passivity records compared"
  python3 -I - "$RUN" <<'PY'
import pathlib, struct, sys
import numpy as np
root = pathlib.Path(sys.argv[1])
for tracer in (1, 2):
    path = root / f'oracle_g9_fct_walk_kt00000001_tr{tracer}.bin'
    with path.open('rb') as h:
        magic = h.read(16).decode('ascii')
        head = struct.unpack('=13i', h.read(52))
        payload = h.read()
    version, kt, tr, kbb, kmm, kaa, krhs, nx, ny, nz, halo, bits, arrays = head[:13]
    n3 = nx * ny * nz
    expected = 6 * 16 + arrays * n3 * 8
    if magic != 'NEMO_G9_FCTW_1 ' or (version, kt, tr, kbb, kmm, kaa, bits, arrays) != (1, 1, tracer, 1, 2, 3, 64, 20):
        raise SystemExit(f'bad header {path.name}: {magic!r} {head}')
    if len(payload) != expected:
        raise SystemExit(f'bad payload {path.name}: {len(payload)}/{expected}')
    # Skip five 16-byte labels; all remaining words must be finite fp64.
    at = 0
    for count in (5, 4, 4, 3, 3, 1):
        label = payload[at:at+16].decode('ascii').rstrip(); at += 16
        values = np.frombuffer(payload, dtype=np.float64, count=count*n3, offset=at)
        if not label or values.size != count*n3 or not np.isfinite(values).all():
            raise SystemExit(f'bad frame {path.name}: {label!r}')
        at += values.nbytes
    print(f'FCT_WALK_RECORD_OK tracer={tracer} arrays={arrays}')
PY
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$TARGET/BLD/bin/nemo.exe" "$TARGET/MY_SRC/traadv_fct.F90" \
    "$TARGET/BLD/ppsrc/nemo/traadv_fct.f90" "$DECK" "$ICE_DECK" \
    >"$EVIDENCE/manifest/producers.sha256"
  printf 'G9R51_ACQUISITION_PASS passivity_records=%s evidence=%s\n' "$n" "$EVIDENCE"
}

preflight
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD_BUILD %s from %s; run two G9 steps; write kt1 stage3 FCT boundaries in %s\n' \
    "$NAME" "$BASE_NAME" "$RUN"
  exit 0
fi
if [[ "$mode" == --admit-existing ]]; then admit; exit 0; fi
mkdir -p "$EVIDENCE"
build
run_walk
admit
