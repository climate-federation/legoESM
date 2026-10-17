#!/usr/bin/env bash
# Grid ladder G6 round-8 (gridladder_r44) acquisition -- OPERATOR-EXECUTED ONLY.
# Default is read-only preflight.  --run builds ONE new config and runs G6's
# committed deck for 6 steps.
#
# Why: after round 8 the given-entry label is at the bar kt 1..6 and the
# independent chain's first departure is kt 1, RK3 stage 2: the stage-2 3-D
# RHS is exact after HPG and VOR (oracle_rkstage2_terms_kt00000001.bin) and
# differs on 3 u cells after dyn_adv (KEG + ZAD).  NEMO records neither the
# RHS between KEG and ZAD nor the stage momentum ww that dyn_zad reads, and
# nothing inside stages 2/3 after kt 1.  This build adds a WRITE-only dump,
# stages 2 and 3, kt 1..6 (oracle_g6_stage_terms_kt*_s*.bin): the stage
# momentum wzv's ww with r3t(Kbb/Kmm/Kaa), r3u/r3v(Kmm), ssh(Kbb/Kmm/Kaa),
# then the 3-D Krhs before HPG, after HPG, after VOR, after KEG and after
# ZAD.  Nothing else changes.
#
# Build = G5 round-31 record config ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1 (G6's
# record binary, pinned by SHA-256) with its MY_SRC and cpp keys copied
# verbatim; MY_SRC/stprk3_stg.F90 is patched (additive) and NEMO's own
# src/OCE/DYN/dynadv.F90 is copied into MY_SRC with one additive write after
# dyn_keg (a public unit, -1 = off, set only around the dump's dyn_adv call).
# Arrays are NATIVE (jpi, jpj, jpk incl. nn_hls=2 halo), Fortran order.
# Passivity: every FIX1 record file kt 1..6 equals round 36's twin_a byte for
# byte, and restarts 2, 4, 6 equal round 36's writer-free reference.
# The agent that wrote this ran neither makenemo nor NEMO.
set -Eeuo pipefail

fail() { printf 'REFUSE: G6 r44 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly BASE_NAME=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly NAME=ORCA2G6_OMIP_L4_R44STGTRM
readonly BASE=$NEMO_ROOT/cfgs/$BASE_NAME TARGET=$NEMO_ROOT/cfgs/$NAME
readonly BASE_BIN_SHA=41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275
readonly G6_DECK_SHA=243fb53106deb926b454a7fdec984432c905252010953c7403bacbbbbe3b482b
readonly G5_ICE_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly WANT_KEYS='key_RK3 key_qco key_si3 key_vco_1d3d'
readonly ROUNDS=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds
readonly INPUT_ROOT=$ROUNDS/round21/oracle_g4_r21/p6_a
readonly R36=$ROUNDS/round36/oracle_g6_r36
readonly EVIDENCE=${EVIDENCE:-$ROUNDS/round44/oracle_g6_r44}
readonly DYNADV_SRC=$NEMO_ROOT/src/OCE/DYN/dynadv.F90
readonly RUN=$EVIDENCE/twin_s
readonly STEPS=6
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

preflight() {
  local f
  [[ -x "$BASE/BLD/bin/nemo.exe" && "$(sha "$BASE/BLD/bin/nemo.exe")" == "$BASE_BIN_SHA" ]] \
    || fail 'G6 record binary (G5 round-31 FIX1) missing or changed'
  [[ "$(sorted_keys "$BASE/cpp_$BASE_NAME.fcm")" == "$WANT_KEYS" ]] || fail 'base key set changed'
  [[ "$(sha "$DECK")" == "$G6_DECK_SHA" && "$(sha "$ICE_DECK")" == "$G5_ICE_SHA" ]] \
    || fail 'committed G6 deck changed'
  for f in "${INPUT_FILES[@]}"; do [[ -f "$INPUT_ROOT/$f" ]] || fail "missing input $f"; done
  for f in 1 2 3 4 5 6; do
    [[ -f "$R36/twin_a/$(printf 'oracle_stage_kt%08d_s1.bin' "$f")" ]] || fail 'round-36 twin_a incomplete'
  done
  [[ -f "$R36/ref_100step/ORCA2G1_00000002_restart.nc" && -f "$R36/ref_100step/ORCA2G1_00000004_restart.nc" \
     && -f "$R36/ref_100step/ORCA2G1_00000006_restart.nc" ]] \
    || fail 'round-36 writer-free restarts missing'
  # the term dump opens one file per call: tiling would open it per tile
  ! grep -Eiq '^[[:space:]]*ln_tile[[:space:]]*=[[:space:]]*\.true\.' "$DECK" || fail 'G6 deck tiles'
  grep -Fq "CALL wzv( kstp, Kbb, Kmm, Kaa, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, np_velocity )" \
    "$BASE/MY_SRC/stprk3_stg.F90" || fail 'base stprk3_stg.F90 stage momentum wzv call moved'
  [[ ! -e "$BASE/MY_SRC/dynadv.F90" ]] || fail 'base already overrides dynadv.F90'
  grep -Fq "CALL dyn_keg     ( kt, nn_dynkeg     , Kmm, puu, pvv, Krhs )" "$DYNADV_SRC" \
    || fail 'NEMO dynadv.F90 dyn_keg call moved'
  printf 'G6R44_PREFLIGHT_OK base=%s target=%s commit=%s\n' "$BASE_NAME" "$NAME" "$COMMIT"
}

patch_stage() { # stprk3_stg.F90 dynadv.F90
  python3 -I - "$1" "$2" <<'PY'
import sys
def edit(p, pairs):
    s = open(p).read()
    for old, new in pairs:
        if s.count(old) != 1:
            raise SystemExit(f"{p}: anchor not unique ({s.count(old)}): {old[:60]!r}")
        s = s.replace(old, new)
    open(p, "w").write(s)
G = "IF( ll_g6s ) "
REC = "WRITE(g6s_unit) uu(:,:,:,Krhs), vv(:,:,:,Krhs)\n"
edit(sys.argv[1], [
 ("#endif\n      !! ---------------------------------------------------------------------\n\n      IF( ln_timing )   CALL timing_start('stp_RK3_stg')",
  "#endif\n      LOGICAL  ::   ll_g6s                 ! G6 r44 stage-term dump (WRITE-only)\n"
  "      INTEGER  ::   g6s_unit, g6s_ios\n      CHARACTER(LEN=256) :: g6s_filename\n"
  "      !! ---------------------------------------------------------------------\n\n      IF( ln_timing )   CALL timing_start('stp_RK3_stg')\n"
  "      ll_g6s = .FALSE."),
 ("               CALL wzv( kstp, Kbb, Kmm, Kaa, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, np_velocity )\n",
  "               CALL wzv( kstp, Kbb, Kmm, Kaa, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, np_velocity )\n"
  "               ll_g6s = lwp .AND. .NOT.ln_tile .AND. kstp <= nit000 + 5 .AND. kstg >= 2\n"
  "               IF( ll_g6s ) THEN\n"
  "                  WRITE(g6s_filename,'(\"oracle_g6_stage_terms_kt\",I8.8,\"_s\",I1,\".bin\")') kstp, kstg\n"
  "                  OPEN( NEWUNIT=g6s_unit, FILE=TRIM(g6s_filename), ACCESS='STREAM', FORM='UNFORMATTED', &\n"
  "                     &  STATUS='REPLACE', ACTION='WRITE', IOSTAT=g6s_ios )\n"
  "                  IF( g6s_ios /= 0 )   CALL ctl_stop( 'cannot open G6 stage-term dump' )\n"
  "                  WRITE(g6s_unit) 'NEMO_G6_STGTRM_1', 1, kstp, kstg, Kbb, Kmm, Kaa, Krhs, jpi, jpj, jpk, nn_hls, STORAGE_SIZE(1._wp)\n"
  "                  WRITE(g6s_unit) ww, r3t(:,:,Kbb), r3t(:,:,Kmm), r3t(:,:,Kaa), r3u(:,:,Kmm), r3v(:,:,Kmm), &\n"
  "                     &  ssh(:,:,Kbb), ssh(:,:,Kmm), ssh(:,:,Kaa)\n"
  "               ENDIF\n"),
 ("            CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )  ! Hydrostratic Pressure Gradient (HPG)\n",
  "            " + G + REC +
  "            CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )  ! Hydrostratic Pressure Gradient (HPG)\n"
  "            " + G + REC),
 ("            CALL    dyn_vor( kstp,      Kmm, uu, vv, Krhs )\n",
  "            CALL    dyn_vor( kstp,      Kmm, uu, vv, Krhs )\n            " + G + REC),
 ("               CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)\n",
  "               " + G + "g6_keg_unit = g6s_unit\n"
  "               CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)\n"
  "               IF( ll_g6s ) THEN\n"
  "                  g6_keg_unit = -1\n"
  "                  " + REC +
  "                  CLOSE(g6s_unit)\n"
  "                  WRITE(numout,*) 'G6_STAGE_TERMS_DUMP ', kstp, kstg, TRIM(g6s_filename)\n"
  "               ENDIF\n"),
])
edit(sys.argv[2], [
 ("   REAL(wp), PUBLIC            ::   r_stb_cstra_dyn  ! stability constraint for dynamic advection\n",
  "   REAL(wp), PUBLIC            ::   r_stb_cstra_dyn  ! stability constraint for dynamic advection\n"
  "   INTEGER , PUBLIC            ::   g6_keg_unit = -1 ! G6 r44: WRITE-only after-KEG dump unit (-1 = off)\n"),
 ("         CALL dyn_keg     ( kt, nn_dynkeg     , Kmm, puu, pvv, Krhs )\n",
  "         CALL dyn_keg     ( kt, nn_dynkeg     , Kmm, puu, pvv, Krhs )\n"
  "         IF( g6_keg_unit >= 0 )   WRITE(g6_keg_unit) puu(:,:,:,Krhs), pvv(:,:,:,Krhs)\n"),
])
PY
}

check_ppsrc_additive() { # base ppsrc, new ppsrc
  python3 -I - "$1" "$2" <<'PY'
import difflib, sys
a = open(sys.argv[1]).read().splitlines(); b = open(sys.argv[2]).read().splitlines()
removed = [l for l in difflib.ndiff(a, b) if l.startswith("- ")]
if removed:
    raise SystemExit(f"compiled {sys.argv[2]} is not additive: {removed[:3]}")
print("PPSRC_ADDITIVE_OK", sys.argv[2].rsplit("/", 1)[-1])
PY
}

build() {
  [[ ! -e "$TARGET" ]] || fail "target config already exists: $TARGET"
  (cd "$NEMO_ROOT" && ./makenemo -r ORCA2_ICE_PISCES -n "$NAME" -m conda-scalarmath del_key key_xios)
  rm -f "$TARGET/MY_SRC/"*
  cp -a "$BASE/MY_SRC/." "$TARGET/MY_SRC/"
  cp "$BASE/cpp_$BASE_NAME.fcm" "$TARGET/cpp_$NAME.fcm"
  diff -rq "$BASE/MY_SRC" "$TARGET/MY_SRC" >/dev/null || fail 'MY_SRC copy differs'
  cp "$DYNADV_SRC" "$TARGET/MY_SRC/dynadv.F90"
  patch_stage "$TARGET/MY_SRC/stprk3_stg.F90" "$TARGET/MY_SRC/dynadv.F90"
  [[ "$(diff -rq "$BASE/MY_SRC" "$TARGET/MY_SRC" | wc -l)" == 2 ]] || fail 'more than stprk3_stg.F90 + dynadv.F90 changed'
  touch "$TARGET/MY_SRC/"*.F90
  (cd "$NEMO_ROOT" && ./makenemo -n "$NAME" -m conda-scalarmath -y --clean)
  (cd "$NEMO_ROOT" && ./makenemo -n "$NAME" -m conda-scalarmath del_key key_xios)
  [[ -x "$TARGET/BLD/bin/nemo.exe" ]] || fail 'no executable'
  [[ "$(sorted_keys "$TARGET/cpp_$NAME.fcm")" == "$WANT_KEYS" ]] || fail 'key set changed'
  check_ppsrc_additive "$BASE/BLD/ppsrc/nemo/stprk3_stg.f90" "$TARGET/BLD/ppsrc/nemo/stprk3_stg.f90"
  check_ppsrc_additive "$BASE/BLD/ppsrc/nemo/dynadv.f90" "$TARGET/BLD/ppsrc/nemo/dynadv.f90"
  local f
  for f in $(cd "$BASE/BLD/ppsrc/nemo" && ls *.f90); do
    [[ "$f" == stprk3_stg.f90 || "$f" == dynadv.f90 ]] && continue
    cmp -s "$BASE/BLD/ppsrc/nemo/$f" "$TARGET/BLD/ppsrc/nemo/$f" || fail "ppsrc $f differs"
  done
}

run_s() {
  [[ ! -e "$RUN" ]] || fail "run directory exists: $RUN"
  mkdir -p "$RUN"
  local f rc=0
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$RUN/$f"; done
  cp "$ICE_REF" "$RUN/namelist_ice_ref"; cp "$ICE_DECK" "$RUN/namelist_ice_cfg"
  sed -E -e "s/^(   nn_itend    = ).*/\1    $STEPS/" -e "s/^(   nn_stock    = ).*/\1    $STEPS/" \
         -e "s/^(   nn_stocklist = ).*/\12,4,6,0,0,0,0,0,0,0/" "$DECK" >"$RUN/namelist_cfg"
  cp "$TARGET/BLD/bin/nemo.exe" "$RUN/nemo"
  (cd "$RUN"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$RUN/run.exit"
  [[ "$rc" -eq 0 && "$(tr -d '[:space:]' <"$RUN/time.step")" == "$STEPS" ]] || fail "run did not complete $STEPS steps"
  grep -q 'vector form : keg + zad + vor is used' "$RUN/ocean.output" || fail 'not vector form'
}

admit() {
  local kt f n
  # passivity 1: every record file the FIX1 writers produce, kt 1..4, byte-identical to round 36
  n=0
  for f in $(cd "$R36/twin_a" && ls oracle_*kt0000000[1-6]*.bin); do
    cmp -s "$R36/twin_a/$f" "$RUN/$f" || fail "record $f differs from round 36 (writer not passive)"
    n=$((n + 1))
  done
  ((n >= 24)) || fail "only $n round-36 record files compared"
  # passivity 2: restarts 2 and 4 equal the writer-free reference, every variable
  python3 -I - "$RUN" "$R36/ref_100step" <<'PY'
import sys, netCDF4, numpy as np
for kt in (2, 4, 6):
    a = netCDF4.Dataset(f"{sys.argv[1]}/ORCA2G1_{kt:08d}_restart.nc")
    b = netCDF4.Dataset(f"{sys.argv[2]}/ORCA2G1_{kt:08d}_restart.nc")
    names = [v for v in b.variables if b.variables[v].ndim >= 2]
    if set(names) - set(a.variables):
        raise SystemExit(f"kt {kt}: restart variables missing")
    for v in names:
        if not np.array_equal(np.asarray(a.variables[v][:]), np.asarray(b.variables[v][:])):
            raise SystemExit(f"kt {kt}: restart {v} differs (writer not passive)")
    print(f"RESTART_PASSIVE kt={kt} variables={len(names)}")
PY
  # the new stage-term dump: kt 1..6, stages 2 and 3, full payload, finite
  python3 -I - "$RUN" <<'PY'
import pathlib, struct, sys
import numpy as np
root = pathlib.Path(sys.argv[1])
for kt in range(1, 7):
    for stg in (2, 3):
        path = root / f"oracle_g6_stage_terms_kt{kt:08d}_s{stg}.bin"
        with path.open("rb") as h:
            magic = h.read(16).decode("ascii")
            head = struct.unpack("=12i", h.read(48))
            vals = np.fromfile(h, dtype=np.float64)
        ver, k, s, kbb, kmm, kaa, krhs, jpi, jpj, jpk, hls, bits = head
        want = (1 + 2 * 5) * jpi * jpj * jpk + 8 * jpi * jpj
        if magic != "NEMO_G6_STGTRM_1" or (ver, k, s, bits, hls) != (1, kt, stg, 64, 2) or vals.size != want:
            raise SystemExit(f"bad {path.name}: {magic!r} {head} values={vals.size}/{want}")
        if not np.isfinite(vals).all():
            raise SystemExit(f"non-finite {path.name}")
    print(f"STAGE_TERMS_OK kt={kt}")
PY
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$TARGET/BLD/bin/nemo.exe" "$TARGET/MY_SRC/stprk3_stg.F90" "$TARGET/MY_SRC/dynadv.F90" "$TARGET/BLD/ppsrc/nemo/stprk3_stg.f90" "$TARGET/BLD/ppsrc/nemo/dynadv.f90" \
    "$DECK" "$ICE_DECK" >"$EVIDENCE/manifest/producers.sha256"
  printf 'G6R44_ACQUISITION_PASS steps=1..%s stage_terms=kt1..6,s2-3 evidence=%s\n' "$STEPS" "$EVIDENCE"
}

preflight
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD_BUILD %s from %s (stprk3_stg.F90 + dynadv.F90 write-only stage-term dump), run %s steps in %s\n' \
    "$NAME" "$BASE_NAME" "$STEPS" "$RUN"
  exit 0
fi
if [[ "$mode" == --admit-existing ]]; then admit; exit 0; fi
mkdir -p "$EVIDENCE"
build
run_s
admit
