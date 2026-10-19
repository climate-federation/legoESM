#!/usr/bin/env bash
# Grid ladder G6 round-6 (gridladder_r42) acquisition -- OPERATOR-EXECUTED ONLY.
# Default is read-only preflight.  --run builds ONE new config and runs G6's
# committed deck for 4 steps.
#
# Why: with Decision 132 landed, kt 1 is at the bar and the first non-bit
# boundary NEMO records is stp_2D's exit at kt 2/3 (uu_b/vv_b, ~2000 cells at
# one ulp) from NEMO's own restart entry.  NEMO records nothing INSIDE stp_2D
# after kt 1 (the slow-forcing dump in stp2d.F90 is gated kt == nit000), and
# G6 starts from rest, so every stp_2D momentum-advection term saw zero
# velocity at kt 1.  This build adds a WRITE-only dump of the stp_2D 3-D
# right-hand side after HPG, after VOR, the first wzv's ww with its sea-level
# operands, after KEG and after ZAD (oracle_stp2d_terms_kt*.bin), and extends
# the existing slow-forcing dump to kt <= nit000 + 2.  Nothing else changes.
#
# Build = G5 round-31 record config ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1 (G6's
# record binary, pinned by SHA-256) with its MY_SRC and cpp keys copied
# verbatim; only MY_SRC/stp2d.F90 is patched (additive; guards widened only
# on the five write-only slow-forcing blocks).  Passivity: every FIX1 record
# file kt 1..4 equals round 36's twin_a byte for byte, and restarts 2 and 4
# equal round 36's writer-free reference in every variable.
# The agent that wrote this ran neither makenemo nor NEMO.
set -Eeuo pipefail

fail() { printf 'REFUSE: G6 r42 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly BASE_NAME=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly NAME=ORCA2G6_OMIP_L4_R42STP2D
readonly BASE=$NEMO_ROOT/cfgs/$BASE_NAME TARGET=$NEMO_ROOT/cfgs/$NAME
readonly BASE_BIN_SHA=41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275
readonly G6_DECK_SHA=243fb53106deb926b454a7fdec984432c905252010953c7403bacbbbbe3b482b
readonly G5_ICE_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly WANT_KEYS='key_RK3 key_qco key_si3 key_vco_1d3d'
readonly ROUNDS=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds
readonly INPUT_ROOT=$ROUNDS/round21/oracle_g4_r21/p6_a
readonly R36=$ROUNDS/round36/oracle_g6_r36
readonly EVIDENCE=${EVIDENCE:-$ROUNDS/round42/oracle_g6_r42}
readonly RUN=$EVIDENCE/twin_s
readonly STEPS=4
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
  for f in 1 2 3 4; do
    [[ -f "$R36/twin_a/$(printf 'oracle_stage_kt%08d_s1.bin' "$f")" ]] || fail 'round-36 twin_a incomplete'
  done
  [[ -f "$R36/ref_100step/ORCA2G1_00000002_restart.nc" && -f "$R36/ref_100step/ORCA2G1_00000004_restart.nc" ]] \
    || fail 'round-36 writer-free restarts missing'
  # the term dump opens one file per call: tiling would open it per tile
  ! grep -Eiq '^[[:space:]]*ln_tile[[:space:]]*=[[:space:]]*\.true\.' "$DECK" || fail 'G6 deck tiles'
  grep -Fq "CALL wzv    ( kt, Kbb, Kbb, Kaa , uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )" \
    "$BASE/MY_SRC/stp2d.F90" || fail 'base stp2d.F90 first wzv call moved'
  printf 'G6R42_PREFLIGHT_OK base=%s target=%s commit=%s\n' "$BASE_NAME" "$NAME" "$COMMIT"
}

patch_stp2d() { # file
  python3 -I - "$1" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
def once(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique ({s.count(old)}): {old[:60]!r}")
    s = s.replace(old, new)
REC = ("            WRITE(g6_unit) l4_canon_3d(uu(:,:,:,Krhs),'U'), "
       "l4_canon_3d(vv(:,:,:,Krhs),'V')\n")
G = "         IF( lwp .AND. kt <= nit000 + 2 ) THEN\n"
E = "         ENDIF\n"
once("      CHARACTER(LEN=256) :: l2_slow_filename\n",
     "      CHARACTER(LEN=256) :: l2_slow_filename\n"
     "      INTEGER  ::   g6_unit, g6_ios        ! G6 r42 stp_2D term dump (WRITE-only)\n"
     "      CHARACTER(LEN=256) :: g6_filename\n")
once("         CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )            ! horizontal gradient of Hydrostatic pressure\n",
     "         CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )            ! horizontal gradient of Hydrostatic pressure\n"
     + G +
     "            WRITE(g6_filename,'(\"oracle_stp2d_terms_kt\",I8.8,\".bin\")') kt\n"
     "            OPEN( NEWUNIT=g6_unit, FILE=TRIM(g6_filename), ACCESS='STREAM', FORM='UNFORMATTED', &\n"
     "               &  STATUS='REPLACE', ACTION='WRITE', IOSTAT=g6_ios )\n"
     "            IF( g6_ios /= 0 )   CALL ctl_stop( 'cannot open G6 stp_2D term dump' )\n"
     "            WRITE(g6_unit) 'NEMO_G6_STP2D_1 ', 1, kt, Kbb, Kaa, Krhs, jpi, jpj, jpk, STORAGE_SIZE(1._wp)\n"
     + REC + E)
once("         CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )\n",
     "         CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )\n" + G + REC + E)
once("         CALL wzv    ( kt, Kbb, Kbb, Kaa , uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )  ! ww guess at Kbb (n)\n",
     "         CALL wzv    ( kt, Kbb, Kbb, Kaa , uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )  ! ww guess at Kbb (n)\n"
     + G +
     "            WRITE(g6_unit) l4_canon_3d(ww,'T'), l4_canon_2d(ssh(:,:,Kaa),'T'), &\n"
     "               & l4_canon_2d(ssh(:,:,Kbb),'T'), l4_canon_2d(r3t(:,:,Kaa),'T'), &\n"
     "               & l4_canon_2d(r3t(:,:,Kbb),'T'), l4_canon_2d(r3u(:,:,Kbb),'U'), &\n"
     "               & l4_canon_2d(r3v(:,:,Kbb),'V')\n" + E)
once("            CALL dyn_keg( kt, nn_dynkeg, Kbb, uu, vv, Krhs )      !- horizontal Gradient of KE\n",
     "            CALL dyn_keg( kt, nn_dynkeg, Kbb, uu, vv, Krhs )      !- horizontal Gradient of KE\n"
     + G.replace("         ", "            ", 1) + "   " + REC + "   " + E)
once("            CALL dyn_zad( kt, Kbb, uu, vv, Krhs )                 !- vertical advection\n",
     "            CALL dyn_zad( kt, Kbb, uu, vv, Krhs )                 !- vertical advection\n"
     + G.replace("         ", "            ", 1) + "   " + REC +
     "               CLOSE(g6_unit)\n"
     "               WRITE(numout,*) 'G6_STP2D_TERMS_DUMP ', kt, TRIM(g6_filename)\n"
     + "   " + E)
n = s.count("IF( lwp .AND. kt == nit000 )")
if n != 5:
    raise SystemExit(f"expected 5 write-only slow-forcing guards, found {n}")
s = s.replace("IF( lwp .AND. kt == nit000 )", "IF( lwp .AND. kt <= nit000 + 2 )")
open(p, "w").write(s)
PY
}

check_ppsrc_additive() { # base ppsrc, new ppsrc
  python3 -I - "$1" "$2" <<'PY'
import difflib, sys
a = open(sys.argv[1]).read().splitlines(); b = open(sys.argv[2]).read().splitlines()
removed = [l for l in difflib.ndiff(a, b) if l.startswith("- ")]
bad = [l for l in removed if "IF( lwp .AND. kt == nit000 )" not in l]
if bad or len(removed) != 5:
    raise SystemExit(f"compiled stp2d.f90 is not additive: {len(removed)} removed, {bad[:3]}")
print("STP2D_ADDITIVE_OK removed_guards=5")
PY
}

build() {
  [[ ! -e "$TARGET" ]] || fail "target config already exists: $TARGET"
  (cd "$NEMO_ROOT" && ./makenemo -r ORCA2_ICE_PISCES -n "$NAME" -m conda-scalarmath del_key key_xios)
  rm -f "$TARGET/MY_SRC/"*
  cp -a "$BASE/MY_SRC/." "$TARGET/MY_SRC/"
  cp "$BASE/cpp_$BASE_NAME.fcm" "$TARGET/cpp_$NAME.fcm"
  diff -rq "$BASE/MY_SRC" "$TARGET/MY_SRC" >/dev/null || fail 'MY_SRC copy differs'
  patch_stp2d "$TARGET/MY_SRC/stp2d.F90"
  [[ "$(diff -rq "$BASE/MY_SRC" "$TARGET/MY_SRC" | wc -l)" == 1 ]] || fail 'more than stp2d.F90 changed'
  touch "$TARGET/MY_SRC/"*.F90
  (cd "$NEMO_ROOT" && ./makenemo -n "$NAME" -m conda-scalarmath -y --clean)
  (cd "$NEMO_ROOT" && ./makenemo -n "$NAME" -m conda-scalarmath del_key key_xios)
  [[ -x "$TARGET/BLD/bin/nemo.exe" ]] || fail 'no executable'
  [[ "$(sorted_keys "$TARGET/cpp_$NAME.fcm")" == "$WANT_KEYS" ]] || fail 'key set changed'
  check_ppsrc_additive "$BASE/BLD/ppsrc/nemo/stp2d.f90" "$TARGET/BLD/ppsrc/nemo/stp2d.f90"
  local f
  for f in $(cd "$BASE/BLD/ppsrc/nemo" && ls *.f90); do
    [[ "$f" == stp2d.f90 ]] && continue
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
         -e "s/^(   nn_stocklist = ).*/\12,4,0,0,0,0,0,0,0,0/" "$DECK" >"$RUN/namelist_cfg"
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
  for f in $(cd "$R36/twin_a" && ls oracle_*kt0000000[1-4]*.bin); do
    cmp -s "$R36/twin_a/$f" "$RUN/$f" || fail "record $f differs from round 36 (writer not passive)"
    n=$((n + 1))
  done
  ((n >= 16)) || fail "only $n round-36 record files compared"
  # passivity 2: restarts 2 and 4 equal the writer-free reference, every variable
  python3 -I - "$RUN" "$R36/ref_100step" <<'PY'
import sys, netCDF4, numpy as np
for kt in (2, 4):
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
  # the new term dump: kt 1..3, full payload, finite
  python3 -I - "$RUN" <<'PY'
import pathlib, struct, sys
import numpy as np
root = pathlib.Path(sys.argv[1])
for kt in (1, 2, 3):
    for name, nd3, nd2, magic_want in ((f"oracle_stp2d_terms_kt{kt:08d}.bin", 9, 6, "NEMO_G6_STP2D_1"),):
        path = root / name
        with path.open("rb") as h:
            magic = h.read(16).decode("ascii").rstrip()
            head = struct.unpack("=9i", h.read(36))
            vals = np.fromfile(h, dtype=np.float64)
        ver, k, kbb, kaa, krhs, jpi, jpj, jpk, bits = head
        want = nd3 * jpi * jpj * jpk + nd2 * jpi * jpj
        if magic != magic_want or (ver, k, bits) != (1, kt, 64) or vals.size != want:
            raise SystemExit(f"bad {name}: {magic!r} {head} values={vals.size}/{want}")
        if not np.isfinite(vals).all():
            raise SystemExit(f"non-finite {name}")
    if not (root / f"oracle_slow_forcing_kt{kt:08d}.bin").is_file():
        raise SystemExit(f"missing slow forcing kt {kt}")
    print(f"STP2D_TERMS_OK kt={kt}")
PY
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$TARGET/BLD/bin/nemo.exe" "$TARGET/MY_SRC/stp2d.F90" "$TARGET/BLD/ppsrc/nemo/stp2d.f90" \
    "$DECK" "$ICE_DECK" >"$EVIDENCE/manifest/producers.sha256"
  printf 'G6R42_ACQUISITION_PASS steps=1..%s terms=kt1..3 evidence=%s\n' "$STEPS" "$EVIDENCE"
}

preflight
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD_BUILD %s from %s (stp2d.F90 write-only term dump), run %s steps in %s\n' \
    "$NAME" "$BASE_NAME" "$STEPS" "$RUN"
  exit 0
fi
if [[ "$mode" == --admit-existing ]]; then admit; exit 0; fi
mkdir -p "$EVIDENCE"
build
run_s
admit
