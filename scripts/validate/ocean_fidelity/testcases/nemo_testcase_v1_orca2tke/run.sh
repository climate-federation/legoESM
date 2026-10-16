#!/usr/bin/env bash
# Vertical-mixing lane V1 round-1 acquisition (operator executed only).
# V1 = certified G5 + ORCA2's TKE closure (ln_zdftke=T, ln_zdfcst=F, ORCA2's
# &namzdf_tke).  TKE is a run-time switch: zdftke.f90 and zdfphy.f90 compiled
# into G5's binaries are byte-identical to ORCA2_OMIP_L4's, so no cpp key moves.
#  - writer-free reference: G5's pinned writer-free binary, reused UNBUILT;
#  - record build: G5's record MY_SRC set, plus ONE additive write-only
#    writer in stprk3.F90 for SI3's mean ice thickness hm_i, which TKE reads
#    under nn_mxlice=2 (zdftke.f90:620-623) and which G5's record lacks.
# TKE's own state (en, avt_k, avm_k, dissl) is in NEMO's restart (tke_rst,
# zdftke.f90:922-926), so restarts at every writable step carry it.
# Default is read-only preflight.  --run builds, smokes, finds NEMO's
# boundary with the writer-free binary, then acquires three twins.
set -Eeuo pipefail

fail() { printf 'REFUSE: V1 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly G5_TOOL=$here/../nemo_testcase_g5_orca2ice/run.sh
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly G5_SRC=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly REF_BIN=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe
readonly TARGET_NAME=ORCA2V1_OMIP_L4_R1FULLENTRY
readonly TARGET=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly REF_BIN_SHA=e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7
readonly G5_REC_BIN_SHA=41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275
readonly G5_MYSRC_SET_SHA=28e265096d11d22c4f342b342b3065575115dcf38aaa0c60fe21330cd3f14607
readonly G5_STPRK3_SHA=fa8a0f6c18d0c08cee3d74a1d379125614ccd805193de4e9f7ea0abb5812f3f3
readonly G5_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly V1_DECK_SHA=7eec2924000da6472780619c9ecb82bd83684aa4678ce33472f56eb1b1e1fcad
readonly ICE_DECK_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref
readonly FSBC=2
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round1/oracle_v1_r1}
readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly TWIN_A=$EVIDENCE/twin_a TWIN_B=$EVIDENCE/twin_b TWIN_C=$EVIDENCE/twin_c
readonly DECK=$here/namelist_cfg ICE_DECK=$here/namelist_ice_cfg
readonly INPUT_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc
  ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc q_10.15JUNE2009_fill.nc
  ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc
  sss_data.nc runoff_core_monthly.nc namelist_ref)
# ponytail: the G5 tool pins every input's SHA-256 already; V1 re-reads them
# from it rather than keeping a second copy that could drift.
input_sha() { sed -nE "s/^  \[$1\]=([0-9a-f]{64})$/\1/p" "$G5_TOOL"; }

sha() { sha256sum "$1" | awk '{print $1}'; }
mysrc_set_sha() {
  (cd "$1" && for f in $(ls | LC_ALL=C sort); do printf '%s  %s\n' "$(sha "$f")" "$f"; done) |
    sha256sum | awk '{print $1}'
}

# The only edit to G5's stprk3.F90: three ADDED blocks, nothing removed.
patch_stprk3() { # in, out
  awk '
    /^   USE sbc_ice, ONLY : snwice_mass, snwice_mass_b, snwice_fmass, rCdU_ice$/ {
      print; print "   USE ice, ONLY : hm_i   ! V1: WRITE-only hm_i record"; n1++; next }
    /^ +CALL l4_dump_ocean_surface_input\( kstp, Nbb \) / {
      print; print "                         CALL v1_dump_hm_i( kstp )                        ! V1: WRITE-only hm_i record"; n2++; next }
    /^#include "l4_oracle_canon_subroutines.h90"$/ {
      print "   SUBROUTINE v1_dump_hm_i( kstp )"
      print "      !! V1 WRITE-only record of SI3 mean ice thickness hm_i after sbc,"
      print "      !! the operand of zdftke tke_avn under nn_mxlice=2.  No model array is set."
      print "      INTEGER, INTENT(in) :: kstp"
      print "      INTEGER :: iunit, ios"
      print "      CHARACTER(LEN=64) :: cl_file"
      print "      CHARACTER(LEN=16) :: magic"
      print "      IF( STORAGE_SIZE( 1._wp ) /= 64 )   CALL ctl_stop( '\''V1 hm_i dump requires 64-bit wp'\'' )"
      print "      IF( mpprank /= 0 )   CALL ctl_stop( '\''V1 hm_i dump is single-rank only'\'' )"
      print "      WRITE(cl_file,'\''(\"oracle_v1_hm_i_kt\",I8.8,\".bin\")'\'') kstp"
      print "      OPEN( NEWUNIT=iunit, FILE=TRIM(cl_file), ACCESS='\''STREAM'\'', &"
      print "         &  FORM='\''UNFORMATTED'\'', STATUS='\''NEW'\'', ACTION='\''WRITE'\'', IOSTAT=ios )"
      print "      IF( ios /= 0 )   CALL ctl_stop( '\''cannot open V1 hm_i file'\'' )"
      print "      magic = '\''NEMO_V1_HMI_1'\''"
      print "      WRITE(iunit) magic, 1, kstp, jpi, jpj, STORAGE_SIZE(1._wp)"
      print "      WRITE(iunit) l4_canon_2d(hm_i,'\''T'\'')"
      print "      CLOSE(iunit)"
      print "   END SUBROUTINE v1_dump_hm_i"
      print ""
      print; n3++; next }
    { print }
    END { if (n1 != 1 || n2 != 1 || n3 != 1) exit 1 }
  ' "$1" >"$2" || fail "stprk3.F90 anchors not found exactly once"
  # additive only: no line of G5's file is removed or changed
  ! diff "$1" "$2" | grep -q '^<' || fail 'hm_i patch removed or changed a G5 line'
}

check_hmi_frames() { # directory, completed steps
  python3 -I - "$1" "$2" <<'PY'
import pathlib, struct, sys
import numpy as np
root, steps = pathlib.Path(sys.argv[1]), int(sys.argv[2])
for kt in range(1, steps + 1):
    path = root / f"oracle_v1_hm_i_kt{kt:08d}.bin"
    with path.open("rb") as stream:
        magic = stream.read(16).decode("ascii").rstrip()
        version, k, jpi, jpj, bits = struct.unpack("=5i", stream.read(20))
        values = np.fromfile(stream, dtype=np.float64)
    if (magic, version, k, bits) != ("NEMO_V1_HMI_1", 1, kt, 64) or values.size != jpi * jpj:
        raise SystemExit(f"bad V1 hm_i frame: {path.name} {magic!r} {version} {k} {bits} {values.size}")
    if not np.isfinite(values).all() or values.min() < 0.0:
        raise SystemExit(f"bad V1 hm_i values: {path.name}")
PY
}

mode=${1:---preflight-only}
if [[ "$mode" == --check-hmi-frames ]]; then
  (($# == 3)) || exit 64
  check_hmi_frames "$2" "$3"; printf 'V1_HMI_FRAMES_OK %s steps=1..%s\n' "$2" "$3"; exit 0
fi
if [[ "$mode" == --patch-stprk3 ]]; then
  (($# == 3)) || exit 64
  patch_stprk3 "$2" "$3"; printf 'V1_STPRK3_PATCH_OK %s\n' "$3"; exit 0
fi
case "$mode" in --preflight-only|--run) ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in worktree'
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] \
  || fail 'commit the exact acquisition tool before running it'
readonly COMMIT=$(git -C "$repo" rev-parse HEAD)

preflight() {
  local f want
  for f in "$G5_TOOL" "$DECK" "$ICE_DECK" "$REF_BIN" "$ICE_REF" \
           "$G5_SRC/BLD/bin/nemo.exe" "$G5_SRC/MY_SRC/stprk3.F90" \
           "$G5_SRC/cpp_ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1.fcm"; do
    [[ -f "$f" ]] || fail "missing $f"
  done
  for f in "${INPUT_FILES[@]}"; do
    want=$(input_sha "$f"); [[ ${#want} == 64 ]] || fail "G5 tool pins no SHA for $f"
    [[ -f "$INPUT_ROOT/$f" && "$(sha "$INPUT_ROOT/$f")" == "$want" ]] \
      || fail "missing or changed input $f"
  done
  [[ "$(sha "$DECK")" == "$V1_DECK_SHA" && "$(sha "$ICE_DECK")" == "$ICE_DECK_SHA" ]] \
    || fail 'committed V1 deck changed'
  [[ "$(sha "$REF_BIN")" == "$REF_BIN_SHA" ]] || fail 'G5 writer-free binary changed'
  [[ "$(sha "$G5_SRC/BLD/bin/nemo.exe")" == "$G5_REC_BIN_SHA" ]] || fail 'G5 record binary changed'
  [[ "$(mysrc_set_sha "$G5_SRC/MY_SRC")" == "$G5_MYSRC_SET_SHA" ]] || fail 'G5 record MY_SRC set changed'
  [[ "$(sha "$G5_SRC/MY_SRC/stprk3.F90")" == "$G5_STPRK3_SHA" ]] || fail 'G5 stprk3.F90 changed'
  [[ "$(sha "$G5_SRC/cpp_ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1.fcm")" == "$G5_CPP_SHA" ]] \
    || fail 'G5 cpp keys changed'
  # TKE is compiled into the reused binaries exactly as in ORCA2_OMIP_L4.
  for f in zdftke.f90 zdfphy.f90; do
    cmp -s "$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/ppsrc/nemo/$f" \
      "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/$f" || fail "$f differs from ORCA2_OMIP_L4"
  done
  "$G5_TOOL" --check-writer-contract "$G5_SRC/MY_SRC/icedyn_adv_pra.F90" "$ICE_REF" "$ICE_DECK" \
    >/dev/null || fail 'G5 ice writer contract no longer holds for the V1 overlay'
  local probe; probe=$(mktemp); patch_stprk3 "$G5_SRC/MY_SRC/stprk3.F90" "$probe"; rm -f "$probe"
  printf 'V1_PREFLIGHT_OK commit=%s reference=G5_writer_free_unbuilt record=%s expected_boundary=kt11_after_10_completed\n' \
    "$COMMIT" "$TARGET_NAME"
}

render_deck() { # target, itend, stocklist
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
         -e "s/^(   nn_stock    = ).*/\1    $2/" \
         -e "s/^(   nn_stocklist = ).*/\1$3/" "$DECK" >"$1"
}
pad10() { local a=("$@"); while ((${#a[@]} < 10)); do a+=(0); done; local IFS=,; printf '%s' "${a[*]}"; }
# restarts at every writable step 2..N across the twins; step 1 is never requested
list_a() { pad10 "$1"; }
list_b() { pad10 $(seq 2 2 "$1"); }
list_c() { (($1 >= 3)) && pad10 $(seq 3 2 "$1") || pad10 0; }

build_target() {
  [[ ! -e "$TARGET" ]] || fail "record target config already exists: $TARGET"
  (cd "$NEMO_ROOT" && ./makenemo -r ORCA2_ICE_PISCES -n "$TARGET_NAME" \
    -m conda-scalarmath del_key key_xios)
  cp "$G5_SRC/cpp_ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1.fcm" "$TARGET/cpp_$TARGET_NAME.fcm"
  find "$TARGET/MY_SRC" -maxdepth 1 -type f -delete
  cp -a "$G5_SRC/MY_SRC/." "$TARGET/MY_SRC/"
  patch_stprk3 "$G5_SRC/MY_SRC/stprk3.F90" "$TARGET/MY_SRC/stprk3.F90"
  touch "$TARGET/MY_SRC/"*.F90
  (cd "$NEMO_ROOT" && ./makenemo -n "$TARGET_NAME" -m conda-scalarmath -y --clean)
  (cd "$NEMO_ROOT" && ./makenemo -n "$TARGET_NAME" -m conda-scalarmath del_key key_xios)
  [[ -x "$TARGET/BLD/bin/nemo.exe" ]] || fail "$TARGET_NAME produced no executable"
  grep -Fq 'CALL v1_dump_hm_i( kstp )' "$TARGET/BLD/ppsrc/nemo/stprk3.f90" \
    && grep -Fq 'CALL l4_dump_ocean_surface_input( kstp, Nbb )' "$TARGET/BLD/ppsrc/nemo/stprk3.f90" \
    || fail 'compiled V1 record binary lacks a writer'
  grep -Fq 'ORCA2G1: the ssh bump' "$TARGET/BLD/ppsrc/nemo/restart.f90" \
    || fail 'V1 record binary lacks the certified bump'
  cmp -s "$TARGET/BLD/ppsrc/nemo/zdftke.f90" "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90" \
    || fail 'V1 record binary compiled a different zdftke'
}

stage_run() { # binary, directory, itend, stocklist
  local f
  [[ ! -e "$2" ]] || fail "run directory exists: $2"
  mkdir -p "$2"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$2/$f"; done
  cp "$ICE_REF" "$2/namelist_ice_ref"
  cp "$ICE_DECK" "$2/namelist_ice_cfg"
  render_deck "$2/namelist_cfg" "$3" "$4"
  cp "$1" "$2/nemo"
}

run_one() { # binary, directory, itend, stocklist, expected=done|boundary
  local dir=$2 rc=0
  stage_run "$1" "$dir" "$3" "$4"
  (cd "$dir"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$dir/run.exit"
  [[ -f "$dir/ocean.output" && -f "$dir/time.step" ]] || fail "$dir produced no NEMO log"
  # the run resolved V1's closure, not G5's, and not the IWM-forced floors
  grep -Eq 'nn_ice += +2$' "$dir/ocean.output" || fail "$dir did not resolve nn_ice=2"
  grep -Eq 'ln_zdftke = +T$' "$dir/ocean.output" && grep -Eq 'ln_zdfcst = +F$' "$dir/ocean.output" \
    && grep -Eq 'ln_zdfiwm = +F$' "$dir/ocean.output" || fail "$dir did not resolve V1's switches"
  grep -Eq 'nn_mxlice = +2$' "$dir/ocean.output" && grep -Eq 'nn_eice = +1$' "$dir/ocean.output" \
    || fail "$dir reset nn_mxlice/nn_eice"
  grep -Fq 'minimum mixing length with your parameters rmxl_min' "$dir/ocean.output" \
    && ! grep -Fq 'Internal wave-driven mixing case' "$dir/ocean.output" \
    || fail "$dir took the IWM-forced TKE floors"
  if [[ "$5" == done ]]; then
    [[ "$rc" -eq 0 && "$(tr -d '[:space:]' <"$dir/time.step")" == "$3" ]] \
      || fail "$dir did not complete $3 steps"
  fi
}

completed_steps() {
  local t rc
  t=$(tr -d '[:space:]' <"$1/time.step"); rc=$(<"$1/run.exit")
  if [[ "$rc" -eq 0 && "$t" == 100 ]]; then printf 100; return; fi
  grep -q 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$1/ocean.output" \
    && [[ -f "$1/output.abort.nc" ]] || fail "reference stopped outside stp_ctl at kt=$t"
  printf '%s' $((t - 1))
}

admit() {
  local steps=$1 kt name dir step
  for dir in "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
    [[ "$(<"$dir/run.exit")" == 0 ]] || fail "$dir is not a completed twin"
  done
  for kt in $(seq 1 "$steps"); do
    for name in $(printf 'oracle_ocean_surface_input_kt%08d.bin oracle_v1_hm_i_kt%08d.bin' "$kt" "$kt"); do
      for dir in "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
        [[ -s "$dir/$name" ]] || fail "$dir lacks $name"
      done
      cmp -s "$TWIN_A/$name" "$TWIN_B/$name" && cmp -s "$TWIN_A/$name" "$TWIN_C/$name" \
        || fail "twins differ in $name"
    done
  done
  for step in $(seq 2 "$steps"); do
    name=$(printf 'ORCA2G1_%08d_restart.nc' "$step")
    [[ -f "$TWIN_A/$name" || -f "$TWIN_B/$name" || -f "$TWIN_C/$name" ]] \
      || fail "no twin wrote restart step $step"
  done
  [[ -f "$REF/ORCA2G1_$(printf '%08d' "$steps")_restart.nc" ]] \
    || fail "writer-free reference lacks restart step $steps"
  "$G5_TOOL" --check-surface-payload "$TWIN_A" "$steps" >/dev/null
  check_hmi_frames "$TWIN_A" "$steps"
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$DECK" "$ICE_DECK" "$REF_BIN" "$TARGET/BLD/bin/nemo.exe" \
    "$TARGET/MY_SRC/stprk3.F90" >"$EVIDENCE/manifest/producers.sha256"
  printf 'V1_ACQUISITION_PASS steps=1..%s restarts=2..%s evidence=%s\n' "$steps" "$steps" "$EVIDENCE"
}

preflight
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD_BUILD record=%s; reuse writer-free %s; then smoke, boundary, twins A/B/C\n' \
    "$TARGET" "$REF_BIN"
  exit 0
fi
for dir in "$SMOKE" "$REF" "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
  [[ ! -e "$dir" ]] || fail "live run directory exists: $dir"
done
mkdir -p "$EVIDENCE"
build_target

# Smoke is the first NEMO execution; only after it passes is the boundary run allowed.
run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE" 2 "$(pad10 2)" done
check_hmi_frames "$SMOKE" 2
run_one "$REF_BIN" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps "$REF")
steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
((steps >= 2)) || fail "NEMO completed only $completed steps"
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_C" "$steps" "$(list_c "$steps")" done
admit "$steps"
