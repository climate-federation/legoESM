#!/usr/bin/env bash
# Grid ladder G5 round-1 acquisition (operator executed only).
# Default is read-only preflight.  --run builds an isolated MY_SRC config,
# smokes it, discovers NEMO's completed boundary, then acquires three twins.
set -Eeuo pipefail

fail() { printf 'REFUSE: G5 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_NAME=ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY
readonly TARGET_NAME=ORCA2G5_OMIP_L4_R31FULLENTRY
readonly SOURCE=$NEMO_ROOT/cfgs/$SOURCE_NAME TARGET=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly R5_BIN_SHA=0d4360984a193ff9bce2cfbaa13a45dbd26fcd18b215e0159c95254075830630
readonly R5_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly R5_MYSRC_SET_SHA=8c94f3b8cf01de142e726a9c2c9a43be9c11a74b8f2b6ad36371cdafe91b9766
readonly BUMP_RESTART_SHA=c49f94bd68f39c0254fa082e5b59feb3a918b6d2d173c19535380f3a6731327b
readonly BUMP_ISTATE_SHA=af1459b31f74e88c235ba2ebbf0c17bd3f02c0e6ce16e039e254f32a25b34782
readonly ICE_REF_SHA=055ab5e6839c12dbae61bf1c3bcdda46a8f95cf275b7f294d68d7f8e73179307
readonly G5_DECK_SHA=c57eeb83d8f59c5b1d44df1295a10f5f30d28537358c8d327fcecc3d538f569e
readonly G5_ICE_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly WANT_KEYS='key_RK3 key_qco key_si3 key_vco_1d3d'
readonly FSBC=2
readonly ROUNDS=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds
readonly INPUT_ROOT=$ROUNDS/round21/oracle_g4_r21/p6_a
readonly EVIDENCE=${EVIDENCE:-$ROUNDS/round31/oracle_g5_r31}
readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly TWIN_A=$EVIDENCE/twin_a TWIN_B=$EVIDENCE/twin_b TWIN_C=$EVIDENCE/twin_c
readonly INPUT_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc
  ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc q_10.15JUNE2009_fill.nc
  ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc
  sss_data.nc runoff_core_monthly.nc namelist_ref)
readonly -A INPUT_SHA=(
  [domain_cfg.nc]=7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839
  [data_1m_potential_temperature_nomask.nc]=ca00905c27078305e80130ea6dd07fc575de2ae3257fad38003b329458cf7f7a
  [data_1m_salinity_nomask.nc]=ad648d972f0631bde7e1b598d98470a979b7f2649271fc9cf5ccfca158e31d0c
  [u_10.15JUNE2009_fill.nc]=219bf3a6891dfe25871ba9d739e8ad465e12934b62f6524422a985b3858ae96b
  [v_10.15JUNE2009_fill.nc]=b3913b212b6a194fa8f79214ea56819728cbef94a06c6a82e94d65337d6e7b3a
  [ncar_rad.15JUNE2009_fill.nc]=050ee263198d49ce285070bc4d3e75953091c61c8f6a0d992484e79974fc37a7
  [t_10.15JUNE2009_fill.nc]=bee5df15a69d757f3911660910efc54c936555ec8bce742215ec047a16a4338b
  [q_10.15JUNE2009_fill.nc]=4c46f4527e8e5c1f3cc12942a87d0938c10431fcdb56cff7d25a9b5db2e69806
  [ncar_precip.15JUNE2009_fill.nc]=ed17b5069ddb13cd2a596079a0102e7792899e5dbe4ce1ba5a20e29802b5337a
  [slp.15JUNE2009_fill.nc]=19dea3069caa8f513218c7cbc11b3dce83d7cb9363e44b787e892917625c81fb
  [weights_core2_orca2_bicub.nc]=2cfefd58b0231f355cf999f943a48586f63e4e9264f0cbcc7197334f12b3e7e8
  [weights_core2_orca2_bilin.nc]=e5fe31582615262d84b06d01aeaa1893f4d942dc978aebc3e13f56cda484d766
  [chlorophyll.nc]=9178f44245f9465c584168eeb47b28e83c358a9ef78ed614cd623c4456ad7890
  [sss_data.nc]=ea4609d6deca4ccf95b6307a7b7eed1bff3b767146eb69c2a82762824bf1ed49
  [runoff_core_monthly.nc]=b92ff46e4c406096a1e1fe3815dfdcce12b234e84541a3b6dffb1037e777d4eb
  [namelist_ref]=b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd
)

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in worktree'
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] \
  || fail 'commit the exact acquisition tool before running it'
readonly COMMIT=$(git -C "$repo" rev-parse HEAD)
readonly DECK=$here/namelist_cfg ICE_DECK=$here/namelist_ice_cfg
readonly R5_BIN=$SOURCE/BLD/bin/nemo.exe R5_CPP=$SOURCE/cpp_$SOURCE_NAME.fcm
readonly G1_SOURCE=$NEMO_ROOT/cfgs/ORCA2G1_OMIP_L1_P6/MY_SRC
readonly ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref

sha() { sha256sum "$1" | awk '{print $1}'; }
sorted_keys() { sed -E 's/.*fppkeys *//' "$1" | xargs -n1 | LC_ALL=C sort | xargs; }
source_set_sha() {
  find "$SOURCE/MY_SRC" -maxdepth 1 -type f ! -name '*.orig' -printf '%f\n' | LC_ALL=C sort |
    while read -r f; do sha256sum "$SOURCE/MY_SRC/$f"; done | sha256sum | awk '{print $1}'
}

preflight() {
  local f
  for f in "$DECK" "$ICE_DECK" "$R5_BIN" "$R5_CPP" "$ICE_REF" \
           "$G1_SOURCE/restart.F90" "$G1_SOURCE/usrdef_istate.F90"; do
    [[ -f "$f" ]] || fail "missing $f"
  done
  for f in "${INPUT_FILES[@]}"; do
    [[ -f "$INPUT_ROOT/$f" && "$(sha "$INPUT_ROOT/$f")" == "${INPUT_SHA[$f]}" ]] \
      || fail "missing or changed input $f"
  done
  [[ "$(sha "$DECK")" == "$G5_DECK_SHA" && "$(sha "$ICE_DECK")" == "$G5_ICE_SHA" ]] \
    || fail 'committed G5 deck changed'
  [[ "$(sha "$R5_BIN")" == "$R5_BIN_SHA" && "$(sha "$R5_CPP")" == "$R5_CPP_SHA" ]] \
    || fail 'reusable ORCA2 R5 build changed'
  [[ "$(source_set_sha)" == "$R5_MYSRC_SET_SHA" ]] || fail 'R5 MY_SRC writer set changed'
  [[ "$(sha "$G1_SOURCE/restart.F90")" == "$BUMP_RESTART_SHA" \
     && "$(sha "$G1_SOURCE/usrdef_istate.F90")" == "$BUMP_ISTATE_SHA" ]] \
    || fail 'certified G4 bump sources changed'
  [[ "$(sha "$ICE_REF")" == "$ICE_REF_SHA" ]] || fail 'SI3 namelist reference changed'
  [[ "$(sorted_keys "$R5_CPP")" == "$WANT_KEYS" ]] || fail 'R5 lacks the exact key_si3 key set'
  grep -Fq 'CALL l4_dump_ocean_surface_input( kstp, Nbb )' "$SOURCE/MY_SRC/stprk3.F90" \
    && grep -Fq "l4_canon_2d(rCdU_ice,'T')" "$SOURCE/MY_SRC/stprk3.F90" \
    && grep -Fq "l4_canon_2d(fwfice,'T')" "$SOURCE/MY_SRC/stprk3.F90" \
    && grep -Fq "l4_canon_2d(fr_i,'T')" "$SOURCE/MY_SRC/stprk3.F90" \
    || fail 'R5 per-step surface writer lacks a required ice/ocean field'
  printf 'G5_PREFLIGHT_OK source=%s commit=%s expected_boundary=kt11_after_10_completed\n' \
    "$SOURCE_NAME" "$COMMIT"
}

render_deck() { # target, itend, stocklist
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
         -e "s/^(   nn_stock    = ).*/\1    $2/" \
         -e "s/^(   nn_stocklist = ).*/\1$3/" "$DECK" >"$1"
}
pad10() { local a=("$@"); while ((${#a[@]} < 10)); do a+=(0); done; local IFS=,; printf '%s' "${a[*]}"; }
list_a() { pad10 "$1"; }
list_b() { pad10 $(seq 2 2 "$1"); }
list_c() { (($1 >= 3)) && pad10 $(seq 3 2 "$1") || pad10 0; }

build_target() {
  [[ ! -e "$TARGET" ]] || fail "target config already exists: $TARGET"
  (cd "$NEMO_ROOT" && ./makenemo -r ORCA2_ICE_PISCES -n "$TARGET_NAME" \
    -m conda-scalarmath del_key key_xios)
  local src
  while IFS= read -r -d '' src; do cp -a "$src" "$TARGET/MY_SRC/$(basename "$src")"; done \
    < <(find "$SOURCE/MY_SRC" -maxdepth 1 -type f ! -name '*.orig' -print0 | sort -z)
  cp "$G1_SOURCE/restart.F90" "$G1_SOURCE/usrdef_istate.F90" "$TARGET/MY_SRC/"
  cp "$R5_CPP" "$TARGET/cpp_$TARGET_NAME.fcm"
  touch "$TARGET/MY_SRC/"*.F90
  (cd "$NEMO_ROOT" && ./makenemo -n "$TARGET_NAME" -m conda-scalarmath del_key key_xios)
  [[ -x "$TARGET/BLD/bin/nemo.exe" ]] || fail 'G5 build produced no executable'
  grep -Fq 'ORCA2G1: the ssh bump' "$TARGET/BLD/ppsrc/nemo/restart.f90" \
    && grep -Fq 'CALL l4_dump_ocean_surface_input( kstp, Nbb )' "$TARGET/BLD/ppsrc/nemo/stprk3.f90" \
    || fail 'compiled G5 binary lacks bump or per-step writer'
}

stage_run() { # directory, itend, stocklist
  local dir=$1 f
  [[ ! -e "$dir" ]] || fail "run directory exists: $dir"
  mkdir -p "$dir"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$dir/$f"; done
  cp "$ICE_REF" "$dir/namelist_ice_ref"
  cp "$ICE_DECK" "$dir/namelist_ice_cfg"
  render_deck "$dir/namelist_cfg" "$2" "$3"
  cp "$TARGET/BLD/bin/nemo.exe" "$dir/nemo"
}

run_one() { # directory, itend, stocklist, expected=done|boundary
  local dir=$1 rc=0
  stage_run "$dir" "$2" "$3"
  (cd "$dir"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$dir/run.exit"
  [[ -f "$dir/ocean.output" && -f "$dir/time.step" ]] || fail "$dir produced no NEMO log"
  grep -Eq 'nn_ice += +2$' "$dir/ocean.output" || fail "$dir did not resolve nn_ice=2"
  if [[ "$4" == done ]]; then
    [[ "$rc" -eq 0 && "$(tr -d '[:space:]' <"$dir/time.step")" == "$2" ]] \
      || fail "$dir did not complete $2 steps"
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
    for kt in $(seq 1 "$steps"); do
      name=$(printf 'oracle_ocean_surface_input_kt%08d.bin' "$kt")
      [[ -s "$dir/$name" ]] || fail "$dir lacks $name"
    done
  done
  for kt in $(seq 1 "$steps"); do
    name=$(printf 'oracle_ocean_surface_input_kt%08d.bin' "$kt")
    cmp -s "$TWIN_A/$name" "$TWIN_B/$name" && cmp -s "$TWIN_A/$name" "$TWIN_C/$name" \
      || fail "surface twins differ at kt=$kt"
  done
  for step in $(seq 2 "$steps"); do
    name=$(printf 'ORCA2G1_%08d_restart.nc' "$step")
    [[ -f "$TWIN_A/$name" || -f "$TWIN_B/$name" || -f "$TWIN_C/$name" ]] \
      || fail "no twin wrote restart step $step"
  done
  python3 -I - "$TWIN_A" "$steps" <<'PY'
import pathlib, struct, sys
import numpy as np
root, steps = pathlib.Path(sys.argv[1]), int(sys.argv[2])
for kt in range(1, steps + 1):
    path = root / f"oracle_ocean_surface_input_kt{kt:08d}.bin"
    with path.open("rb") as stream:
        magic = stream.read(16).decode("ascii").rstrip()
        header = struct.unpack("=13i", stream.read(52))
        values = np.fromfile(stream, dtype=np.float64)
    if magic != "NEMO_L4_SBCIN_1" or header[0:3] != (1, kt, 1 if kt % 2 else 3):
        raise SystemExit(f"bad G5 surface header: {path.name} {magic!r} {header}")
    if not values.size or not np.isfinite(values).all():
        raise SystemExit(f"bad G5 surface payload: {path.name}")
PY
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$DECK" "$ICE_DECK" "$TARGET/BLD/bin/nemo.exe" \
    >"$EVIDENCE/manifest/producers.sha256"
  printf 'G5_ACQUISITION_PASS steps=1..%s restarts=2..%s expected_stop=kt11 evidence=%s\n' \
    "$steps" "$steps" "$EVIDENCE"
}

preflight
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD_BUILD %s; then smoke, boundary, twins A/B/C\n' "$TARGET"
  exit 0
fi
if [[ "$mode" == --admit-existing ]]; then
  [[ -x "$TARGET/BLD/bin/nemo.exe" && -d "$EVIDENCE" ]] || fail 'no existing target/evidence'
  steps=$(tr -d '[:space:]' <"$TWIN_A/time.step")
  admit "$steps"
  exit 0
fi
[[ ! -e "$EVIDENCE" ]] || fail "evidence exists: $EVIDENCE"
mkdir -p "$EVIDENCE"
build_target

# Smoke is the first NEMO execution.  Only after it passes is the long boundary run allowed.
run_one "$SMOKE" 2 "$(pad10 2)" done
run_one "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps "$REF")
steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
((steps >= 2)) || fail "NEMO completed only $completed steps"
run_one "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$TWIN_C" "$steps" "$(list_c "$steps")" done
admit "$steps"
