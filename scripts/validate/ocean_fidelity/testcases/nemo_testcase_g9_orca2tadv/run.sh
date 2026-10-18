#!/usr/bin/env bash
# Grid ladder G9 round-1 acquisition (Decision 128, lane A) -- OPERATOR-EXECUTED ONLY.
# Default is read-only preflight.  --run smokes, discovers NEMO's completed
# boundary with the writer-free binary, acquires three record twins, then
# admits them (passivity against the writer-free reference first).
#
# G9 = the certified G5 deck with ONE switch, ORCA2's &namtra_adv
# (ln_traadv_fct = T, nn_fct_h = 2, nn_fct_v = 2, so ln_traadv_OFF = F).
# NO BUILD: the operator is a namelist switch (traadv.f90:577 tra_adv_init)
# and traadv/traadv_cen/traadv_fct are compiled in every build, so G5's two
# round-31 binaries are reused, pinned by SHA-256 -- the same cpp keys
# (key_RK3 key_qco key_si3 key_vco_1d3d), no new key, no extra input file.
# The record set is G5's (per-step surface input + restarts), plus what the
# record binary already writes at kt 1: NEMO's stage-1 transport operands
# (vv, zvb, vmask, zFv; stprk3_stg.f90:288-314) and the stage 1/2/3 tracer
# RHS before and after tra_adv -- the advection trend on the T-pivot row.
# The agent that wrote this ran neither makenemo nor NEMO.
set -Eeuo pipefail

fail() { printf 'REFUSE: G9 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

resolve_ice_value() { # key, reference namelist, overlay namelist
  awk -F= -v want="$1" '
    { sub(/!.*/, ""); key=$1; gsub(/[[:space:]]/, "", key) }
    tolower(key) == tolower(want) {
      value=$2; gsub(/[[:space:],]/, "", value); found=value
    }
    END { if (found == "") exit 1; print tolower(found) }
  ' "$2" "$3"
}

check_ice_writer_contract() { # writer, reference namelist, overlay namelist
  local writer=$1 ref=$2 overlay=$3 sal lev topo pra path
  for path in "$writer" "$ref" "$overlay"; do [[ -f "$path" ]] || fail "missing $path"; done
  sal=$(resolve_ice_value nn_icesal "$ref" "$overlay") \
    || fail 'cannot resolve nn_icesal for ice writer'
  lev=$(resolve_ice_value ln_pnd_LEV "$ref" "$overlay") \
    || fail 'cannot resolve ln_pnd_LEV for ice writer'
  topo=$(resolve_ice_value ln_pnd_TOPO "$ref" "$overlay") \
    || fail 'cannot resolve ln_pnd_TOPO for ice writer'
  pra=$(resolve_ice_value ln_adv_Pra "$ref" "$overlay") \
    || fail 'cannot resolve ln_adv_Pra for ice writer'
  [[ "$pra" == .true. ]] || fail 'deck does not select the Prather writer'
  if grep -Fq 'l4_orca1ice_pra_dump: selector contract changed' "$writer"; then
    [[ "$sal" == 2 && "$lev" == .false. && "$topo" == .false. ]] \
      || fail "Prather writer requires nn_icesal=2 and ponds off; deck resolves nn_icesal=$sal ln_pnd_LEV=$lev ln_pnd_TOPO=$topo"
  fi
  if [[ "$sal" == 4 ]]; then
    grep -Fq "IF( nn_icesal == 4 ) WRITE(iunit) l4_canon_4d(sxsi,'T')" "$writer" \
      || fail 'Prather writer does not serialize the selected nn_icesal=4 moments'
  fi
  if [[ "$lev" == .true. || "$topo" == .true. ]]; then
    grep -Fq 'IF( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN' "$writer" \
      || fail 'Prather writer does not serialize the selected pond moments'
  fi
}

check_surface_payloads() { # directory, completed steps
  python3 -I - "$1" "$2" <<'PY'
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
    _, _, _, jpi, jpj, jpts, _, halo, nfull, nreduced, nhalo1, n3d, bits = header
    if (nfull, nreduced, nhalo1, n3d, bits) != (20, 12, 1, 2, 64):
        raise SystemExit(f"bad G5 surface schema: {path.name} {header}")
    expected_values = (
        nfull * jpi * jpj
        + nreduced * (jpi - 2 * halo) * (jpj - 2 * halo)
        + nhalo1 * (jpi - 2) * (jpj - 2)
        + n3d * (jpi - 2 * halo) * (jpj - 2 * halo) * jpts
    )
    expected_bytes = 16 + 13 * 4 + expected_values * (bits // 8)
    if path.stat().st_size != expected_bytes or values.size != expected_values:
        raise SystemExit(
            f"bad G5 surface payload size: {path.name} "
            f"bytes={path.stat().st_size}/{expected_bytes} values={values.size}/{expected_values}"
        )
    if not values.size or not np.isfinite(values).all():
        raise SystemExit(f"bad G5 surface payload: {path.name}")
PY
}

mode=${1:---preflight-only}
if [[ "$mode" == --check-writer-contract ]]; then
  (($# == 4)) || exit 64
  check_ice_writer_contract "$2" "$3" "$4"
  printf 'G5_WRITER_CONTRACT_OK %s\n' "$2"
  exit 0
fi
if [[ "$mode" == --check-surface-payload ]]; then
  (($# == 3)) || exit 64
  check_surface_payloads "$2" "$3"
  printf 'G5_SURFACE_PAYLOAD_OK %s steps=1..%s\n' "$2" "$3"
  exit 0
fi
case "$mode" in --preflight-only|--run|--admit-existing) ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_NAME=ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY
readonly P2_NAME=ORCA2_OMIP_L4_P2VBND_R2
readonly REF_TARGET_NAME=ORCA2G5_OMIP_L4_R31REF_FIX1
readonly TARGET_NAME=ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1
readonly REF_BIN_SHA=e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7
readonly REC_BIN_SHA=41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275
readonly SOURCE=$NEMO_ROOT/cfgs/$SOURCE_NAME P2=$NEMO_ROOT/cfgs/$P2_NAME
readonly REF_TARGET=$NEMO_ROOT/cfgs/$REF_TARGET_NAME TARGET=$NEMO_ROOT/cfgs/$TARGET_NAME
readonly R5_BIN_SHA=0d4360984a193ff9bce2cfbaa13a45dbd26fcd18b215e0159c95254075830630
readonly R5_CPP_SHA=2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67
readonly P2_BIN_SHA=5b470620cae10795786e6e548b4492840bc1ba89c676f23322c86ff15f38c1bd
readonly P2_PRA_SHA=d1be8a22cbb6dd2f7c695e31164ba3d09b4cd3f5bcbe5f71ed0cfec5749f3828
readonly HYBRID_MYSRC_SET_SHA=f1bfa3ee99cf841e9dab64204b0967e91715084daf74b43489935c8c3181e59c
readonly BUMP_RESTART_SHA=c49f94bd68f39c0254fa082e5b59feb3a918b6d2d173c19535380f3a6731327b
readonly BUMP_ISTATE_SHA=af1459b31f74e88c235ba2ebbf0c17bd3f02c0e6ce16e039e254f32a25b34782
readonly ICE_REF_SHA=055ab5e6839c12dbae61bf1c3bcdda46a8f95cf275b7f294d68d7f8e73179307
readonly G9_DECK_SHA=570db996c34e98ccdabf3c367455c1ee6d0b4f18df9f3bdd7264cf053f1f0720
readonly G5_ICE_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly WANT_KEYS='key_RK3 key_qco key_si3 key_vco_1d3d'
readonly FSBC=2
readonly ROUNDS=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds
readonly INPUT_ROOT=$ROUNDS/round21/oracle_g4_r21/p6_a
readonly EVIDENCE=${EVIDENCE:-$ROUNDS/round46/oracle_g9_r46}
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
readonly P2_BIN=$P2/BLD/bin/nemo.exe P2_PRA=$P2/MY_SRC/icedyn_adv_pra.F90
readonly G1_SOURCE=$NEMO_ROOT/cfgs/ORCA2G1_OMIP_L1_P6/MY_SRC
readonly ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref

sha() { sha256sum "$1" | awk '{print $1}'; }
sorted_keys() { sed -E 's/.*fppkeys *//' "$1" | xargs -n1 | LC_ALL=C sort | xargs; }
hybrid_source_set_sha() {
  {
    find "$SOURCE/MY_SRC" -maxdepth 1 -type f ! -name '*.orig' \
      ! -name 'icedyn_adv_pra.F90' -printf '%f\n' | LC_ALL=C sort |
      while read -r f; do printf '%s  %s\n' "$(sha "$SOURCE/MY_SRC/$f")" "$f"; done
    printf '%s  icedyn_adv_pra.F90\n' "$(sha "$P2_PRA")"
  } | LC_ALL=C sort -k2 | sha256sum | awk '{print $1}'
}

preflight() {
  local f
  for f in "$DECK" "$ICE_DECK" "$R5_BIN" "$R5_CPP" "$P2_BIN" "$P2_PRA" "$ICE_REF" \
           "$G1_SOURCE/restart.F90" "$G1_SOURCE/usrdef_istate.F90"; do
    [[ -f "$f" ]] || fail "missing $f"
  done
  for f in "${INPUT_FILES[@]}"; do
    [[ -f "$INPUT_ROOT/$f" && "$(sha "$INPUT_ROOT/$f")" == "${INPUT_SHA[$f]}" ]] \
      || fail "missing or changed input $f"
  done
  [[ "$(sha "$DECK")" == "$G9_DECK_SHA" && "$(sha "$ICE_DECK")" == "$G5_ICE_SHA" ]] \
    || fail 'committed G9 deck changed'
  grep -Fq "WRITE(l1_unit) l4_canon_3d(ts(:,:,:,jp_tem,Krhs),'T')" "$TARGET/BLD/ppsrc/nemo/stprk3_stg.f90" \
    && grep -Fq "l4_canon_3d(vmask,'V'), l4_canon_3d(zFv,'V')" "$TARGET/BLD/ppsrc/nemo/stprk3_stg.f90" \
    || fail 'G5 record binary lacks the stage tracer/transport operand writers'
  [[ "$(sha "$R5_BIN")" == "$R5_BIN_SHA" && "$(sha "$R5_CPP")" == "$R5_CPP_SHA" ]] \
    || fail 'reusable ORCA2 R5 build changed'
  [[ "$(sha "$P2_BIN")" == "$P2_BIN_SHA" && "$(sha "$P2_PRA")" == "$P2_PRA_SHA" ]] \
    || fail 'proven ORCA2 ice writer changed'
  [[ "$(hybrid_source_set_sha)" == "$HYBRID_MYSRC_SET_SHA" ]] \
    || fail 'hybrid G5 MY_SRC writer set changed'
  [[ "$(sha "$G1_SOURCE/restart.F90")" == "$BUMP_RESTART_SHA" \
     && "$(sha "$G1_SOURCE/usrdef_istate.F90")" == "$BUMP_ISTATE_SHA" ]] \
    || fail 'certified G4 bump sources changed'
  [[ "$(sha "$ICE_REF")" == "$ICE_REF_SHA" ]] || fail 'SI3 namelist reference changed'
  [[ "$(sorted_keys "$R5_CPP")" == "$WANT_KEYS" ]] || fail 'R5 lacks the exact key_si3 key set'
  check_ice_writer_contract "$P2_PRA" "$ICE_REF" "$ICE_DECK"
  grep -Fq 'CALL l4_dump_ocean_surface_input( kstp, Nbb )' "$SOURCE/MY_SRC/stprk3.F90" \
    && grep -Fq "l4_canon_2d(rCdU_ice,'T')" "$SOURCE/MY_SRC/stprk3.F90" \
    && grep -Fq "l4_canon_2d(fwfice,'T')" "$SOURCE/MY_SRC/stprk3.F90" \
    && grep -Fq "l4_canon_2d(fr_i,'T')" "$SOURCE/MY_SRC/stprk3.F90" \
    || fail 'R5 per-step surface writer lacks a required ice/ocean field'
  [[ -x "$REF_TARGET/BLD/bin/nemo.exe" && -x "$TARGET/BLD/bin/nemo.exe" ]] \
    || fail 'G5 round-31 binaries missing'
  [[ "$(sha "$REF_TARGET/BLD/bin/nemo.exe")" == "$REF_BIN_SHA" \
     && "$(sha "$TARGET/BLD/bin/nemo.exe")" == "$REC_BIN_SHA" ]] \
    || fail 'G5 round-31 binaries changed'
  [[ "$(sorted_keys "$REF_TARGET/cpp_$REF_TARGET_NAME.fcm")" == "$WANT_KEYS" \
     && "$(sorted_keys "$TARGET/cpp_$TARGET_NAME.fcm")" == "$WANT_KEYS" ]] \
    || fail 'G5 binaries do not carry the exact key set'
  grep -Fq 'CALL l4_dump_ocean_surface_input( kstp, Nbb )' "$TARGET/BLD/ppsrc/nemo/stprk3.f90" \
    || fail 'G5 record binary lacks the per-step writer'
  ! grep -Rqs 'l4_dump_ocean_surface_input' "$REF_TARGET/BLD/ppsrc/nemo" \
    || fail 'G5 reference binary carries the surface writer'
  check_ice_writer_contract "$TARGET/BLD/ppsrc/nemo/icedyn_adv_pra.f90" "$ICE_REF" "$ICE_DECK"
  printf 'G9_PREFLIGHT_OK source=%s commit=%s expected_boundary=kt11_after_10_completed\n' \
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

stage_run() { # binary, directory, itend, stocklist
  local binary=$1 dir=$2 f
  [[ ! -e "$dir" ]] || fail "run directory exists: $dir"
  mkdir -p "$dir"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$dir/$f"; done
  cp "$ICE_REF" "$dir/namelist_ice_ref"
  cp "$ICE_DECK" "$dir/namelist_ice_cfg"
  render_deck "$dir/namelist_cfg" "$3" "$4"
  cp "$binary" "$dir/nemo"
}

run_one() { # binary, directory, itend, stocklist, expected=done|boundary
  local dir=$2 rc=0
  stage_run "$1" "$dir" "$3" "$4"
  (cd "$dir"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$dir/run.exit"
  [[ -f "$dir/ocean.output" && -f "$dir/time.step" ]] || fail "$dir produced no NEMO log"
  grep -Eq 'nn_ice += +2$' "$dir/ocean.output" || fail "$dir did not resolve nn_ice=2"
  # NEMO's own log: ORCA2's FCT tracer advection, orders 2/2 (traadv.f90:
  # 609-621, :666-668), and G5's free slip, no lateral momentum diffusion
  # and momentum advection OFF unchanged.
  grep -Eq 'No advection on T & S +ln_traadv_OFF = +F$' "$dir/ocean.output" \
    && grep -Eq 'centered scheme +ln_traadv_cen = +F$' "$dir/ocean.output" \
    && grep -Eq 'Flux Corrected Transport scheme +ln_traadv_fct = +T$' "$dir/ocean.output" \
    && grep -Eq 'MUSCL scheme +ln_traadv_mus = +F$' "$dir/ocean.output" \
    && grep -Eq 'UBS scheme +ln_traadv_ubs = +F$' "$dir/ocean.output" \
    && grep -Eq 'QUICKEST scheme +ln_traadv_qck = +F$' "$dir/ocean.output" \
    && grep -Eq '==>>> +FCT +scheme is used\. Horizontal order +: +2 +Vertical +order +: +2' "$dir/ocean.output" \
    || fail "$dir did not resolve ORCA2's 2nd/2nd-order FCT tracer advection"
  grep -Eq 'laplacian operator +ln_dynldf_lap = +F$' "$dir/ocean.output" \
    || fail "$dir resolved lateral momentum diffusion; G9 keeps G5's ln_dynldf_OFF"
  grep -Eq 'free-slip +: Cd = 0 +ln_drg_OFF += +T$' "$dir/ocean.output" \
    || fail "$dir resolved bottom drag; G9 keeps G5's ln_drg_OFF"
  ! grep -q 'vector form : keg + zad + vor is used' "$dir/ocean.output" \
    || fail "$dir resolved momentum advection; G9 keeps G5's ln_dynadv_OFF"
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
  local steps=$1 kt name dir step completed expected_steps
  completed=$(completed_steps "$REF")
  expected_steps=$((completed < 10 ? completed : 10))
  expected_steps=$((expected_steps - expected_steps % FSBC))
  [[ "$steps" -eq "$expected_steps" ]] \
    || fail "record length $steps disagrees with writer-free reference boundary $completed"
  [[ -f "$REF/ORCA2G1_$(printf '%08d' "$steps")_restart.nc" ]] \
    || fail "writer-free reference lacks restart step $steps"
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
  check_surface_payloads "$TWIN_A" "$steps"
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$DECK" "$ICE_DECK" "$REF_TARGET/BLD/bin/nemo.exe" \
    "$TARGET/BLD/bin/nemo.exe" "$P2_PRA" \
    >"$EVIDENCE/manifest/producers.sha256"
  # Writers passive: the record twins' restarts equal the writer-free
  # reference's (every variable, byte for byte), then the G5 frame admission.
  (cd "$repo" && PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src \
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
    python "$here/../nemo_testcase_g5_orca2ice/g5_records.py" admit \
      --evidence "$EVIDENCE" --steps "$steps" --json "$EVIDENCE/manifest/admission.json") \
    || fail 'passivity/frame admission refused'
  printf 'G9_ACQUISITION_PASS steps=1..%s restarts=2..%s reference_completed=%s evidence=%s\n' \
    "$steps" "$steps" "$completed" "$EVIDENCE"
}

preflight
if [[ "$mode" == --preflight-only ]]; then
  printf 'WOULD_RUN (no build) writer-free=%s record=%s: smoke, boundary, twins A/B/C, admission\n' \
    "$REF_TARGET" "$TARGET"
  exit 0
fi
if [[ "$mode" == --admit-existing ]]; then
  [[ -d "$EVIDENCE" ]] || fail 'no existing evidence'
  steps=$(tr -d '[:space:]' <"$TWIN_A/time.step")
  admit "$steps"
  exit 0
fi
for dir in "$SMOKE" "$REF" "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
  [[ ! -e "$dir" ]] || fail "live run directory exists: $dir"
done
mkdir -p "$EVIDENCE"

# Smoke is the first NEMO execution.  Only after it passes is the long boundary run allowed.
run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE" 2 "$(pad10 2)" done
run_one "$REF_TARGET/BLD/bin/nemo.exe" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps "$REF")
steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
((steps >= 2)) || fail "NEMO completed only $completed steps"
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$TARGET/BLD/bin/nemo.exe" "$TWIN_C" "$steps" "$(list_c "$steps")" done
admit "$steps"
