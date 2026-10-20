#!/usr/bin/env bash
# Vertical-mixing lane V5 acquisition (operator executed only).
# V5 = certified V4 + ORCA2's horizontally varying background diffusivity,
# verbatim ORCA2_OMIP_L4/EXP00/namelist_cfg:401 (nn_havtb = 1); deck = V4's
# with that one line.  nn_havtb is a run-time switch read by zdf_phy_init
# (zdfphy.f90:217-224); zdfphy.f90/zdf_oce.f90 and zdfphy.o's .text/.rodata in
# both reused binaries are identical to ORCA2_OMIP_L4's, so NOTHING is built:
#  - writer-free reference: G5's pinned writer-free binary (V1's reference);
#  - record: V1's pinned record binary (surface-input + hm_i writers).
# Inputs = V4's, nothing new (the shape reads gphit off the domain file).
# Records = V4's: restarts at every writable step 2..N over twins A/B/C, the
# surface-input and hm_i frames each step, the kt-1 post-zdf_phy frame (avm,
# avt, avs, en; l2_dump_zdf).  zdf_iwm_init resets avtb_2d = 1 (zdfiwm.f90:443),
# so the shape reaches only the avt_k seed (zdfphy.f90:227) that tke_tke reads
# at kt 1 (zdftke.f90:424): the kt-1 frame must differ from V4's admitted frame
# in en, and only in columns with -15 <= gphit < 15.
# Default is read-only preflight; --run smokes, finds NEMO's boundary with the
# writer-free binary, then acquires three twins.
set -Eeuo pipefail

fail() { printf 'REFUSE: V5 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly G5_TOOL=$here/../nemo_testcase_g5_orca2ice/run.sh
readonly V1_TOOL=$here/../nemo_testcase_v1_orca2tke/run.sh
readonly V3_TOOL=$here/../nemo_testcase_v3_orca2ddm/run.sh
readonly V4_TOOL=$here/../nemo_testcase_v4_orca2iwm/run.sh
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_BIN=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe
readonly REC_BIN=$NEMO_ROOT/cfgs/ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/bin/nemo.exe
readonly REF_BIN_SHA=e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7
readonly REC_BIN_SHA=b2ea4a47d6d6aae827809084aab2faf5546f79185c6fff89561db5f33e56c07b
readonly V5_DECK_SHA=3358b329bfee1330efad05f05c05fe314a3837e6e9195b2d53a2df88ab188102
readonly ICE_DECK_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref
readonly FSBC=2
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly IWM_FILE=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/zdfiwm_forcing_orca2.nc
readonly IWM_FILE_SHA=f4279899be77119fda4f9e20b3d5e3a70e4e478fe5ead0dc661656a1bba296c2
readonly V4_FRAME=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round14/oracle_v4_r14/twin_a/oracle_zdf_entry_kt00000001.bin
readonly V4_FRAME_SHA=18a8a8362f2534fc621855ae1014a2ed04843ebd7873d9ff16e7bd5f6952da94
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round15/oracle_v5_r15}
readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly TWIN_A=$EVIDENCE/twin_a TWIN_B=$EVIDENCE/twin_b TWIN_C=$EVIDENCE/twin_c
readonly DECK=$here/namelist_cfg ICE_DECK=$here/namelist_ice_cfg
readonly INPUT_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc
  ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc q_10.15JUNE2009_fill.nc
  ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc
  sss_data.nc runoff_core_monthly.nc namelist_ref)
input_sha() { sed -nE "s/^  \[$1\]=([0-9a-f]{64})$/\1/p" "$G5_TOOL"; }
sha() { sha256sum "$1" | awk '{print $1}'; }

check_havtb_frame() { # dir: P2, the kt-1 frame differs from V4's in en, only in the band
  /home/dbalwada/legoESM/.venv/bin/python -I - "$1/oracle_zdf_entry_kt00000001.bin" "$V4_FRAME" "$INPUT_ROOT/domain_cfg.nc" <<'PY' || fail "$1 kt-1 frame does not show nn_havtb as NEMO's code places it"
import struct, sys
import numpy as np
import netCDF4
def read(path):
    with open(path, "rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        version, kt, _, nx, ny, nz, bits = struct.unpack("=7i", fh.read(28))
        data = np.fromfile(fh, dtype=np.float64)
    n3, ni = nx * ny * nz, (nx - 4) * (ny - 4) * nz
    assert magic == "NEMO_L4_ZDF___2" and (version, kt, bits) == (2, 1, 64)
    assert data.size == n3 + 3 * ni and np.isfinite(data).all()
    out = {"avm": data[:n3].reshape((nx, ny, nz), order="F")[2:-2, 2:-2]}
    for i, name in enumerate(("avt", "avs", "en")):
        out[name] = data[n3 + i * ni:n3 + (i + 1) * ni].reshape((nx - 4, ny - 4, nz), order="F")
    return {k: np.swapaxes(v, 0, 1) for k, v in out.items()}
v5, v4 = read(sys.argv[1]), read(sys.argv[2])
with netCDF4.Dataset(sys.argv[3]) as ds:
    gphit = np.squeeze(np.asarray(ds.variables["gphit"][:], dtype=np.float64))
assert gphit.shape == v5["en"].shape[:2]
band = ((-15.0 <= gphit) & (gphit < 15.0))[..., None]
ok = True
for name in ("avm", "avt", "avs", "en"):
    ne = v5[name] != v4[name]
    print(f"V5_HAVTB_FRAME {name} unequal_vs_v4={int(ne.sum())} outside_band={int((ne & ~band).sum())}")
    ok &= not (ne & ~band).any()
ok &= bool((v5["en"] != v4["en"]).any())
sys.exit(0 if ok else 1)
PY
}

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run) ;; --check-havtb-frame) check_havtb_frame "$2"; exit 0 ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in worktree'
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] \
  || fail 'commit the exact acquisition tool before running it'
readonly COMMIT=$(git -C "$repo" rev-parse HEAD)

preflight() {
  local f want b
  for f in "$G5_TOOL" "$V1_TOOL" "$V3_TOOL" "$V4_TOOL" "$DECK" "$ICE_DECK" "$REF_BIN" "$REC_BIN" "$ICE_REF" "$IWM_FILE" "$V4_FRAME"; do
    [[ -f "$f" ]] || fail "missing $f"
  done
  for f in "${INPUT_FILES[@]}"; do
    want=$(input_sha "$f"); [[ ${#want} == 64 ]] || fail "G5 tool pins no SHA for $f"
    [[ -f "$INPUT_ROOT/$f" && "$(sha "$INPUT_ROOT/$f")" == "$want" ]] \
      || fail "missing or changed input $f"
  done
  [[ "$(sha "$DECK")" == "$V5_DECK_SHA" && "$(sha "$ICE_DECK")" == "$ICE_DECK_SHA" ]] \
    || fail 'committed V5 deck changed'
  [[ "$(sha "$IWM_FILE")" == "$IWM_FILE_SHA" ]] || fail 'ORCA2 internal-wave input changed'
  [[ "$(sha "$V4_FRAME")" == "$V4_FRAME_SHA" ]] || fail "V4's admitted kt-1 frame changed"
  [[ "$(sha "$REF_BIN")" == "$REF_BIN_SHA" ]] || fail 'G5 writer-free binary changed'
  [[ "$(sha "$REC_BIN")" == "$REC_BIN_SHA" ]] || fail 'V1 record binary changed'
  # zdf_phy_init's nn_havtb (and V4's IWM/DDM/EVD/TKE) compiled into both binaries exactly as ORCA2_OMIP_L4.
  local m sec
  for b in ORCA2G5_OMIP_L4_R31REF_FIX1 ORCA2V1_OMIP_L4_R1FULLENTRY; do
    for m in zdfphy.o zdfiwm.o; do
      for sec in .text .rodata; do
        cmp -s <(objcopy -O binary --only-section=$sec "$NEMO_ROOT/cfgs/$b/BLD/obj/$m" /dev/stdout) \
               <(objcopy -O binary --only-section=$sec "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/obj/$m" /dev/stdout) \
          || fail "$b $m $sec differs from ORCA2_OMIP_L4"
      done
    done
    for f in zdfiwm.f90 zdfddm.f90 zdfevd.f90 zdfphy.f90 zdftke.f90 trazdf.f90 zdf_oce.f90; do
      cmp -s "$NEMO_ROOT/cfgs/$b/BLD/ppsrc/nemo/$f" "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/$f" \
        || fail "$b $f differs from ORCA2_OMIP_L4"
    done
    cmp -s "$NEMO_ROOT/cfgs/$b/BLD/obj/zdfddm.o" "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/obj/zdfddm.o" \
      || fail "$b zdfddm.o differs from ORCA2_OMIP_L4"
  done
  printf 'V5_PREFLIGHT_OK commit=%s reference=G5_writer_free record=V1_record_binary (nothing built)\n' "$COMMIT"
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
  local f
  [[ ! -e "$2" ]] || fail "run directory exists: $2"
  mkdir -p "$2"
  for f in "${INPUT_FILES[@]}"; do cp "$INPUT_ROOT/$f" "$2/$f"; done
  cp "$IWM_FILE" "$2/zdfiwm_forcing_orca2.nc"
  cp "$ICE_REF" "$2/namelist_ice_ref"
  cp "$ICE_DECK" "$2/namelist_ice_cfg"
  render_deck "$2/namelist_cfg" "$3" "$4"
  cp "$1" "$2/nemo"
}

check_v5_log() { # V5 resolved V4's switches plus ORCA2's nn_havtb
  local o=$1/ocean.output
  grep -Eq 'horizontal variation for avtb +nn_havtb += +1$' "$o" || fail "$1 did not resolve nn_havtb=1"
  grep -Eq 'nn_ice += +2$' "$o" || fail "$1 did not resolve nn_ice=2"
  grep -Eq 'ln_zdftke = +T$' "$o" && grep -Eq 'ln_zdfcst = +F$' "$o" \
    || fail "$1 did not resolve V1's switches"
  grep -Eq 'ln_zdfiwm = +T$' "$o" \
    && grep -Fq 'zdf_iwm_init : internal wave-driven mixing' "$o" \
    && grep -Eq 'Variable \(T\) or constant \(F\) mixing efficiency += +F$' "$o" \
    && grep -Eq 'Differential internal wave-driven mixing \(T\) or not \(F\) += +T$' "$o" \
    && grep -Fq 'Force the background value applied to avm & avt in TKE' "$o" \
    && [[ $(grep -Ec 'Dissipation (above abyssal hills|along topographic slopes|scaling with N\^2|due to shoaling): .*TW' "$o") == 4 ]] \
    || fail "$1 did not resolve and read ORCA2's IWM"
  grep -Fq 'Internal wave-driven mixing case:   force   rn_emin = 1.e-10 and rmxl_min = 1.e-3' "$o" \
    && ! grep -Fq 'minimum mixing length with your parameters' "$o" \
    || fail "$1 did not take zdf_tke_init's ln_zdfiwm arm"
  grep -Eq 'ln_zdfddm = +T$' "$o" \
    && grep -Eq 'rn_avts = +1\.0000000000000000E-004 *$' "$o" \
    && grep -Eq 'rn_hsbfr= +1\.6000000000000001 *$' "$o" \
    && grep -Fq 'use double diffusive mixing: avs /= avt' "$o" \
    || fail "$1 did not resolve ORCA2's DDM"
  grep -Eq 'ln_zdfevd = +T$' "$o" && grep -Eq 'nn_evdm += +0$' "$o" \
    && grep -Fq 'convection: use enhanced vertical diffusion scheme' "$o" \
    && grep -Fq 'zdf_evd : Enhanced Vertical Diffusion (evd)' "$o" \
    || fail "$1 did not resolve and run ORCA2's EVD"
  grep -Eq 'nn_mxlice = +2$' "$o" && grep -Eq 'nn_eice = +1$' "$o" || fail "$1 changed V1's TKE"
}

run_one() { # binary, directory, itend, stocklist, expected=done|boundary
  local rc=0
  stage_run "$1" "$2" "$3" "$4"
  (cd "$2"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$2/run.exit"
  [[ -f "$2/ocean.output" && -f "$2/time.step" ]] || fail "$2 produced no NEMO log"
  check_v5_log "$2"
  if [[ "$5" == done ]]; then
    [[ "$rc" -eq 0 && "$(tr -d '[:space:]' <"$2/time.step")" == "$3" ]] \
      || fail "$2 did not complete $3 steps"
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
  for kt in $(seq 1 "$steps"); do
    for name in $(printf 'oracle_ocean_surface_input_kt%08d.bin oracle_v1_hm_i_kt%08d.bin' "$kt" "$kt"); do
      for dir in "$TWIN_A" "$TWIN_B" "$TWIN_C"; do [[ -s "$dir/$name" ]] || fail "$dir lacks $name"; done
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
  "$V1_TOOL" --check-hmi-frames "$TWIN_A" "$steps" >/dev/null
  check_havtb_frame "$TWIN_A"
  "$V4_TOOL" --check-iwm-frame "$TWIN_A" >/dev/null
  "$V3_TOOL" --check-ddm-frame "$TWIN_A" >/dev/null
  cmp -s "$TWIN_A/oracle_zdf_entry_kt00000001.bin" "$TWIN_B/oracle_zdf_entry_kt00000001.bin" \
    && cmp -s "$TWIN_A/oracle_zdf_entry_kt00000001.bin" "$TWIN_C/oracle_zdf_entry_kt00000001.bin" \
    || fail 'twins differ in the kt-1 zdf frame'
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$DECK" "$ICE_DECK" "$REF_BIN" "$REC_BIN" "$IWM_FILE" >"$EVIDENCE/manifest/producers.sha256"
  printf 'V5_ACQUISITION_PASS steps=1..%s restarts=2..%s evidence=%s\n' "$steps" "$steps" "$EVIDENCE"
}

preflight
[[ "$mode" == --run ]] || { printf 'WOULD_RUN smoke, boundary, twins A/B/C into %s\n' "$EVIDENCE"; exit 0; }
for dir in "$SMOKE" "$REF" "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
  [[ ! -e "$dir" ]] || fail "live run directory exists: $dir"
done
mkdir -p "$EVIDENCE"
# Smoke is the first NEMO execution; only after it passes is the boundary run allowed.
run_one "$REC_BIN" "$SMOKE" 2 "$(pad10 2)" done
"$V1_TOOL" --check-hmi-frames "$SMOKE" 2 >/dev/null
check_havtb_frame "$SMOKE"
run_one "$REF_BIN" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps "$REF")
steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
((steps >= 2)) || fail "NEMO completed only $completed steps"
run_one "$REC_BIN" "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$REC_BIN" "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$REC_BIN" "$TWIN_C" "$steps" "$(list_c "$steps")" done
admit "$steps"
