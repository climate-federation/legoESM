#!/usr/bin/env bash
# Vertical-mixing lane V2 acquisition (operator executed only).
# V2 = certified V1 + ORCA2's enhanced vertical diffusion, verbatim
# ORCA2_OMIP_L4/EXP00/namelist_cfg:390-392 (ln_zdfevd=T, nn_evdm=0, rn_evd=100).
# EVD is a run-time switch: zdfevd/zdfphy/zdftke compiled into both reused
# binaries are byte-identical to ORCA2_OMIP_L4's, so NOTHING is built:
#  - writer-free reference: G5's pinned writer-free binary (V1's reference);
#  - record: V1's pinned record binary (surface-input + hm_i writers).
# Records = V1's: restarts at every writable step 2..N over twins A/B/C,
# oracle_ocean_surface_input_kt*.bin and oracle_v1_hm_i_kt*.bin each step.
# Default is read-only preflight; --run smokes, finds NEMO's boundary with the
# writer-free binary, then acquires three twins.
set -Eeuo pipefail

fail() { printf 'REFUSE: V2 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly G5_TOOL=$here/../nemo_testcase_g5_orca2ice/run.sh
readonly V1_TOOL=$here/../nemo_testcase_v1_orca2tke/run.sh
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_BIN=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe
readonly REC_BIN=$NEMO_ROOT/cfgs/ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/bin/nemo.exe
readonly REF_BIN_SHA=e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7
readonly REC_BIN_SHA=b2ea4a47d6d6aae827809084aab2faf5546f79185c6fff89561db5f33e56c07b
readonly V2_DECK_SHA=2f4e18441c59170cd901473ef3255a012818491baf167383a88dd9ddfb3e6ff3
readonly ICE_DECK_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref
readonly FSBC=2
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round12/oracle_v2_r12}
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

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run) ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in worktree'
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] \
  || fail 'commit the exact acquisition tool before running it'
readonly COMMIT=$(git -C "$repo" rev-parse HEAD)

preflight() {
  local f want b
  for f in "$G5_TOOL" "$V1_TOOL" "$DECK" "$ICE_DECK" "$REF_BIN" "$REC_BIN" "$ICE_REF"; do
    [[ -f "$f" ]] || fail "missing $f"
  done
  for f in "${INPUT_FILES[@]}"; do
    want=$(input_sha "$f"); [[ ${#want} == 64 ]] || fail "G5 tool pins no SHA for $f"
    [[ -f "$INPUT_ROOT/$f" && "$(sha "$INPUT_ROOT/$f")" == "$want" ]] \
      || fail "missing or changed input $f"
  done
  [[ "$(sha "$DECK")" == "$V2_DECK_SHA" && "$(sha "$ICE_DECK")" == "$ICE_DECK_SHA" ]] \
    || fail 'committed V2 deck changed'
  [[ "$(sha "$REF_BIN")" == "$REF_BIN_SHA" ]] || fail 'G5 writer-free binary changed'
  [[ "$(sha "$REC_BIN")" == "$REC_BIN_SHA" ]] || fail 'V1 record binary changed'
  # EVD (and the TKE it follows) compiled into both binaries exactly as ORCA2_OMIP_L4.
  for b in ORCA2G5_OMIP_L4_R31REF_FIX1 ORCA2V1_OMIP_L4_R1FULLENTRY; do
    for f in zdfevd.f90 zdfphy.f90 zdftke.f90; do
      cmp -s "$NEMO_ROOT/cfgs/$b/BLD/ppsrc/nemo/$f" "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/$f" \
        || fail "$b $f differs from ORCA2_OMIP_L4"
    done
  done
  printf 'V2_PREFLIGHT_OK commit=%s reference=G5_writer_free record=V1_record_binary (nothing built)\n' "$COMMIT"
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
  cp "$ICE_REF" "$2/namelist_ice_ref"
  cp "$ICE_DECK" "$2/namelist_ice_cfg"
  render_deck "$2/namelist_cfg" "$3" "$4"
  cp "$1" "$2/nemo"
}

check_v2_log() { # V2 resolved V1's closure plus ORCA2's EVD, and ran zdf_evd
  local o=$1/ocean.output
  grep -Eq 'nn_ice += +2$' "$o" || fail "$1 did not resolve nn_ice=2"
  grep -Eq 'ln_zdftke = +T$' "$o" && grep -Eq 'ln_zdfcst = +F$' "$o" \
    && grep -Eq 'ln_zdfiwm = +F$' "$o" && grep -Eq 'ln_zdfddm = +F$' "$o" \
    || fail "$1 did not resolve V1's switches"
  grep -Eq 'ln_zdfevd = +T$' "$o" && grep -Eq 'nn_evdm += +0$' "$o" \
    && grep -Fq 'convection: use enhanced vertical diffusion scheme' "$o" \
    && grep -Fq 'zdf_evd : Enhanced Vertical Diffusion (evd)' "$o" \
    || fail "$1 did not resolve and run ORCA2's EVD"
  grep -Eq 'nn_mxlice = +2$' "$o" && grep -Eq 'nn_eice = +1$' "$o" \
    && ! grep -Fq 'Internal wave-driven mixing case' "$o" || fail "$1 changed V1's TKE"
}

run_one() { # binary, directory, itend, stocklist, expected=done|boundary
  local rc=0
  stage_run "$1" "$2" "$3" "$4"
  (cd "$2"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$2/run.exit"
  [[ -f "$2/ocean.output" && -f "$2/time.step" ]] || fail "$2 produced no NEMO log"
  check_v2_log "$2"
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
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$DECK" "$ICE_DECK" "$REF_BIN" "$REC_BIN" >"$EVIDENCE/manifest/producers.sha256"
  printf 'V2_ACQUISITION_PASS steps=1..%s restarts=2..%s evidence=%s\n' "$steps" "$steps" "$EVIDENCE"
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
run_one "$REF_BIN" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps "$REF")
steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
((steps >= 2)) || fail "NEMO completed only $completed steps"
run_one "$REC_BIN" "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$REC_BIN" "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$REC_BIN" "$TWIN_C" "$steps" "$(list_c "$steps")" done
admit "$steps"
