#!/usr/bin/env bash
# Lane B rung L2 acquisition (operator executed only).
# L2 = certified G5 + ORCA2's standard isoneutral tracer diffusion.  The
# existing G5 binaries already contain ldf_slp, mode-20 ldf_c2d,
# traldf_iso_lap + MSC, and tra_zdf_imp's K33 path, so nothing is built.
set -Eeuo pipefail

fail() { printf 'REFUSE: L2 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly G5_TOOL=$here/../nemo_testcase_g5_orca2ice/run.sh
readonly G5_DECK=$here/../nemo_testcase_g5_orca2ice/namelist_cfg
readonly ICE_DECK=$here/../nemo_testcase_g5_orca2ice/namelist_ice_cfg
readonly RENDER=$here/l2_deck.py CHECKER=$here/l2_records.py
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_BIN=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe
readonly REC_BIN=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/bin/nemo.exe
readonly REF_BIN_SHA=e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7
readonly REC_BIN_SHA=41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275
readonly G5_DECK_SHA=c57eeb83d8f59c5b1d44df1295a10f5f30d28537358c8d327fcecc3d538f569e
readonly L2_DECK_SHA=0f683bf8838a6b979c1df7bff247cb3867b86061b0e9966df019233d8e62c14b
readonly ICE_DECK_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly G5_STAGE_SHA=7ac36ec2fe76bf9b07e270b48dcfa05ee41334792b8472914b1aa053b73a4f2c
readonly ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref
readonly FSBC=2
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly G5_RECORD=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round31/oracle_g5_r31/twin_a
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round17/oracle_l2_r17}
readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly TWIN_A=$EVIDENCE/twin_a TWIN_B=$EVIDENCE/twin_b TWIN_C=$EVIDENCE/twin_c
readonly INPUT_FILES=(domain_cfg.nc data_1m_potential_temperature_nomask.nc
  data_1m_salinity_nomask.nc u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc
  ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc q_10.15JUNE2009_fill.nc
  ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc
  sss_data.nc runoff_core_monthly.nc namelist_ref)
input_sha() { sed -nE "s/^  \[$1\]=([0-9a-f]{64})$/\1/p" "$G5_TOOL"; }
sha() { sha256sum "$1" | awk '{print $1}'; }

repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in worktree'
readonly PYPATH=$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src
check_acting() {
  PYTHONPATH=$PYPATH JAX_PLATFORMS=cpu /home/dbalwada/legoESM/.venv/bin/python \
    "$CHECKER" --check "$1" --g5 "$G5_RECORD" --domain "$INPUT_ROOT/domain_cfg.nc" \
    || fail "$1 does not show wet-only isoneutral tracer diffusion"
}

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run) ;; --check-acting) check_acting "$2"; exit 0 ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] \
  || fail 'commit the exact acquisition tool before running it'
readonly COMMIT=$(git -C "$repo" rev-parse HEAD)

preflight() {
  local f want b m rendered
  for f in "$G5_TOOL" "$G5_DECK" "$ICE_DECK" "$RENDER" "$CHECKER" \
           "$REF_BIN" "$REC_BIN" "$ICE_REF" "$G5_RECORD/oracle_rktracer_stage3_kt00000001.bin"; do
    [[ -f "$f" ]] || fail "missing $f"
  done
  for f in "${INPUT_FILES[@]}"; do
    want=$(input_sha "$f"); [[ ${#want} == 64 ]] || fail "G5 tool pins no SHA for $f"
    [[ -f "$INPUT_ROOT/$f" && "$(sha "$INPUT_ROOT/$f")" == "$want" ]] \
      || fail "missing or changed input $f"
  done
  [[ "$(sha "$G5_DECK")" == "$G5_DECK_SHA" ]] || fail "G5 deck changed"
  [[ "$(sha "$ICE_DECK")" == "$ICE_DECK_SHA" ]] || fail "ice deck changed"
  [[ "$(sha "$REF_BIN")" == "$REF_BIN_SHA" ]] || fail 'G5 writer-free binary changed'
  [[ "$(sha "$REC_BIN")" == "$REC_BIN_SHA" ]] || fail 'G5 record binary changed'
  [[ "$(sha "$G5_RECORD/oracle_rktracer_stage3_kt00000001.bin")" == "$G5_STAGE_SHA" ]] \
    || fail 'G5 admitted stage-3 record changed'
  rendered=$(mktemp "$EVIDENCE.rendered.XXXXXX")
  /home/dbalwada/legoESM/.venv/bin/python "$RENDER" "$G5_DECK" "$rendered"
  [[ "$(sha "$rendered")" == "$L2_DECK_SHA" ]] || fail 'rendered L2 deck changed'
  rm -f "$rendered"
  for b in ORCA2G5_OMIP_L4_R31REF_FIX1 ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1; do
    for m in ldfslp.o ldftra.o ldfc1d_c2d.o traldf.o traldf_iso.o trazdf.o; do
      cmp -s <(objcopy -O binary --only-section=.text "$NEMO_ROOT/cfgs/$b/BLD/obj/$m" /dev/stdout) \
             <(objcopy -O binary --only-section=.text "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/obj/$m" /dev/stdout) \
        || fail "$b $m .text differs from ORCA2_OMIP_L4"
    done
  done
  cmp -s <(objcopy -O binary --only-section=.text "$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/obj/stprk3_stg.o" /dev/stdout) \
         <(objcopy -O binary --only-section=.text "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/obj/stprk3_stg.o" /dev/stdout) \
    || fail 'record stprk3_stg.o .text differs from ORCA2_OMIP_L4'
  printf 'L2_PREFLIGHT_OK commit=%s reference=G5_writer_free record=G5_record_binary (nothing built)\n' "$COMMIT"
}

render_deck() { # target, itend, stocklist
  /home/dbalwada/legoESM/.venv/bin/python "$RENDER" "$G5_DECK" "$1.base"
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
         -e "s/^(   nn_stock    = ).*/\1    $2/" \
         -e "s/^(   nn_stocklist = ).*/\1$3/" "$1.base" >"$1"
  rm -f "$1.base"
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
  cp "$ICE_REF" "$2/namelist_ice_ref"; cp "$ICE_DECK" "$2/namelist_ice_cfg"
  render_deck "$2/namelist_cfg" "$3" "$4"; cp "$1" "$2/nemo"
}

check_l2_log() {
  local o=$1/ocean.output
  grep -Eq 'ln_traldf_OFF += +F$' "$o" && grep -Eq 'ln_traldf_lap += +T$' "$o" \
    && grep -Eq 'ln_traldf_blp += +F$' "$o" && grep -Eq 'ln_traldf_lev += +F$' "$o" \
    && grep -Eq 'ln_traldf_hor += +F$' "$o" && grep -Eq 'ln_traldf_iso += +T$' "$o" \
    && grep -Eq 'ln_traldf_triad += +F$' "$o" && grep -Eq 'ln_traldf_msc += +T$' "$o" \
    && grep -Eq 'nn_aht_ijk_t += +20$' "$o" \
    && grep -Eq 'rn_Ud += +1\.7999999999999999E-002 +m/s$' "$o" \
    && grep -Eq 'rn_Ld += +200000\.00000000000 +m$' "$o" \
    && grep -Fq 'traldf_iso_lap   : iso-neutral standard laplacian diffusive operator' "$o" \
    && grep -Eq 'ln_ldfeiv += +F$' "$o" || fail "$1 did not resolve ORCA2 namtra_ldf with EIV off"
  grep -Eq 'ln_trabbl += +F$' "$o" && grep -Eq 'ln_trabbc += +F$' "$o" \
    || fail "$1 enabled BBL or geothermal heating"
}

run_one() { # binary, dir, itend, stocklist, done|boundary
  local rc=0
  stage_run "$1" "$2" "$3" "$4"
  (cd "$2"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$2/run.exit"
  [[ -f "$2/ocean.output" && -f "$2/time.step" ]] || fail "$2 produced no NEMO log"
  check_l2_log "$2"
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
    for name in $(printf 'oracle_ocean_surface_input_kt%08d.bin oracle_step_entry_kt%08d.bin' "$kt" "$kt"); do
      for dir in "$TWIN_A" "$TWIN_B" "$TWIN_C"; do [[ -s "$dir/$name" ]] || fail "$dir lacks $name"; done
      cmp -s "$TWIN_A/$name" "$TWIN_B/$name" && cmp -s "$TWIN_A/$name" "$TWIN_C/$name" \
        || fail "twins differ in $name"
    done
  done
  for dir in "$TWIN_A" "$TWIN_B" "$TWIN_C"; do [[ -s "$dir/oracle_rktracer_stage3_kt00000001.bin" ]] || fail "$dir lacks stage-3 tracer record"; done
  cmp -s "$TWIN_A/oracle_rktracer_stage3_kt00000001.bin" "$TWIN_B/oracle_rktracer_stage3_kt00000001.bin" \
    && cmp -s "$TWIN_A/oracle_rktracer_stage3_kt00000001.bin" "$TWIN_C/oracle_rktracer_stage3_kt00000001.bin" \
    || fail 'twins differ in stage-3 tracer record'
  for step in $(seq 2 "$steps"); do
    name=$(printf 'ORCA2G1_%08d_restart.nc' "$step")
    [[ -f "$TWIN_A/$name" || -f "$TWIN_B/$name" || -f "$TWIN_C/$name" ]] || fail "no twin wrote restart step $step"
  done
  [[ -f "$REF/ORCA2G1_$(printf '%08d' "$steps")_restart.nc" ]] || fail "writer-free reference lacks restart step $steps"
  "$G5_TOOL" --check-surface-payload "$TWIN_A" "$steps" >/dev/null
  check_acting "$TWIN_A"
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$G5_DECK" "$ICE_DECK" "$REF_BIN" "$REC_BIN" >"$EVIDENCE/manifest/producers.sha256"
  printf 'L2_ACQUISITION_PASS steps=1..%s restarts=2..%s evidence=%s\n' "$steps" "$steps" "$EVIDENCE"
}

preflight
[[ "$mode" == --run ]] || { printf 'WOULD_RUN smoke, boundary, twins A/B/C into %s\n' "$EVIDENCE"; exit 0; }
for dir in "$SMOKE" "$REF" "$TWIN_A" "$TWIN_B" "$TWIN_C"; do [[ ! -e "$dir" ]] || fail "run directory exists: $dir"; done
mkdir -p "$EVIDENCE"
run_one "$REC_BIN" "$SMOKE" 2 "$(pad10 2)" done; check_acting "$SMOKE"
run_one "$REF_BIN" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps "$REF"); steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
((steps >= 2)) || fail "NEMO completed only $completed steps"
run_one "$REC_BIN" "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$REC_BIN" "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$REC_BIN" "$TWIN_C" "$steps" "$(list_c "$steps")" done
admit "$steps"
