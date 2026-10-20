#!/usr/bin/env bash
# Lane B rung L1 acquisition (operator executed only).
# L1 = certified G5 + ORCA2's bottom boundary layer, verbatim
# ORCA2_OMIP_L4/EXP00/namelist_cfg:268-272 (ln_trabbl = T, nn_bbl_ldf = 1,
# nn_bbl_adv = 0, rn_ahtbbl = 1000, rn_gambbl = 10); deck = G5's with those
# five lines.  ln_trabbl is a run-time switch (tra_bbl_init trabbl.f90:527;
# bbl :357 and tra_bbl :154, stage 3 only, stprk3_stg.f90:560/:739); trabbl.o
# in both reused binaries is ORCA2_OMIP_L4's, so NOTHING is built:
#  - writer-free reference: G5's pinned writer-free binary;
#  - record: G5's pinned record binary (step-entry, stage, surface-input frames).
# Inputs = G5's sixteen, nothing new.  Records = G5's set.  BBL acts only on the
# stage-3 tracer update, so the kt-2 step-entry record (state after kt 1) must
# keep G5's admitted u, v and sea level bit for bit and change T/S somewhere,
# only in columns touching a wet face whose bottom levels differ (l1_records.py).
# Default is read-only preflight; --run smokes, finds NEMO's boundary with the
# writer-free binary, then acquires three twins.
set -Eeuo pipefail

fail() { printf 'REFUSE: L1 acquisition: %s\n' "$*" >&2; exit 66; }
trap 'fail "unexpected failure at line ${BASH_LINENO[0]:-${LINENO}}"' ERR

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly G5_TOOL=$here/../nemo_testcase_g5_orca2ice/run.sh
readonly G5_DECK=$here/../nemo_testcase_g5_orca2ice/namelist_cfg
readonly CHECKER=$here/l1_records.py
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_BIN=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe
readonly REC_BIN=$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/bin/nemo.exe
readonly REF_BIN_SHA=e8ebf4ae617f7aa61475776fcbe96b09e04bf09e7b846af39166bee1155808c7
readonly REC_BIN_SHA=41cd9a63844218e7c37e75708813649603b80df0366102b8ce3e340f89a60275
readonly G5_DECK_SHA=c57eeb83d8f59c5b1d44df1295a10f5f30d28537358c8d327fcecc3d538f569e
readonly L1_DECK_SHA=d922a0f24a7fbac889bc1a733b6e21137ac846f0807d38ef262bcbc22ea00d74
readonly ICE_DECK_SHA=9d5ae3fae3878a74c4fe88f0fd8db40129fe97ca6ffa6d699121d1cd44713dd8
readonly ICE_REF=$NEMO_ROOT/cfgs/SHARED/namelist_ice_ref
readonly FSBC=2
readonly INPUT_ROOT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/oracle_g4_r21/p6_a
readonly G5_ENTRY=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round31/oracle_g5_r31/twin_a/oracle_step_entry_kt00000002.bin
readonly G5_ENTRY_SHA=930211d3a83ebb315165bdac5da60e8ec56b328cac1a20442a406670400ca987
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/round16/oracle_l1_r16}
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

repo=$(git -C "$here" rev-parse --show-toplevel) || fail 'not in worktree'
readonly PYPATH=$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src
check_bbl_entry() { # dir: BBL acted in the kt-2 entry record, only where NEMO's code allows
  PYTHONPATH=$PYPATH JAX_PLATFORMS=cpu /home/dbalwada/legoESM/.venv/bin/python "$CHECKER" \
    --check-bbl-entry "$1" --g5-entry "$G5_ENTRY" --domain "$INPUT_ROOT/domain_cfg.nc" \
    || fail "$1 kt-2 entry record does not show BBL as NEMO's code places it"
}

mode=${1:---preflight-only}
case "$mode" in --preflight-only|--run) ;; --check-bbl-entry) check_bbl_entry "$2"; exit 0 ;; *) exit 64 ;; esac

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ]] \
  || fail 'commit the exact acquisition tool before running it'
readonly COMMIT=$(git -C "$repo" rev-parse HEAD)

preflight() {
  local f want b m sec
  for f in "$G5_TOOL" "$G5_DECK" "$CHECKER" "$DECK" "$ICE_DECK" "$REF_BIN" "$REC_BIN" "$ICE_REF" "$G5_ENTRY"; do
    [[ -f "$f" ]] || fail "missing $f"
  done
  for f in "${INPUT_FILES[@]}"; do
    want=$(input_sha "$f"); [[ ${#want} == 64 ]] || fail "G5 tool pins no SHA for $f"
    [[ -f "$INPUT_ROOT/$f" && "$(sha "$INPUT_ROOT/$f")" == "$want" ]] \
      || fail "missing or changed input $f"
  done
  [[ "$(sha "$DECK")" == "$L1_DECK_SHA" && "$(sha "$ICE_DECK")" == "$ICE_DECK_SHA" ]] \
    || fail 'committed L1 deck changed'
  [[ "$(sha "$G5_DECK")" == "$G5_DECK_SHA" ]] || fail "G5's certified deck changed"
  # L1's deck = G5's + ORCA2's five &nambbl lines (and its three header lines), nothing else
  [[ "$(diff "$G5_DECK" "$DECK" | grep -c '^>')" == 9 && "$(diff "$G5_DECK" "$DECK" | grep -c '^<')" == 2 ]] \
    || fail 'L1 deck is not G5 plus the nambbl lines'
  [[ "$(sha "$G5_ENTRY")" == "$G5_ENTRY_SHA" ]] || fail "G5's admitted kt-2 entry record changed"
  [[ "$(sha "$REF_BIN")" == "$REF_BIN_SHA" ]] || fail 'G5 writer-free binary changed'
  [[ "$(sha "$REC_BIN")" == "$REC_BIN_SHA" ]] || fail 'G5 record binary changed'
  # tra_bbl_init / bbl / tra_bbl_dif compiled into both binaries exactly as ORCA2_OMIP_L4.
  for b in ORCA2G5_OMIP_L4_R31REF_FIX1 ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1; do
    for m in trabbl.o trabbc.o; do
      for sec in .text .rodata; do
        cmp -s <(objcopy -O binary --only-section=$sec "$NEMO_ROOT/cfgs/$b/BLD/obj/$m" /dev/stdout) \
               <(objcopy -O binary --only-section=$sec "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/obj/$m" /dev/stdout) \
          || fail "$b $m $sec differs from ORCA2_OMIP_L4"
      done
    done
    cmp -s "$NEMO_ROOT/cfgs/$b/BLD/ppsrc/nemo/trabbl.f90" "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/trabbl.f90" \
      || fail "$b trabbl.f90 differs from ORCA2_OMIP_L4"
  done
  # The record binary's stage program and 2-D eos_rab are ORCA2_OMIP_L4's object code.
  for m in stprk3_stg.o eosbn2.o; do
    cmp -s <(objcopy -O binary --only-section=.text "$NEMO_ROOT/cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/obj/$m" /dev/stdout) \
           <(objcopy -O binary --only-section=.text "$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/BLD/obj/$m" /dev/stdout) \
      || fail "record binary $m .text differs from ORCA2_OMIP_L4"
  done
  printf 'L1_PREFLIGHT_OK commit=%s reference=G5_writer_free record=G5_record_binary (nothing built)\n' "$COMMIT"
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

check_l1_log() { # L1 resolved G5's switches plus ORCA2's five nambbl lines
  local o=$1/ocean.output
  grep -Eq 'nn_ice += +2$' "$o" || fail "$1 did not resolve nn_ice=2"
  grep -Eq 'ln_zdfcst = +T$' "$o" && grep -Eq 'ln_zdftke = +F$' "$o" || fail "$1 changed G5's vertical mixing"
  grep -Fq 'tra_bbl_init : bottom boundary layer initialisation' "$o" \
    && grep -Eq 'bottom boundary layer flag +ln_trabbl += +T$' "$o" \
    && grep -Eq 'diffusive bbl \(=1\) +or not \(=0\) +nn_bbl_ldf = +1$' "$o" \
    && grep -Eq 'advective bbl \(=1/2\) or not \(=0\) +nn_bbl_adv = +0$' "$o" \
    && grep -Eq 'diffusive bbl coefficient +rn_ahtbbl += +1000\.0+ +m2/s$' "$o" \
    && grep -Eq 'advective bbl coefficient +rn_gambbl += +10\.0+ +s$' "$o" \
    && ! grep -Fq 'Advective BBL' "$o" \
    || fail "$1 did not resolve ORCA2's diffusive BBL"
  grep -Eq 'Apply a geothermal heating at ocean bottom +ln_trabbc += +F$' "$o" \
    || fail "$1 switched geothermal heating on"
}

run_one() { # binary, directory, itend, stocklist, expected=done|boundary
  local rc=0
  stage_run "$1" "$2" "$3" "$4"
  (cd "$2"; mpirun -np 1 --oversubscribe ./nemo >run.user.log 2>&1) || rc=$?
  printf '%s\n' "$rc" >"$2/run.exit"
  [[ -f "$2/ocean.output" && -f "$2/time.step" ]] || fail "$2 produced no NEMO log"
  check_l1_log "$2"
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
  for step in $(seq 2 "$steps"); do
    name=$(printf 'ORCA2G1_%08d_restart.nc' "$step")
    [[ -f "$TWIN_A/$name" || -f "$TWIN_B/$name" || -f "$TWIN_C/$name" ]] \
      || fail "no twin wrote restart step $step"
  done
  [[ -f "$REF/ORCA2G1_$(printf '%08d' "$steps")_restart.nc" ]] \
    || fail "writer-free reference lacks restart step $steps"
  "$G5_TOOL" --check-surface-payload "$TWIN_A" "$steps" >/dev/null
  check_bbl_entry "$TWIN_A"
  mkdir -p "$EVIDENCE/manifest"
  printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
  sha256sum "$DECK" "$ICE_DECK" "$REF_BIN" "$REC_BIN" "$G5_ENTRY" >"$EVIDENCE/manifest/producers.sha256"
  printf 'L1_ACQUISITION_PASS steps=1..%s restarts=2..%s evidence=%s\n' "$steps" "$steps" "$EVIDENCE"
}

preflight
[[ "$mode" == --run ]] || { printf 'WOULD_RUN smoke, boundary, twins A/B/C into %s\n' "$EVIDENCE"; exit 0; }
for dir in "$SMOKE" "$REF" "$TWIN_A" "$TWIN_B" "$TWIN_C"; do
  [[ ! -e "$dir" ]] || fail "live run directory exists: $dir"
done
mkdir -p "$EVIDENCE"
# Smoke is the first NEMO execution; only after it passes is the boundary run allowed.
run_one "$REC_BIN" "$SMOKE" 2 "$(pad10 2)" done
check_bbl_entry "$SMOKE"
run_one "$REF_BIN" "$REF" 100 '2,4,6,8,10,12,14,16,18,20' boundary
completed=$(completed_steps "$REF")
steps=$((completed < 10 ? completed : 10)); steps=$((steps - steps % FSBC))
((steps >= 2)) || fail "NEMO completed only $completed steps"
run_one "$REC_BIN" "$TWIN_A" "$steps" "$(list_a "$steps")" done
run_one "$REC_BIN" "$TWIN_B" "$steps" "$(list_b "$steps")" done
run_one "$REC_BIN" "$TWIN_C" "$steps" "$(list_c "$steps")" done
admit "$steps"
