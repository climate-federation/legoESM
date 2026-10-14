#!/usr/bin/env bash
# Grid ladder G1 ACQUISITION (DECISION 117) -- OPERATOR-EXECUTED ONLY.
#
#   preflight (default):  run.sh
#   acquire:              run.sh --run
#   re-admit:             run.sh --admit-existing
#
# The agent that wrote this ran neither DOMAINcfg, makenemo nor NEMO.
#
# WHAT THIS ACQUIRES.  NEMO 5.0.2 on ORCA2's horizontal grid (T-pivot north
# fold, cyclic east-west, ORCA2's coast) with a FLAT 100 m bottom and ONE wet
# level, running the TSUNAMI program: uniform T/S, TSUNAMI's ssh bump at
# (ji,jj) = (51,74), zero forcing, no ice, advection / lateral mixing / drag
# OFF, split-explicit free surface, EEN, constant vertical mixing.
#
#   1. tools/DOMAINcfg builds domain_cfg.nc from ORCA2's shipped
#      ORCA_R2_zps_domcfg.nc (as the coordinates file) and ORCA2's own
#      land-sea mask over one 100 m level (domaincfg_namelist_cfg).
#      make_domain_cfg.exe exits without MPI_Finalize AFTER closing the
#      file, so mpirun's status is recorded but not judged: the domain is
#      admitted by check_domain_cfg.py on the file itself.  If a complete
#      domain_cfg.nc from an earlier --run is the only thing in the
#      evidence directory, --run admits it and resumes at step 2.
#   2. cfgs/ORCA2G1_OMIP_L1     = ORCA2_ICE_PISCES minus key_si3 key_top
#      key_xios (= the ORCA2_OMIP_L4 keys minus key_si3), OCE only, with
#      MY_SRC/usrdef_istate.F90 + usrdef_sbc.F90 + restart.F90 with ONE added
#      statement (the ssh bump on a read domain): the reference binary.
#      cfgs/ORCA2G1_OMIP_L1_P3  = the same + the read-only step writer.
#   3. Runs (mpirun -np 1): a 2-step SMOKE of the reference binary, the
#      100-step reference, the 100-step record (twin a: stage frames at
#      kt = 1..10, the final frame at every kt) and a 10-step twin b.
#   4. check_records.py: self-describing headers, twin identity, and the
#      record build's restarts bit-identical to the reference build's.
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE: unhandled command failure at run.sh line %s (exit %s)\n' \
    "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly REF_NAME=ORCA2G1_OMIP_L1
readonly RUN_NAME=${RUN_NAME:-ORCA2G1_OMIP_L1_P3}      # run_r3.sh: a NEW target name
readonly STEPS=100
readonly STAGE_STEPS=${STAGE_STEPS:-10}             # must equal the patch gate
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round1/oracle_g1_r1}
readonly SHIPPED_DOMCFG=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/ORCA_R2_zps_domcfg.nc
readonly SHIPPED_DOMCFG_SHA=7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839
readonly L4_KEYS_FILE=$NEMO_ROOT/cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm
readonly DOMTOOL=$NEMO_ROOT/tools/DOMAINcfg

mode=--preflight-only
case "${1:---preflight-only}" in
  --run|--preflight-only|--admit-existing) mode=${1:---preflight-only} ;;
  *) printf 'Usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run.sh is not inside a git worktree\n' >&2; exit 66; }
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]]; then
  printf 'REFUSE: legoESM worktree is dirty; commit the exact acquisition tool first\n' >&2
  exit 66
fi
readonly COMMIT=$(git -C "$repo_root" rev-parse HEAD)

readonly DECK=$here/namelist_cfg
readonly DOM_NML=$here/domaincfg_namelist_cfg
readonly ISTATE=$here/usrdef_istate.F90
readonly SBC=$here/usrdef_sbc.F90
readonly STEP_MODULE=$here/g1_step_record.F90
readonly STEP_PATCH=${STEP_PATCH:-$here/stprk3_g1_step_record.patch}
readonly BATHY_TOOL=$here/make_bathy_level.py
readonly CHECKER=$here/check_records.py
readonly DOMCHECK=$here/check_domain_cfg.py
readonly SHIPPED_STPRK3=$NEMO_ROOT/src/OCE/stprk3.F90
readonly SSH_PATCH=$here/restart_g1_usrdef_ssh.patch
readonly SHIPPED_RESTART=$NEMO_ROOT/src/OCE/IOM/restart.F90

for f in "$DECK" "$DOM_NML" "$ISTATE" "$SBC" "$STEP_MODULE" "$STEP_PATCH" \
         "$BATHY_TOOL" "$CHECKER" "$DOMCHECK" "$SHIPPED_STPRK3" "$SHIPPED_DOMCFG" \
         "$SSH_PATCH" "$SHIPPED_RESTART" "$L4_KEYS_FILE" "$DOMTOOL/namelist_ref" "$NEMO_ROOT/makenemo" \
         "$NEMO_ROOT/tools/maketools" "$NEMO_ROOT/cfgs/SHARED/namelist_ref" \
         "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm"; do
  [[ -f "$f" ]] || { printf 'REFUSE: missing input %s\n' "$f" >&2; exit 67; }
done
[[ "$(sha256sum "$SHIPPED_DOMCFG" | awk '{print $1}')" == "$SHIPPED_DOMCFG_SHA" ]] \
  || { printf 'REFUSE: the shipped ORCA2 domain file changed\n' >&2; exit 67; }

# The compiled keys: the ORCA2_OMIP_L4 build's WITHOUT key_si3 / key_top.
want_keys=$(sed -E 's/.*fppkeys *//' "$L4_KEYS_FILE" | xargs -n1 \
  | grep -vx -e key_si3 -e key_top | LC_ALL=C sort | xargs)
[[ "$want_keys" == "key_RK3 key_qco key_vco_1d3d" ]] \
  || { printf 'REFUSE: ORCA2_OMIP_L4 keys minus si3/top are "%s"; re-survey\n' "$want_keys" >&2; exit 68; }

# The record writer only ADDS lines to NEMO's own stprk3.F90, unfuzzed.
dry=$(mktemp -d)
trap 'rm -rf "$dry"' EXIT
cp "$SHIPPED_STPRK3" "$dry/stprk3.F90"
cp "$SHIPPED_RESTART" "$dry/restart.F90"
patch -s --fuzz=0 "$dry/stprk3.F90" <"$STEP_PATCH"
patch -s --fuzz=0 "$dry/restart.F90" <"$SSH_PATCH"
[[ "$(cat "$STEP_PATCH" "$SSH_PATCH" | grep -c '^-[^-]' || true)" -eq 0 ]] \
  || { printf 'REFUSE: a source patch removes lines\n' >&2; exit 68; }
[[ "$(grep -c '^+[^+]' "$SSH_PATCH")" -eq 1 ]] \
  || { printf 'REFUSE: the ssh patch adds more than its one statement\n' >&2; exit 68; }

render_deck() {        # $1 = target, $2 = nn_itend, $3 = nn_stocklist
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
         -e "s/^(   nn_stock    = ).*/\1    $2/" \
         -e "s/^(   nn_stocklist = ).*/\1$3/" "$DECK" >"$1"
  local moved
  moved=$(diff "$DECK" "$1" | grep -c '^>' || true)
  [[ "$moved" -eq 0 || "$moved" -eq 3 ]] \
    || { printf 'REFUSE: rendering changed %s deck lines, not the 3 run-length lines\n' "$moved" >&2; exit 68; }
  awk '/^&namrun/{p=1} p&&/^\//{exit} p' "$1" | grep -Ev 'nn_itend|nn_stock' >"$1.namrun"
  awk '/^&namrun/{p=1} p&&/^\//{exit} p' "$DECK" | grep -Ev 'nn_itend|nn_stock' \
    | cmp -s - "$1.namrun" \
    || { printf 'REFUSE: rendered namrun differs beyond the run length\n' >&2; exit 68; }
  rm -f "$1.namrun"
}
render_deck "$dry/smoke" 2 '2,0,0,0,0,0,0,0,0,0'
render_deck "$dry/ten" 10 '10,0,0,0,0,0,0,0,0,0'
render_deck "$dry/full" 100 '10,20,30,40,50,60,70,80,90,100'
cmp -s "$dry/full" "$DECK" \
  || { printf 'REFUSE: the committed deck is not the 100-step deck\n' >&2; exit 68; }
python3 -I -m py_compile "$BATHY_TOOL" "$CHECKER" "$DOMCHECK"
printf 'PREFLIGHT: patch applies; keys "%s"; commit %s\n' "$want_keys" "$COMMIT"

readonly SMOKE=$EVIDENCE/smoke_2step REF=$EVIDENCE/ref_100step
readonly TWIN_A=$EVIDENCE/p3_a_100step TWIN_B=$EVIDENCE/p3_b_10step
readonly DOMDIR=$EVIDENCE/domaincfg

domain_cfg_admit() {   # $1 = directory for the logs; refuses unless complete
  local plant
  for plant in missing-variable bad-level thin-cell wrong-jperio wrong-dims nonfinite flat-e3t; do
    if python3 -I "$DOMCHECK" --domain-cfg "$DOMDIR/domain_cfg.nc" --plant "$plant" \
         >"$1/domain_plant_${plant}.log" 2>&1; then
      printf 'REFUSE: domain plant %s stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$1/domain_plant_${plant}.log" \
      || { printf 'REFUSE: domain plant %s did not fire\n' "$plant" >&2; exit 72; }
  done
  python3 -I "$DOMCHECK" --domain-cfg "$DOMDIR/domain_cfg.nc" --shipped "$SHIPPED_DOMCFG" \
       --json "$1/domain_cfg_check.json" >/dev/null \
    || { printf 'REFUSE: %s is not a complete G1 domain file\n' "$DOMDIR/domain_cfg.nc" >&2; exit 70; }
  # Facts the receipt quotes (reported, not judged).
  grep -n 'close file: domain_cfg.nc ok' "$DOMDIR/ocean.output" | sed 's/^/G1_DOMAINCFG_CLOSE_LINE ocean.output:/' || true
  printf 'G1_DOMAINCFG_NOT_FINALISED_LINES %s\n' "$(grep -c 'exited without calling "finalize"' "$DOMDIR/run.user.log" || true)"
  printf 'G1_DOMAIN_CFG_SHA256 %s\n' "$(sha256sum "$DOMDIR/domain_cfg.nc" | awk '{print $1}')"
}

# The DOMAINcfg inputs of an existing domain_cfg.nc are the committed ones.
domain_cfg_same_inputs() {
  local dom_commit
  dom_commit=$(cat "$EVIDENCE/manifest/legoesm_commit_domaincfg" 2>/dev/null \
    || cat "$EVIDENCE/manifest/legoesm_commit" 2>/dev/null || true)
  git -C "$repo_root" cat-file -e "${dom_commit:-none}^{commit}" 2>/dev/null \
    || { printf 'REFUSE: the commit that ran DOMAINcfg ("%s") is not in this clone\n' "$dom_commit" >&2; exit 69; }
  git -C "$repo_root" diff --quiet "$dom_commit" HEAD -- "$DOM_NML" "$BATHY_TOOL" \
    || { printf 'REFUSE: the DOMAINcfg namelist or bathymetry tool changed since %s\n' "$dom_commit" >&2; exit 69; }
  cmp -s "$DOMDIR/namelist_cfg" "$DOM_NML" && cmp -s "$DOMDIR/namelist_ref" "$DOMTOOL/namelist_ref" \
    && cmp -s "$DOMDIR/make_domain_cfg.exe" "$DOMTOOL/BLD/bin/make_domain_cfg.exe" \
    && [[ "$DOMDIR/coordinates.nc" -ef "$SHIPPED_DOMCFG" && -f "$DOMDIR/bathy_level.nc" ]] \
    || { printf 'REFUSE: %s was not built from the committed DOMAINcfg inputs\n' "$DOMDIR" >&2; exit 69; }
  # An earlier admission's checksums, when present, must still hold.
  [[ ! -f "$DOMDIR/domaincfg.sha256" ]] || ( cd "$DOMDIR" && sha256sum --quiet -c domaincfg.sha256 >&2 ) \
    || { printf 'REFUSE: %s no longer matches its recorded checksums\n' "$DOMDIR" >&2; exit 69; }
  printf '%s\n' "$dom_commit"
}

admit() {
  local plant
  for plant in truncation nonfinite missing-frame twin-ulp missing-step; do
    if python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" \
         --steps "$STEPS" --stage-steps "$STAGE_STEPS" --plant "$plant" \
         >"$EVIDENCE/manifest/plant_${plant}.log" 2>&1; then
      printf 'REFUSE: record plant %s stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/manifest/plant_${plant}.log" \
      || { printf 'REFUSE: record plant %s did not fire\n' "$plant" >&2; exit 72; }
  done
  python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" --reference "$REF" \
    --steps "$STEPS" --stage-steps "$STAGE_STEPS" --restart-steps 10,100 \
    --json "$EVIDENCE/g1_round1_admission.json"
  ( cd "$EVIDENCE" && sha256sum g1_round1_admission.json domaincfg/domain_cfg.nc \
      p3_a_100step/mesh_mask.nc ref_100step/ORCA2G1_00000100_restart.nc \
      p3_a_100step/ORCA2G1_00000100_restart.nc ) >"$EVIDENCE/manifest/outputs.sha256"
  printf 'G1_DOMAIN_CFG_SHA256 %s\n' "$(sha256sum "$DOMDIR/domain_cfg.nc" | awk '{print $1}')"
  printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
}

if [[ "$mode" == --admit-existing ]]; then
  for target in "$DOMDIR/domain_cfg.nc" "$REF" "$TWIN_A" "$TWIN_B"; do
    [[ -e "$target" ]] || { printf 'REFUSE: nothing to admit, %s is absent\n' "$target" >&2; exit 69; }
  done
  mkdir -p "$EVIDENCE/manifest"
  domain_cfg_admit "$EVIDENCE/manifest"
  admit
  exit 0
fi

for name in "$REF_NAME" "$RUN_NAME"; do
  [[ ! -e "$NEMO_ROOT/cfgs/$name" ]] || {
    printf 'REFUSE: %s already exists; use --admit-existing or move it aside\n' \
      "$NEMO_ROOT/cfgs/$name" >&2; exit 69; }
done
resume=0
if [[ -e "$EVIDENCE" ]]; then
  # Only a domain-only directory (step 1 done, nothing after it) resumes.
  [[ -f "$DOMDIR/domain_cfg.nc" && "$(ls -A "$EVIDENCE" | LC_ALL=C sort | xargs)" == "domaincfg manifest" ]] || {
    printf 'REFUSE: evidence directory %s already exists; use --admit-existing\n' "$EVIDENCE" >&2; exit 69; }
  resume=1
  dom_commit=$(domain_cfg_same_inputs)
fi

if [[ "$mode" != --run ]]; then
  if [[ "$resume" -eq 1 ]]; then
    domain_cfg_admit "$dry"
    printf 'WOULD: admit the existing %s (DOMAINcfg ran at commit %s) and resume at the NEMO builds\n' \
      "$DOMDIR/domain_cfg.nc" "$dom_commit"
  else
    printf 'WOULD: maketools -m conda-scalarmath -n DOMAINcfg; mpirun -np 1 make_domain_cfg.exe in %s\n' "$DOMDIR"
  fi
  printf 'WOULD: makenemo -r ORCA2_ICE_PISCES -n %s|%s -d OCE -m conda-scalarmath del_key "key_si3 key_top key_xios"\n' \
    "$REF_NAME" "$RUN_NAME"
  printf 'WOULD: mpirun -np 1 ./nemo in %s/{smoke_2step,ref_100step,p3_a_100step,p3_b_10step}; then %s\n' \
    "$EVIDENCE" "$CHECKER"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

mkdir -p "$EVIDENCE/manifest" "$DOMDIR"
if [[ "$resume" -eq 1 && ! -e "$EVIDENCE/manifest/legoesm_commit_domaincfg" ]]; then
  mv "$EVIDENCE/manifest/legoesm_commit" "$EVIDENCE/manifest/legoesm_commit_domaincfg"
  mv "$EVIDENCE/manifest/inputs.sha256" "$EVIDENCE/manifest/inputs_domaincfg.sha256"
fi
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
sha256sum "$SHIPPED_DOMCFG" "$DECK" "$DOM_NML" "$ISTATE" "$SBC" "$STEP_MODULE" \
  "$STEP_PATCH" "$SSH_PATCH" "$SHIPPED_RESTART" "$BATHY_TOOL" "$CHECKER" "$DOMCHECK" \
  "$SHIPPED_STPRK3" "$DOMTOOL/namelist_ref" \
  "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  >"$EVIDENCE/manifest/inputs.sha256"

# ---- 1. domain_cfg.nc from NEMO's own tools/DOMAINcfg ----------------------
if [[ "$resume" -eq 1 ]]; then
  printf 'G1_DOMAINCFG_RESUME admitting the existing domain_cfg.nc (DOMAINcfg ran at commit %s)\n' "$dom_commit"
else
if [[ ! -x "$DOMTOOL/BLD/bin/make_domain_cfg.exe" ]]; then
  ( cd "$NEMO_ROOT/tools" && ./maketools -m conda-scalarmath -n DOMAINcfg )
fi
[[ -x "$DOMTOOL/BLD/bin/make_domain_cfg.exe" ]] \
  || { printf 'REFUSE: maketools did not produce make_domain_cfg.exe\n' >&2; exit 70; }
(
  cd "$DOMDIR"
  cp "$DOMTOOL/namelist_ref" namelist_ref
  cp "$DOM_NML" namelist_cfg
  ln -s "$SHIPPED_DOMCFG" coordinates.nc
  python3 -I "$BATHY_TOOL" --source "$SHIPPED_DOMCFG" --target bathy_level.nc | tee bathy_level.log
  cp "$DOMTOOL/BLD/bin/make_domain_cfg.exe" make_domain_cfg.exe
  # The tool exits without MPI_Finalize after closing domain_cfg.nc: the
  # status is recorded, and the file check below decides.
  dom_status=0
  mpirun -np 1 --oversubscribe ./make_domain_cfg.exe >run.user.log 2>&1 || dom_status=$?
  printf 'G1_DOMAINCFG_MPIRUN_EXIT %s\n' "$dom_status" | tee mpirun_exit.log
  [[ -f domain_cfg.nc ]] || { printf 'REFUSE: DOMAINcfg wrote no domain_cfg.nc\n' >&2; exit 70; }
)
fi
domain_cfg_admit "$EVIDENCE/manifest" | tee "$DOMDIR/domain_cfg_admission.log"
( cd "$DOMDIR" && sha256sum make_domain_cfg.exe namelist_cfg namelist_ref bathy_level.nc \
    domain_cfg.nc >domaincfg.sha256 )

# ---- 2. the two NEMO builds ------------------------------------------------
cd "$NEMO_ROOT"
build_one() {          # $1 = config name, $2 = 1 to add the record writer
  local name=$1 instrumented=$2 cfg=$NEMO_ROOT/cfgs/$1
  ./makenemo -r ORCA2_ICE_PISCES -n "$name" -d 'OCE' -m conda-scalarmath \
    del_key 'key_si3 key_top key_xios'
  mkdir -p "$cfg/MY_SRC"
  # cp, NOT cp -a: preserved mtimes let fcm skip a new file.
  cp "$ISTATE" "$SBC" "$cfg/MY_SRC/"
  # ln_read_cfg = T makes rst_read_ssh set ssh = 0 without calling the user
  # routine; this one ADDED statement applies the bump (both builds).
  cp "$SHIPPED_RESTART" "$cfg/MY_SRC/restart.F90"
  patch --fuzz=0 "$cfg/MY_SRC/restart.F90" <"$SSH_PATCH"
  if [[ "$instrumented" -eq 1 ]]; then
    cp "$SHIPPED_STPRK3" "$cfg/MY_SRC/stprk3.F90"
    cp "$STEP_MODULE" "$cfg/MY_SRC/"
    patch --fuzz=0 "$cfg/MY_SRC/stprk3.F90" <"$STEP_PATCH"
  fi
  touch "$cfg/MY_SRC/"*.F90
  ./makenemo -n "$name" -m conda-scalarmath
  [[ -x "$cfg/BLD/bin/nemo.exe" ]] \
    || { printf 'REFUSE: %s did not build\n' "$name" >&2; exit 70; }
  local pp=$cfg/BLD/ppsrc/nemo fkeys hits=0
  fkeys=$(sed -E 's/.*fppkeys *//' "$cfg/cpp_$name.fcm" | xargs -n1 | LC_ALL=C sort | xargs)
  [[ "$fkeys" == "$want_keys" ]] \
    || { printf 'REFUSE: %s compiled keys "%s", want "%s"\n' "$name" "$fkeys" "$want_keys" >&2; exit 70; }
  grep -q 'CALL stp_RK3' "$pp/nemogcm.f90" \
    || { printf 'REFUSE: %s does not call stp_RK3\n' "$name" >&2; exit 70; }
  grep -q 'ORCA2G1' "$pp/usrdef_istate.f90" && grep -q 'ORCA2G1' "$pp/usrdef_sbc.f90" \
    || { printf 'REFUSE: %s did not compile the G1 user routines\n' "$name" >&2; exit 70; }
  grep -q 'ORCA2G1: the ssh bump' "$pp/restart.f90" \
    || { printf 'REFUSE: %s did not compile the ssh-bump statement\n' "$name" >&2; exit 70; }
  grep -q 'g1r_open' "$pp/stprk3.f90" && hits=1
  [[ "$hits" -eq "$instrumented" ]] \
    || { printf 'REFUSE: %s writer presence %s, instrumented=%s\n' "$name" "$hits" "$instrumented" >&2; exit 70; }
}

run_one() {            # $1 = config name, $2 = run dir, $3 = nn_itend, $4 = stocklist
  local cfg=$NEMO_ROOT/cfgs/$1 dir=$2
  mkdir -p "$dir"
  render_deck "$dir/namelist_cfg" "$3" "$4"
  cp -L "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$dir/namelist_ref"
  cp "$DOMDIR/domain_cfg.nc" "$dir/domain_cfg.nc"
  cp "$cfg/BLD/bin/nemo.exe" "$dir/nemo"
  (
    cd "$dir"
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.log
    # No pipe: a failing mpirun must abort under set -e, not hide in a pipe.
    mpirun -np 1 --oversubscribe ./nemo >>run.user.log 2>&1
    printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.log
  )
  if grep -q 'E R R O R' "$dir/ocean.output"; then
    printf 'REFUSE: %s ocean.output reports an error\n' "$dir" >&2; exit 71
  fi
  # Facts the receipt quotes, read from NEMO's own log (reported, not judged).
  grep -E 'nn_e *=|ln_bt_auto|jpiglo|jpjglo|jpkglo|l_Iperio|l_NFold|c_NFtype|G1_BUMP|ourant' \
    "$dir/ocean.output" >"$dir/resolved_facts.log" || true
}

build_one "$REF_NAME" 0
build_one "$RUN_NAME" 1
sha256sum "$NEMO_ROOT/cfgs/$REF_NAME/BLD/bin/nemo.exe" \
  "$NEMO_ROOT/cfgs/$RUN_NAME/BLD/bin/nemo.exe" >"$EVIDENCE/manifest/binaries.sha256"

# ---- 3. runs: the mandatory 2-step smoke first -----------------------------
run_one "$REF_NAME" "$SMOKE" 2 '2,0,0,0,0,0,0,0,0,0'
run_one "$REF_NAME" "$REF" 100 '10,20,30,40,50,60,70,80,90,100'
run_one "$RUN_NAME" "$TWIN_A" 100 '10,20,30,40,50,60,70,80,90,100'
run_one "$RUN_NAME" "$TWIN_B" 10 '10,0,0,0,0,0,0,0,0,0'

# ---- 4. admission ----------------------------------------------------------
admit
