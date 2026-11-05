#!/usr/bin/env bash
# Sourced by the O8--O16 master: do not execute directly.

readonly C_HERE=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly C_ACK=RUN_FRESH_R5_REF_REC_C2_2_C3_100_C4_2920_DAILY_NP2
readonly C_SOURCE_RUN=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2
readonly C_PPSRC=${C_REAL_PPSRC:-$BASE_BLD/ppsrc/nemo}
readonly C_O8_DIR=$C_HERE/../nemo_si3_o8_o16_master_acquisition
readonly C_OBSERVER_UNITS="dynhpg dynspg_ts dynvor eosbn2 icedyn_adv_pra icedyn_rhg_evp icestp icethd sbcblk stp2d stprk3 stprk3_stg traadv traqsr"

c_apply_patches() {
  local dir=$1
  for unit in stprk3 sbcssm sbcmod icestp iceupdate; do
    patch --batch --fuzz=0 "$dir/$unit.f90" "$C_HERE/${unit}_c.patch" >/dev/null
  done
}

c_placement_gate() {
  local dir=$1 plant=${2:-none}
  "$PY" - "$dir" "$plant" <<'PY'
import sys
from pathlib import Path
d=Path(sys.argv[1]); plant=sys.argv[2]
checks={
 'stprk3.f90':[('CALL c_record_step(5','CALL sbc'),('CALL stp_RK3_stg( 3','CALL c_record_step(6')],
 'sbcssm.f90':[('CALL c_record_ssm(1','zts(:,:,jp_tem)'),("CALL iom_put( 'frq_m'",'CALL c_record_ssm(2')],
 'sbcmod.f90':[('CALL c_record_sbc(3','CALL ice_stp'),('utauU  (ji,jj) =','CALL c_record_sbc(4')],
 'icestp.f90':[('CALL o8o16_set_context','IF( MOD( kt-1, nn_fsbc )'),('CALL ice_var_agg(2)','CALL o8o16_update(15)'),('CALL o8o16_update(15)','CALL ice_update_flx'),('CALL ice_update_flx','CALL o8o16_update(16)')],
 'iceupdate.f90':[('CALL o8o16_tau(19','IF( MOD( kt-1, nn_fsbc )'),('CALL o8o16_tau(20',"CALL timing_stop('iceupdate')")],
}
for name,pairs in checks.items():
    text=(d/name).read_text()
    for i,(a,b) in enumerate(pairs):
        ai=text.find(a); bi=text.find(b,ai+1)
        if plant=='source-placement' and name=='sbcssm.f90' and i==0: ai=bi+1
        if ai<0 or bi<0 or ai>=bi: raise SystemExit(f'STATUS PLANT-FIRED: {name}: {a} !< {b}')
    body='\n'.join(line for line in text.splitlines() if 'CALL c_record_' in line or 'CALL o8o16_' in line)
    if '=' in body: raise SystemExit(f'STATUS PLANT-FIRED: observer assignment in {name}')
print('PASS_C_OBSERVER_PLACEMENT_AND_ASSIGNMENT_FREE')
PY
}

c_patch_preflight() {
  local scratch=$1 unit
  mkdir -p "$scratch/ppsrc"
  for unit in stprk3 sbcssm sbcmod icestp iceupdate; do cp "$C_PPSRC/$unit.f90" "$scratch/ppsrc/$unit.f90"; done
  c_apply_patches "$scratch/ppsrc"
  c_placement_gate "$scratch/ppsrc"
  expect_refusal c_placement_gate "$scratch/ppsrc" source-placement
  ! rg -n '(^|[^=!<>])=([^=]|$)' "$C_HERE"/*_c.patch | rg 'CALL c_record|CALL o8o16' >/dev/null || refuse 'observer patch contains assignment'
}

c_stub_run() {
  local scratch=$1
  local stub_root=$scratch/stub_nemo stub_ev=$scratch/stub_evidence
  mkdir -p "$stub_root/cfgs/BASE/BLD/ppsrc/nemo" "$stub_root/cfgs/SHARED"
  env C_LADDER_STUB=1 C_PROFILE_SKIP_STUB=1 C_LADDER_OPERATOR_ACK="$C_ACK" \
    C_REAL_PPSRC="$C_PPSRC" \
    MIN_FREE_RAM_GB=1 MIN_FREE_DISK_KB=1024 MAX_EVIDENCE_DISK_KB=1048576 \
    NEMO_ROOT="$stub_root" NEMO_SOURCE_ROOT="$stub_root" BASE_BLD="$stub_root/cfgs/BASE/BLD" \
    REF_CFG=C_STUB_REF REC_CFG=C_STUB_REC EVIDENCE_ROOT="$scratch" \
    "$C_HERE/run.sh" --run "$stub_ev" > "$scratch/stub_run.log"
  grep -q '^STUB_REACHED_EXACT_NEMO_START: C2 REF np=2$' "$scratch/stub_run.log" || refuse 'stub did not reach exact --run NEMO-start point'
  local retained_root=$scratch/retained
  mkdir -p "$retained_root"
  expect_refusal env C_LADDER_STUB=1 C_PROFILE_SKIP_STUB=1 C_PLANT=retention C_LADDER_OPERATOR_ACK="$C_ACK" \
    C_REAL_PPSRC="$C_PPSRC" \
    MIN_FREE_RAM_GB=1 MIN_FREE_DISK_KB=1024 MAX_EVIDENCE_DISK_KB=1048576 \
    NEMO_ROOT="$stub_root" NEMO_SOURCE_ROOT="$stub_root" BASE_BLD="$stub_root/cfgs/BASE/BLD" \
    REF_CFG=C_RETAIN_REF REC_CFG=C_RETAIN_REC EVIDENCE_ROOT="$retained_root" \
    "$C_HERE/run.sh" --run "$retained_root/evidence"
  find "$retained_root/refused" -maxdepth 1 -type d -name 'evidence_*' | grep -q . || refuse 'retention plant did not retain refused evidence'
}

c_run_preflight() {
  local scratch
  scratch=$(mktemp -d "$TMPDIR/c_ladder_profile.XXXXXX"); ACTIVE_SCRATCH=$scratch
  "$PY" "$C_HERE/preflight.py" > "$scratch/source.json"
  for p in drop-sbc-ssm swap-flux-update pretend-mlf; do expect_refusal "$PY" "$C_HERE/preflight.py" --plant "$p"; done
  c_patch_preflight "$scratch"
  for rung in C2 C3 C4; do "$PY" "$C_HERE/deck.py" --destination "$scratch/deck_$rung" --rung "$rung" >/dev/null; done
  expect_refusal ram_gate 999999
  expect_refusal disk_gate "$scratch" 999999999999999
  "$PY" -m json.tool "$C_HERE/record_schema.json" >/dev/null
  "$PY" "$C_HERE/scorer.py" --self-test baseline --scratch "$scratch" >/dev/null
  for p in numeric truncation wrong-parity; do expect_refusal "$PY" "$C_HERE/scorer.py" --self-test "$p" --scratch "$scratch"; done
  mkdir -p "$scratch/passivity_ref" "$scratch/passivity_rec"
  for f in ocean_0000 ocean_0001 ice_0000 ice_0001; do printf 'same\n' > "$scratch/passivity_ref/$f"; printf 'same\n' > "$scratch/passivity_rec/$f"; done
  c_passivity_gate "$scratch/passivity_ref" "$scratch/passivity_rec" none
  expect_refusal c_passivity_gate "$scratch/passivity_ref" "$scratch/passivity_rec" mismatch
  [[ ${C_PROFILE_SKIP_STUB:-0} == 1 ]] || c_stub_run "$scratch"
  printf 'PASS_C_MASTER_PLANTS_SOURCE_NUMERIC_PASSIVITY_TRUNCATION_PARITY_DISK_RAM_RETENTION\n'
  printf 'PREFLIGHT_OK: exact --run stub reached C2 REF NEMO start; no real NEMO integration was run\n'
  rm -rf -- "$scratch"; ACTIVE_SCRATCH=""
}

c_render_build() {
  local root=$1 instrument=$2 unit flags_text link_flags
  local -a link_objects
  cp -a "$BASE_BLD" "$root"
  flags_text=$(production_flags); read -r -a flags <<< "$flags_text"
  if [[ "$instrument" == 1 ]]; then
    c_apply_patches "$root/ppsrc/nemo"
    cp "$C_HERE/c_ladder_writer.F90" "$root/ppsrc/nemo/c_ladder_writer.f90"
    cp "$C_O8_DIR/o8_o16_master_writer.F90" "$root/ppsrc/nemo/o8_o16_master_writer.f90"
    mpif90 -c -cpp "${flags[@]}" -I"$root/inc" -J"$root/inc" "$root/ppsrc/nemo/o8_o16_master_writer.f90" -o "$root/obj/o8_o16_master_writer.o"
    mpif90 -c -cpp "${flags[@]}" -I"$root/inc" -J"$root/inc" "$root/ppsrc/nemo/c_ladder_writer.f90" -o "$root/obj/c_ladder_writer.o"
    for unit in stprk3 sbcssm sbcmod icestp iceupdate; do
      mpif90 -c "${flags[@]}" -I"$root/inc" -J"$root/inc" "$root/ppsrc/nemo/$unit.f90" -o "$root/obj/$unit.o"
    done
  else
    # R5FULLENTRY is a physics-identical oracle build with legacy dump streams.
    # Recompile exactly those MY_SRC observer units from pristine NEMO source so
    # REF (and especially year-long C4) is an observer-free standard executable.
    for unit in $C_OBSERVER_UNITS; do
      local source
      if [[ "$unit" == stprk3 ]]; then source=$NEMO_SOURCE_ROOT/src/OCE/stprk3.F90
      else source=$(find "$NEMO_SOURCE_ROOT/src" -iname "$unit.F90" -print -quit); fi
      [[ -f "$source" ]] || refuse "missing pristine NEMO source for $unit"
      cpp -Dkey_nosignedzero -Dkey_si3 -Dkey_qco -Dkey_vco_1d3d -Dkey_RK3 -P -traditional \
        -I"$BASE_BLD/../WORK" "$source" > "$root/ppsrc/nemo/$unit.f90"
      ! rg -q 'oracle_|l[1234]_dump|l3xchg_dump' "$root/ppsrc/nemo/$unit.f90" \
        || refuse "REF sanitization left an observer in $unit"
      mpif90 -c "${flags[@]}" -I"$root/inc" -J"$root/inc" "$root/ppsrc/nemo/$unit.f90" -o "$root/obj/$unit.o"
    done
  fi
  link_flags=$(awk -F'= ' '/^export LDFLAGS = / {print $2; exit}' "$BASE_BLD/Makefile")
  mapfile -t link_objects < <("$PY" "$C_HERE/link_objects.py" "$root/Makefile" --object-dir "$root/obj")
  if [[ "$instrument" == 1 ]]; then link_objects+=("$root/obj/o8_o16_master_writer.o" "$root/obj/c_ladder_writer.o"); fi
  # shellcheck disable=SC2086
  mpif90 -o "$root/bin/nemo.exe" "${link_objects[@]}" $link_flags
}

c_stage_and_run() {
  local rung=$1 arm=$2 build=$3 dir
  dir=$EVIDENCE/${arm}_${rung}
  local budget
  case "$rung" in C2) budget=900 ;; C3) budget=21600 ;; C4) budget=10800 ;; esac
  "$PY" "$C_HERE/deck.py" --destination "$dir" --rung "$rung" >/dev/null
  cp "$build/bin/nemo.exe" "$dir/nemo"
  if [[ "$arm" == REC ]]; then mkdir -p "$dir/c_ladder_records" "$dir/o8_o16_master_records" "$dir/o17_master_records"; fi
  if [[ ${C_LADDER_STUB:-0} == 1 ]]; then
    printf 'STUB_REACHED_EXACT_NEMO_START: %s %s np=2\n' "$rung" "$arm"
    touch "$dir/STUB_WOULD_START_NEMO"
    return 99
  fi
  (cd "$dir" && timeout "$budget" mpirun -np 2 --oversubscribe ./nemo >run.user.stdout.log 2>&1)
  [[ -s "$dir/ocean.output" && -s "$dir/output.namelist.dyn" && -s "$dir/output.namelist.ice" ]] \
    || refuse "$rung $arm lacks required resolved outputs"
  ! grep -Eiq 'E[[:space:]]+R[[:space:]]+R[[:space:]]+O[[:space:]]+R|ABORT' "$dir/ocean.output" "$dir/run.user.stdout.log" \
    || refuse "fatal token in $rung $arm"
  touch "$dir/NEMO_RUN_COMPLETE"
}

c_compare_passivity() {
  local ref=$EVIDENCE/REF_C2 rec=$EVIDENCE/REC_C2 f
  c_passivity_gate "$ref" "$rec" none \
    ORCA2_00000002_restart_0000.nc ORCA2_00000002_restart_0001.nc \
    ORCA2_00000002_restart_ice_0000.nc ORCA2_00000002_restart_ice_0001.nc
  printf 'PASS_C2_REF_REC_BYTE_PASSIVITY\n'
}

c_passivity_gate() {
  local ref=$1 rec=$2 plant=$3; shift 3
  local -a files=("$@")
  local f
  [[ ${#files[@]} -gt 0 ]] || files=(ocean_0000 ocean_0001 ice_0000 ice_0001)
  for f in "${files[@]}"; do
    if [[ "$plant" == mismatch && "$f" == "${files[0]}" ]]; then refuse 'planted REF-vs-REC byte mismatch'; return 1; fi
    if ! cmp -s "$ref/$f" "$rec/$f"; then refuse "REF-vs-REC passivity mismatch: $f"; return 1; fi
  done
}

c_measure_and_retain() {
  local c2_kib c3_kib c4_kib count
  c2_kib=$(( $(du -sk "$EVIDENCE/REF_C2" | awk '{print $1}') + $(du -sk "$EVIDENCE/REC_C2" | awk '{print $1}') ))
  c3_kib=$(du -sk "$EVIDENCE/REC_C3" | awk '{print $1}')
  c4_kib=$(du -sk "$EVIDENCE/REF_C4" | awk '{print $1}')
  [[ "$c2_kib" -le 6291456 ]] || refuse 'C2 exceeded 6 GiB'
  [[ "$c3_kib" -le 157286400 ]] || refuse 'C3 exceeded 150 GiB'
  [[ "$c4_kib" -le 31457280 ]] || refuse 'C4 exceeded 30 GiB'
  count=$(find "$EVIDENCE/REF_C4" -maxdepth 1 -type f -name 'ORCA2_*restart*.nc' | wc -l)
  [[ "$count" -eq 1460 ]] || refuse "C4 daily ocean+ice restart count $count != 1460"
  count=$(find "$EVIDENCE/REC_C3/c_ladder_records" -type f -name '*.bin' | wc -l)
  [[ "$count" -eq 1200 ]] || refuse "C3 C-record count $count != 1200"
  count=$(find "$EVIDENCE/REC_C3/o8_o16_master_records" -type f \( -name 'o8o16_kind15_*' -o -name 'o8o16_kind16_*' \) | wc -l)
  [[ "$count" -eq 200 ]] || refuse "C3 update-flux record count $count != 200"
  count=$(find "$EVIDENCE/REC_C3/o8_o16_master_records" -type f \( -name 'o8o16_kind19_*' -o -name 'o8o16_kind20_*' \) | wc -l)
  [[ "$count" -eq 400 ]] || refuse "C3 stress record count $count != 400"
  count=$(find "$EVIDENCE/REC_C3/o17_master_records" -type f -name 'o17_kind34_*' | wc -l)
  [[ "$count" -eq 200 ]] || refuse "C3 private stress record count $count != 200"
  "$PY" - "$EVIDENCE/manifest/resource_usage.json" "$c2_kib" "$c3_kib" "$c4_kib" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'C2_kib':int(sys.argv[2]),'C3_kib':int(sys.argv[3]),'C4_kib':int(sys.argv[4]),'retention':'ALL_RAW_RECORDS_RESTARTS_LOGS_AND_MANIFESTS_UNTIL_ADMISSION'},indent=2,sort_keys=True)+'\n')
PY
}

c_write_manifest() {
  local commit
  commit=$(git -C "$REPO" rev-parse HEAD)
  "$PY" - "$EVIDENCE/manifest/run_manifest.json" "$REPO" "$C_HERE" "$REF_ROOT/BLD/bin/nemo.exe" "$REC_ROOT/BLD/bin/nemo.exe" "$commit" <<'PY'
import hashlib,json,sys
from pathlib import Path
out,repo,here,ref,rec,commit=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]),Path(sys.argv[5]),sys.argv[6]
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources=[here/name for name in ('run.sh','c_profile.sh','preflight.py','deck.py','scorer.py','link_objects.py','c_ladder_writer.F90','record_schema.json','namelist_overlays.json')]
sources += sorted(here.glob('*_c.patch'))
prereg=repo/'docs/ocean/fidelity/PREREG_nemo_coupled_c_ladder_acquisition.md'
manifest={
 'status':'PRERUN_MANIFEST','producer_commit':commit,
 'source_sha256':{str(p.relative_to(repo)):sha(p) for p in sources+[prereg]},
 'binary_sha256':{'observer_free_REF':sha(ref),'observer_REC':sha(rec)},
 'source_configuration':'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY',
 'forcing_start':'EXACT_R339_R5_STAGED_OMIP_CORE_II_INITIAL_DECK','mpi_ranks':2,
 'runs':{'C2':{'steps':2,'arms':['REF','REC'],'budget_seconds':900,'budget_gib':6},'C3':{'steps':100,'si3_calls':50,'arms':['REC'],'budget_seconds':21600,'budget_gib':150},'C4':{'steps':2920,'arms':['REF'],'budget_seconds':10800,'budget_gib':30,'restart_stride_steps':8,'expected_restart_shards':1460}},
 'scoring':{'C0':'0_ULP','C1':'0_ULP','C2':'NOT_YET_SCORABLE_NO_COMPOSED_LEGOESM_COUPLED_STEP','C3':'NOT_YET_SCORABLE_NO_COMPOSED_LEGOESM_COUPLED_STEP','C4':'NOT_YET_SCORABLE_NO_COMPOSED_LEGOESM_COUPLED_STEP'},
 'retention':'ALL_RAW_RECORDS_RESTARTS_LOGS_RESOLVED_NAMELISTS_BINARIES_AND_MANIFESTS_UNTIL_ADMISSION',
}
out.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY
}

c_real_run() {
  [[ ${C_LADDER_OPERATOR_ACK:-} == "$C_ACK" ]] || refuse "set C_LADDER_OPERATOR_ACK=$C_ACK"
  if [[ ${C_LADDER_STUB:-0} != 1 ]]; then
    [[ -z $(git -C "$REPO" status --porcelain --untracked-files=all) ]] || refuse '--run requires a clean committed tree'
  fi
  [[ ! -e "$REF_ROOT" && ! -e "$REC_ROOT" && ! -e "$EVIDENCE" ]] || refuse 'fresh REF/REC/evidence targets must not exist'
  c_run_preflight
  ram_gate "${MIN_FREE_RAM_GB:-150}"; disk_gate "$EVIDENCE" "${MIN_FREE_DISK_KB:-157286400}"
  mkdir -p "$EVIDENCE/manifest" "$REF_ROOT" "$REC_ROOT"; ACTIVE_RUN=$EVIDENCE; CREATED_REF=1; CREATED_REC=1
  [[ ${C_PLANT:-none} != retention ]] || refuse 'retention plant after evidence creation'
  if [[ ${C_LADDER_STUB:-0} == 1 ]]; then
    mkdir -p "$REF_ROOT/BLD/bin" "$REC_ROOT/BLD/bin"
    : > "$REF_ROOT/BLD/bin/nemo.exe"; : > "$REC_ROOT/BLD/bin/nemo.exe"
    c_stage_and_run C2 REF "$REF_ROOT/BLD" || [[ $? == 99 ]]
    ACTIVE_RUN=""; return 0
  fi
  c_render_build "$REF_ROOT/BLD" 0
  c_render_build "$REC_ROOT/BLD" 1
  c_write_manifest
  RUN_STARTED_EPOCH=$(date +%s)
  c_stage_and_run C2 REF "$REF_ROOT/BLD"
  c_stage_and_run C2 REC "$REC_ROOT/BLD"
  c_compare_passivity
  c_stage_and_run C3 REC "$REC_ROOT/BLD"
  c_stage_and_run C4 REF "$REF_ROOT/BLD"
  "$PY" "$C_HERE/scorer.py" --evidence "$EVIDENCE" --ppsrc "$REC_ROOT/BLD/ppsrc/nemo" > "$EVIDENCE/manifest/C0_C1_score.json"
  c_measure_and_retain
  touch "$EVIDENCE/RUN_COMPLETE"; ACTIVE_RUN=""
  printf 'ACQUISITION_DONE: %s\n' "$EVIDENCE"
}

c_profile_main() {
  case "$1" in
    --preflight-only) c_run_preflight ;;
    --run) c_real_run ;;
    --move-aside-builds) move_builds_aside ;;
    *) refuse 'C profile supports --preflight-only, --run, or --move-aside-builds' ;;
  esac
}
