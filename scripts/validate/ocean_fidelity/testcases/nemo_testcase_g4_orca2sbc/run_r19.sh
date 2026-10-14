#!/usr/bin/env bash
# Grid ladder G4, round 3 (gridladder_r19) ACQUISITION -- OPERATOR-EXECUTED ONLY.
#
#   preflight (default):  run_r19.sh
#   acquire:              run_r19.sh --run
#   re-admit:             run_r19.sh --admit-existing
#
# The agent that wrote this ran neither makenemo nor NEMO.
#
# WHAT THIS ACQUIRES.  The admitted round-17 G4 program (same deck, domain,
# forcing files, keys) with NEMO's per-step SURFACE FORCING written at
# kt = 1..STEPS where the ocean reads it, plus the NCAR bulk's
# intermediates, through the G1 step writer (g1_step_record.F90, new frames
# only).  One NEW record build, ORCA2G1_OMIP_L1_P6 = the round-4 record
# build's sources (P5: G1 user routines, ssh bump, stage-frame writer,
# substep writer) plus four ADDITIVE patches whose added lines are writer
# calls, USE lines and comments:
#   stprk3     frame 's' after CALL sbc (stprk3.f90:142): every surface field
#              the ocean consumes, unchanged until read (stress at
#              stp2d.f90:198-199, dynzdf.f90:338,:506; emp/rnf at
#              stp2d.f90:242-245; qns/sfx/emp/rnf_tsc in tra_sbc_RK3; qsr in
#              tra_qsr), the restoring term erp/qrp and the surface means;
#   dynspg_ts  frame 'c' after the copies the solver reads
#              (dynspg_ts.f90:274-276; the sources are DEALLOCATEd at
#              stp2d.f90:270, so they exist only inside dyn_spg_ts);
#   sbcblk     frame 'k' after the ocean stress (sbcblk.f90:788-803);
#   sbcblk_algo_ncar  frame 'n' at turb_ncar entry and one frame per
#              iteration ('A'..'E', nn_iter_algo = 5) holding 1/L and the
#              clamped zeta (:174-178), psi_m and the floored U_N10 (:195-207),
#              the coefficients and the stability flag (:209-228).
# The bulk runs on MOD(kt-1, nn_fsbc) == 0 (sbcblk.f90:554): kt = 1, 3, ...
#
# REUSED, NOT RE-RUN: the round-17 reference (100-step deck, reference binary
# R4REF) is NEMO's boundary: it stopped at kt = 11 in stp_ctl, so the record
# is kt = 1..10 exactly as round 17 (record_steps below, same rule).  Twins
# as G3/G4: a (restart STEPS), b (even restarts), c (odd restarts).
# PASSIVITY: every round-17 field (step frames kt = 1..STEPS, substep
# streams at kt = 1, 2, every restart) must be byte-identical in this run.
set -Eeuo pipefail

refuse_unhandled() {
  local status=$? line=${BASH_LINENO[0]:-${LINENO}}
  trap - ERR
  printf 'REFUSE: unhandled command failure at run_r19.sh line %s (exit %s)\n' "$line" "$status" >&2
  exit "$status"
}
trap refuse_unhandled ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly P5_NAME=ORCA2G1_OMIP_L1_P5 RUN_NAME=ORCA2G1_OMIP_L1_P6
readonly WANT_KEYS='key_RK3 key_qco key_vco_1d3d'
readonly BULK_ITERATIONS=5 FSBC=2                     # namelist_ref:241, deck :nn_fsbc (checked)
readonly RECORD_SPGTS=1,2
readonly R17=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round17/oracle_g4_r17
readonly R17_ADMISSION_SHA=6d05567bc024e364e7b9c540f45210b6b5ce62a12a5829eab4bf4dc5f6d3e368
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round19/oracle_g4_r19}
readonly SHIPPED_DOMCFG=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/ORCA_R2_zps_domcfg.nc
readonly INPUTS=$(dirname "$SHIPPED_DOMCFG")
readonly G4_DECK_SHA=3a45cc6cd3bbe776424c47ba02700a08b8ac576162307de5819d8e539a989353
# the files round 17 pinned by SHA-256 (run.sh PINNED_FILES); here each must
# be byte-equal to round 17's admitted copy
readonly INPUT_FILES=(data_1m_potential_temperature_nomask.nc data_1m_salinity_nomask.nc
  u_10.15JUNE2009_fill.nc v_10.15JUNE2009_fill.nc ncar_rad.15JUNE2009_fill.nc t_10.15JUNE2009_fill.nc
  q_10.15JUNE2009_fill.nc ncar_precip.15JUNE2009_fill.nc slp.15JUNE2009_fill.nc
  weights_core2_orca2_bicub.nc weights_core2_orca2_bilin.nc chlorophyll.nc sss_data.nc
  runoff_core_monthly.nc domain_cfg.nc namelist_ref)

mode=--preflight-only
case "${1:---preflight-only}" in
  --run|--preflight-only|--admit-existing) mode=${1:---preflight-only} ;;
  *) printf 'Usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel 2>/dev/null) \
  || { printf 'REFUSE: run_r19.sh is not inside a git worktree\n' >&2; exit 66; }
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] \
  || { printf 'REFUSE: legoESM worktree is dirty; commit the exact acquisition tool first\n' >&2; exit 66; }
readonly COMMIT=$(git -C "$repo_root" rev-parse HEAD)

readonly DECK=$here/namelist_cfg
readonly G1_DIR=$here/../nemo_testcase_g1_orca2grid
readonly CHECKER=$G1_DIR/check_records.py
readonly P5=$NEMO_ROOT/cfgs/$P5_NAME
# MY_SRC file -> (shipped source, patches in order); modules and user routines copied whole
readonly -A SHIPPED=(
  [stprk3.F90]=$NEMO_ROOT/src/OCE/stprk3.F90
  [dynspg_ts.F90]=$NEMO_ROOT/src/OCE/DYN/dynspg_ts.F90
  [restart.F90]=$NEMO_ROOT/src/OCE/IOM/restart.F90
  [sbcblk.F90]=$NEMO_ROOT/src/OCE/SBC/sbcblk.F90
  [sbcblk_algo_ncar.F90]=$NEMO_ROOT/src/OCE/SBC/sbcblk_algo_ncar.F90 )
readonly -A OLD_PATCH=(            # round 4 (the P5 build); empty = shipped as-is
  [stprk3.F90]=$G1_DIR/stprk3_g1_step_record_stage50.patch
  [dynspg_ts.F90]=$G1_DIR/dynspg_ts_g1_substep_record.patch
  [restart.F90]=$G1_DIR/restart_g1_usrdef_ssh.patch
  [sbcblk.F90]= [sbcblk_algo_ncar.F90]= )
readonly -A NEW_PATCH=(            # this round, additive writer calls only
  [stprk3.F90]=$here/stprk3_g4_forcing_record.patch
  [dynspg_ts.F90]=$here/dynspg_ts_g4_forcing_record.patch
  [restart.F90]=
  [sbcblk.F90]=$here/sbcblk_g4_forcing_record.patch
  [sbcblk_algo_ncar.F90]=$here/sbcblk_algo_ncar_g4_forcing_record.patch )
readonly WHOLE=("$G1_DIR/usrdef_istate.F90" "$G1_DIR/usrdef_sbc.F90" "$G1_DIR/g1_step_record.F90"
                "$G1_DIR/g1_spgts_record.F90")

for f in "$DECK" "$CHECKER" "${SHIPPED[@]}" "${WHOLE[@]}" "$R17/g4_round17_admission.json" \
         "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$NEMO_ROOT/makenemo" "$P5/BLD/bin/nemo.exe"; do
  [[ -f "$f" ]] || { printf 'REFUSE: missing input %s\n' "$f" >&2; exit 67; }
done
for f in "${OLD_PATCH[@]}" "${NEW_PATCH[@]}"; do
  [[ -z "$f" || -f "$f" ]] || { printf 'REFUSE: missing patch %s\n' "$f" >&2; exit 67; }
done
sha_of() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(sha_of "$DECK")" == "$G4_DECK_SHA" ]] \
  || { printf 'REFUSE: the G4 deck is not round 17s\n' >&2; exit 67; }
[[ "$(sha_of "$R17/g4_round17_admission.json")" == "$R17_ADMISSION_SHA" ]] \
  || { printf 'REFUSE: the round-17 admission changed\n' >&2; exit 67; }
# nn_iter_algo: the deck is silent; the namelist_ref every run copies (round
# 17's admitted one) says 5 -- and the record's iteration frames must agree
! grep -q 'nn_iter_algo' "$DECK" \
  && grep -Eq '^ *nn_iter_algo *= *'"$BULK_ITERATIONS"' ' "$R17/p3_a/namelist_ref" \
  && grep -Eq '^ *nn_fsbc *= *'"$FSBC"'( |$)' "$DECK" \
  || { printf 'REFUSE: nn_iter_algo / nn_fsbc do not resolve to %s / %s\n' "$BULK_ITERATIONS" "$FSBC" >&2; exit 68; }

# ---- the sources: old patches reproduce P5's MY_SRC; new patches add calls only
dry=$(mktemp -d)
trap 'rm -rf "$dry"' EXIT
mkdir -p "$dry/p5" "$dry/src"
for f in "${!SHIPPED[@]}"; do
  cp "${SHIPPED[$f]}" "$dry/p5/$f"
  [[ -z "${OLD_PATCH[$f]}" ]] || patch -s --fuzz=0 "$dry/p5/$f" <"${OLD_PATCH[$f]}"
  if [[ -f "$P5/MY_SRC/$f" ]]; then
    cmp -s "$dry/p5/$f" "$P5/MY_SRC/$f" \
      || { printf 'REFUSE: shipped %s + round-4 patch is not P5s MY_SRC copy\n' "$f" >&2; exit 68; }
  else
    [[ -z "${OLD_PATCH[$f]}" ]] || { printf 'REFUSE: P5 has no %s\n' "$f" >&2; exit 68; }
  fi
  cp "$dry/p5/$f" "$dry/src/$f"
  [[ -z "${NEW_PATCH[$f]}" ]] && continue
  patch -s --fuzz=0 "$dry/src/$f" <"${NEW_PATCH[$f]}"
  [[ "$(grep -c '^-[^-]' "${NEW_PATCH[$f]}" || true)" -eq 0 ]] \
    || { printf 'REFUSE: %s removes lines\n' "${NEW_PATCH[$f]}" >&2; exit 68; }
  [[ "$(grep '^+[^+]' "${NEW_PATCH[$f]}" \
        | grep -vcE '^\+ *(USE g1_step_record|USE sbcssr, ONLY : erp, qrp|!|CALL g1r_frame\(|CALL g1r_w2\()' || true)" -eq 0 ]] \
    || { printf 'REFUSE: %s adds something other than writer calls\n' "${NEW_PATCH[$f]}" >&2; exit 68; }
done
for f in "${WHOLE[@]}"; do
  [[ ! -f "$P5/MY_SRC/$(basename "$f")" ]] || cmp -s "$f" "$P5/MY_SRC/$(basename "$f")" \
    || { printf 'REFUSE: %s is not P5s copy\n' "$f" >&2; exit 68; }
  cp "$f" "$dry/src/"
done
# P5's MY_SRC is exactly these files (nothing else reaches the build)
[[ "$(ls "$P5/MY_SRC" | LC_ALL=C sort | xargs)" == \
   "$(ls "$dry/src" | grep -vE '^sbcblk' | LC_ALL=C sort | xargs)" ]] \
  || { printf 'REFUSE: P5s MY_SRC is not the round-4 file set\n' >&2; exit 68; }
python3 -I -m py_compile "$CHECKER"

render_deck() {        # $1 = target, $2 = nn_itend, $3 = nn_stocklist
  sed -E -e "s/^(   nn_itend    = ).*/\1    $2/" \
         -e "s/^(   nn_stock    = ).*/\1    $2/" \
         -e "s/^(   nn_stocklist = ).*/\1$3/" "$DECK" >"$1"
}
pad10() { local l=("$@"); while ((${#l[@]} < 10)); do l+=(0); done; local IFS=,; printf '%s' "${l[*]}"; }
list_a() { pad10 "$1"; }
list_b() { pad10 $(seq 2 2 "$1"); }
list_c() { (($1 >= 4)) && pad10 $(seq 3 2 $(($1 - 1))) || pad10 0; }

# ---- NEMO's boundary, from round 17's admitted reference (run.sh rule) -----
completed_steps() {
  local ref=$R17/ref_100step t
  t=$(tr -d ' \n' <"$ref/time.step")
  if grep -qx 'RUN_DONE' "$ref/run.user.log"; then printf '100'; return; fi
  grep -q 'stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s' "$ref/ocean.output" && [[ -f "$ref/output.abort.nc" ]] \
    || { printf 'REFUSE: the round-17 reference stopped at kt = %s outside stp_ctl\n' "$t" >&2; exit 73; }
  printf '%s' $((t - 1))
}
c=$(completed_steps)
STEPS=$(( c < 10 ? c : 10 )); STEPS=$(( STEPS - STEPS % FSBC ))
((STEPS >= 2)) || { printf 'REFUSE: no record of 2 steps exists\n' >&2; exit 73; }
readonly STEPS
for twin in a b c; do         # the round-17 twins this run must reproduce
  render_deck "$dry/deck_$twin" "$STEPS" "$(list_$twin "$STEPS")"
  cmp -s "$dry/deck_$twin" "$R17/p3_$twin/namelist_cfg" \
    || { printf 'REFUSE: twin %s deck is not round 17s\n' "$twin" >&2; exit 68; }
done
render_deck "$dry/deck_smoke" 2 "$(pad10 2)"
cmp -s "$dry/deck_smoke" "$R17/smoke_2step/namelist_cfg" \
  || { printf 'REFUSE: the smoke deck is not round 17s\n' >&2; exit 68; }
printf 'PREFLIGHT: P5 sources reproduced; 4 additive writer patches apply; record kt = 1..%s (round-17 reference completed %s); NCAR iterations %s, nn_fsbc %s; commit %s\n' \
  "$STEPS" "$c" "$BULK_ITERATIONS" "$FSBC" "$COMMIT"

readonly SMOKE=$EVIDENCE/smoke_2step TWIN_A=$EVIDENCE/p6_a TWIN_B=$EVIDENCE/p6_b TWIN_C=$EVIDENCE/p6_c
readonly CFG=$NEMO_ROOT/cfgs/$RUN_NAME BIN_SHA_FILE=$EVIDENCE/manifest/binary.sha256

# The P6 build is the committed sources and nothing else: MY_SRC equals what
# preflight produced, keys, every compiled file is P5's plus writer lines
# (same file set), and the binary is the one hashed after the build.
judge_build() {
  local f added
  [[ -x "$CFG/BLD/bin/nemo.exe" ]] || { printf 'REFUSE: %s has no binary\n' "$RUN_NAME" >&2; exit 70; }
  [[ "$(ls "$CFG/MY_SRC" | LC_ALL=C sort | xargs)" == "$(ls "$dry/src" | LC_ALL=C sort | xargs)" ]] \
    || { printf 'REFUSE: %s MY_SRC is not the committed file set\n' "$RUN_NAME" >&2; exit 70; }
  for f in "$dry/src/"*.F90; do
    cmp -s "$f" "$CFG/MY_SRC/$(basename "$f")" \
      || { printf 'REFUSE: %s MY_SRC/%s is not the committed source\n' "$RUN_NAME" "$(basename "$f")" >&2; exit 70; }
  done
  [[ "$(sed -E 's/.*fppkeys *//' "$CFG/cpp_$RUN_NAME.fcm" | xargs -n1 | LC_ALL=C sort | xargs)" == "$WANT_KEYS" ]] \
    || { printf 'REFUSE: %s keys are not "%s"\n' "$RUN_NAME" "$WANT_KEYS" >&2; exit 70; }
  # the compiled sources are P5's plus writer lines only
  [[ "$(cd "$P5/BLD/ppsrc/nemo" && ls *.f90 | xargs)" == "$(cd "$CFG/BLD/ppsrc/nemo" && ls *.f90 | xargs)" ]] \
    || { printf 'REFUSE: P6 compiles another file set than P5\n' >&2; exit 70; }
  for f in $(cd "$P5/BLD/ppsrc/nemo" && ls *.f90); do
    [[ -f "$CFG/BLD/ppsrc/nemo/$f" ]] || { printf 'REFUSE: P6 lacks %s\n' "$f" >&2; exit 70; }
    added=$(diff "$P5/BLD/ppsrc/nemo/$f" "$CFG/BLD/ppsrc/nemo/$f" | grep -E '^[<>]' || true)
    [[ -z "$(grep '^<' <<<"$added" || true)" && -z "$(grep '^>' <<<"$added" \
          | grep -vE '^> *(USE g1_step_record|USE sbcssr, ONLY : erp, qrp|!|CALL g1r_frame\(|CALL g1r_w2\()' || true)" ]] \
      || { printf 'REFUSE: compiled %s differs from P5 beyond writer lines\n' "$f" >&2; exit 70; }
  done
  [[ ! -f "$BIN_SHA_FILE" || "$(awk '{print $1}' "$BIN_SHA_FILE")" == "$(sha_of "$CFG/BLD/bin/nemo.exe")" ]] \
    || { printf 'REFUSE: the evidence binary hash is not the built P6 binary\n' >&2; exit 70; }
}

judge_dir() {          # $1 = dir, $2 = round-17 twin dir, $3 = steps, $4 = list
  local dir=$1 r=$2 f step
  for f in "${INPUT_FILES[@]}"; do
    cmp -s "$dir/$f" "$r/$f" || { printf 'REFUSE: %s/%s is not round 17s\n' "$dir" "$f" >&2; exit 73; }
  done
  cmp -s "$dir/namelist_cfg" "$r/namelist_cfg" \
    || { printf 'REFUSE: %s ran another deck than round 17s\n' "$dir" >&2; exit 73; }
  [[ "$(sha_of "$dir/nemo")" == "$(awk '{print $1}' "$BIN_SHA_FILE")" ]] \
    || { printf 'REFUSE: %s was not run with the P6 binary\n' "$dir" >&2; exit 73; }
  grep -qx 'RUN_DONE' "$dir/run.user.log" && [[ "$(tr -d ' \n' <"$dir/time.step")" == "$3" ]] \
    && ! grep -q 'E R R O R' "$dir/ocean.output" \
    || { printf 'REFUSE: %s is not a finished %s-step run\n' "$dir" "$3" >&2; exit 73; }
  for step in ${4//,/ }; do
    [[ "$step" == 0 || -f "$dir/ORCA2G1_$(printf '%08d' "$step")_restart.nc" ]] \
      || { printf 'REFUSE: %s has no restart at listed step %s\n' "$dir" "$step" >&2; exit 73; }
  done
}

admit() {
  local plant
  [[ -f "$BIN_SHA_FILE" ]] || { printf 'REFUSE: no P6 binary hash in the evidence\n' >&2; exit 73; }
  judge_build
  judge_dir "$SMOKE" "$R17/smoke_2step" 2 "$(pad10 2)"
  judge_dir "$TWIN_A" "$R17/p3_a" "$STEPS" "$(list_a "$STEPS")"
  judge_dir "$TWIN_B" "$R17/p3_b" "$STEPS" "$(list_b "$STEPS")"
  judge_dir "$TWIN_C" "$R17/p3_c" "$STEPS" "$(list_c "$STEPS")"
  for plant in truncation nonfinite missing-frame twin-ulp missing-step \
               forcing-missing-iteration forcing-nonfinite forcing-twin-inner forcing-missing-field; do
    if python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" --steps "$STEPS" \
         --stage-steps "$STEPS" --bulk-iterations "$BULK_ITERATIONS" --fsbc "$FSBC" \
         --plant "$plant" >"$EVIDENCE/manifest/plant_${plant}.log" 2>&1; then
      printf 'REFUSE: record plant %s stayed green\n' "$plant" >&2; exit 72
    fi
    grep -q 'STATUS PLANT-FIRED' "$EVIDENCE/manifest/plant_${plant}.log" \
      || { printf 'REFUSE: record plant %s did not fire\n' "$plant" >&2; exit 72; }
  done
  python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_B" --reference "$R17/ref_100step" \
    --steps "$STEPS" --stage-steps "$STEPS" --restart-steps "$STEPS" \
    --bulk-iterations "$BULK_ITERATIONS" --fsbc "$FSBC" \
    --substep-steps "$RECORD_SPGTS" --json "$EVIDENCE/g4_round19_forcing_admission.json"
  python3 -I "$CHECKER" --evidence "$TWIN_A" --twin "$TWIN_C" --steps "$STEPS" \
    --stage-steps "$STEPS" --bulk-iterations "$BULK_ITERATIONS" --fsbc "$FSBC" \
    --json "$EVIDENCE/g4_round19_twin_c.json"
  python3 -I "$CHECKER" --evidence "$SMOKE" --steps 2 --stage-steps 2 \
    --bulk-iterations "$BULK_ITERATIONS" --fsbc "$FSBC" --json "$EVIDENCE/g4_round19_smoke.json"
  # passivity: every round-17 word is reproduced
  python3 -I - "$G1_DIR" "$R17" "$EVIDENCE" "$STEPS" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import check_records as cr
r17, ev, steps = Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
halo = {}
try:
    for twin in "abc":
        old, new = r17 / f"p3_{twin}", ev / f"p6_{twin}"
        for step in range(1, steps + 1):
            _, before = cr.read_stream(cr.step_path(old, step).read_bytes(), f"r17 {twin} {step}")
            _, after = cr.read_stream(cr.step_path(new, step).read_bytes(), f"r19 {twin} {step}",
                                      allow_inner=True)
            for name, data in before.items():
                cr.require(name in after and data.tobytes() == after[name].tobytes(),
                           f"twin {twin} kt={step}: {name} moved against round 17")
        for path in sorted(old.glob("ORCA2G1_*_restart.nc")):
            step = int(path.name.split("_")[1])
            cr.restarts_identical(old, new, [step])
        # substep streams: equal on the consumed support (inner domain of full
        # planes, other arrays whole); halo storage NEMO never sets is counted
        for path in sorted(old.glob("oracle_g1spgts_kt*.bin")):
            header, before = cr.read_substeps(path.read_bytes(), f"r17 {twin} {path.name}")
            _, after = cr.read_substeps((new / path.name).read_bytes(), f"r19 {twin} {path.name}")
            halo[f"{twin}/{path.name}"] = cr.twin_halo_words(
                header, before, after, f"twin {twin}: {path.name} against round 17")
    cr.restarts_identical(r17 / "smoke_2step", ev / "smoke_2step", [2])
    try:    # plant: one moved inner-domain word must refuse
        cr.twin_halo_words(header, before, after, "plant", plant=True)
        sys.exit("REFUSE: the passivity plant stayed green")
    except cr.Refuse:
        pass
except cr.Refuse as exc:
    sys.exit(f"REFUSE: {exc}")
(ev / "g4_round19_passivity.json").write_text(json.dumps(
    {"substep_halo_words_differing": halo, "plant": "fired"}, indent=1) + "\n")
print(f"G4R19_PASSIVE every round-17 step frame kt=1..{steps} and restart of twins a, b, c "
      "and the smoke is byte-identical; substep streams identical on the consumed support "
      f"({sum(halo.values())} halo words differ, counted in g4_round19_passivity.json)")
PY
  ( cd "$EVIDENCE" && sha256sum g4_round19_*.json p6_a/oracle_*.bin p6_b/oracle_g1step_*.bin \
      p6_c/oracle_g1step_*.bin smoke_2step/oracle_g1step_*.bin p6_*/ORCA2G1_*_restart.nc \
      p6_a/ocean.output ) >"$EVIDENCE/manifest/outputs.sha256"
  printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
}

if [[ "$mode" == --admit-existing ]]; then
  for target in "$SMOKE" "$TWIN_A" "$TWIN_B" "$TWIN_C" "$BIN_SHA_FILE"; do
    [[ -e "$target" ]] || { printf 'REFUSE: nothing to admit, %s is absent\n' "$target" >&2; exit 69; }
  done
  admit
  exit 0
fi
[[ ! -e "$EVIDENCE" ]] || { printf 'REFUSE: %s already exists; use --admit-existing\n' "$EVIDENCE" >&2; exit 69; }
[[ ! -e "$CFG" ]] || { printf 'REFUSE: %s already exists (a NEW target name is required)\n' "$CFG" >&2; exit 69; }
if [[ "$mode" != --run ]]; then
  printf 'WOULD: makenemo -r ORCA2_ICE_PISCES -n %s -d OCE -m conda-scalarmath del_key "key_si3 key_top key_xios" with MY_SRC = %s\n' \
    "$RUN_NAME" "$(ls "$dry/src" | xargs)"
  printf 'WOULD: mpirun -np 1 ./nemo: smoke (2 steps), twins a, b, c (%s steps) in %s; then %s\n' \
    "$STEPS" "$EVIDENCE" "$CHECKER"
  printf 'PREFLIGHT_OK\n'
  exit 0
fi

mkdir -p "$EVIDENCE/manifest"
printf '%s\n' "$COMMIT" >"$EVIDENCE/manifest/legoesm_commit"
( sha256sum "$DECK" "$CHECKER" "${SHIPPED[@]}" "${WHOLE[@]}" "$NEMO_ROOT/cfgs/SHARED/namelist_ref"
  for f in "${OLD_PATCH[@]}" "${NEW_PATCH[@]}"; do [[ -z "$f" ]] || sha256sum "$f"; done ) \
  >"$EVIDENCE/manifest/inputs.sha256"

# ---- the record build -------------------------------------------------------
cd "$NEMO_ROOT"
./makenemo -r ORCA2_ICE_PISCES -n "$RUN_NAME" -d 'OCE' -m conda-scalarmath del_key 'key_si3 key_top key_xios'
mkdir -p "$CFG/MY_SRC"
cp "$dry/src/"*.F90 "$CFG/MY_SRC/"          # cp, NOT cp -a: fcm must see new mtimes
touch "$CFG/MY_SRC/"*.F90
./makenemo -n "$RUN_NAME" -m conda-scalarmath
[[ -x "$CFG/BLD/bin/nemo.exe" ]] || { printf 'REFUSE: %s did not build\n' "$RUN_NAME" >&2; exit 70; }
judge_build
sha256sum "$CFG/BLD/bin/nemo.exe" >"$BIN_SHA_FILE"

run_one() {            # $1 = run dir, $2 = round-17 dir, $3 = nn_itend, $4 = list
  local dir=$1 f rc=0
  mkdir -p "$dir"
  render_deck "$dir/namelist_cfg" "$3" "$4"
  for f in "${INPUT_FILES[@]}"; do cp "$2/$f" "$dir/$f"; done   # round 17's admitted copies
  cp "$CFG/BLD/bin/nemo.exe" "$dir/nemo"
  (
    cd "$dir"
    printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.log
    mpirun -np 1 --oversubscribe ./nemo >>run.user.log 2>&1 || rc=$?
    printf 'RUN_FINISHED_UTC=%s\nRUN_EXIT=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>run.user.log
    [[ "$rc" -ne 0 ]] || printf 'RUN_DONE\n' >>run.user.log
  )
  grep -qx 'RUN_DONE' "$dir/run.user.log" && [[ "$(tr -d ' \n' <"$dir/time.step")" == "$3" ]] \
    || { printf 'REFUSE: %s did not finish %s steps (see ocean.output)\n' "$dir" "$3" >&2; exit 71; }
}

# ---- smoke first, then the twins -------------------------------------------
run_one "$SMOKE" "$R17/smoke_2step" 2 "$(pad10 2)"
python3 -I "$CHECKER" --evidence "$SMOKE" --steps 2 --stage-steps 2 \
  --bulk-iterations "$BULK_ITERATIONS" --fsbc "$FSBC" \
  || { printf 'REFUSE: the smoke record is not admitted; twins not run\n' >&2; exit 71; }
printf 'G4R19_SMOKE_OK %s\n' "$SMOKE"
run_one "$TWIN_A" "$R17/p3_a" "$STEPS" "$(list_a "$STEPS")"
run_one "$TWIN_B" "$R17/p3_b" "$STEPS" "$(list_b "$STEPS")"
run_one "$TWIN_C" "$R17/p3_c" "$STEPS" "$(list_c "$STEPS")"
admit
