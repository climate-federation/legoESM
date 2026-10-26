#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED acquisition only.  The agent must not run NEMO, makenemo or
# mpirun from its sandbox; it wrote this file and stopped.
#
#   Usage:  run.sh
#
# WHAT THIS ACQUIRES, AND WHY.  Round 38 measured that the ENTIRE difference
# between legoESM's tracer matrix diffusivity and NEMO's is the ISONEUTRAL
# FOLD: NEMO's ah_wslp2 is IDENTICALLY 0.0 in the round-37 record, legoESM's
# K33 reaches 9.6620886711883531e-13 m2/s, and substituting NEMO's own zwt_mix
# at legoESM's own call site collapses the kt=1 stage-3 T residual from
# 3.1956659540810506e-11 K to 1.4210854715202004e-14 K -- 99.96 per cent of it.
#
# The owner is the CALL SITE.  NEMO computes the neutral slopes ONCE PER STEP,
# on the BEFORE state, OUTSIDE the stage loop (stprk3.F90:178,
# `CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )`), and traldf_iso.F90:154 fills
# ah_wslp2 from them for every stage; legoESM recomputes its K33 INSIDE each
# stage from that stage's own tracers.  At kt = nit000 GYRE has ln_rstart = F
# and its analytic initial T and S are horizontally UNIFORM on wet cells --
# measured, per-level range exactly 0.0 on both, on NEMO's dumped Kbb and on
# legoESM's own initial state -- so NEMO's slopes are exactly zero and
# ah_wslp2 is identically zero at every stage of step 1.
#
# THAT IS THE RECORD GAP.  A record whose isoneutral term is identically zero
# cannot discriminate ANY transcription of it: it can only say whether the
# candidate's fold is also exactly zero.  The moment legoESM's K33 is moved to
# NEMO's placement -- a scientific choice that is ASKED, not taken here --
# there is no oracle on this branch to verify it against.
#
# SO THIS WIDENS THE EXISTING ROUND-35/37 INSTRUMENT FROM ONE STEP TO TWO.
# `ll_l2_tra = ( lwp .AND. kt <= nit000 + 1 )` and a per-kt file name; nothing
# else changes.  kt = nit000+1 is the first step whose BEFORE state carries the
# horizontal structure step 1 created, so it is the first record in which
# NEMO's own ah_wslp2 can be non-zero.
#
# THE STRONGEST CHECK THIS ACQUISITION HAS is that the kt = nit000 record must
# come back BYTE-IDENTICAL to round 37's.  Widening an arm must not move step
# one; if it does, the delta is not WRITE-only and the run is refused.
#
# THE INVARIANT IS CHECKED ON THE FINAL FILE, NOT ON THE PATCH.  This delta
# removes three of the INSTRUMENT's own lines (the arming predicate, the
# hardcoded file name and the reset comment); it removes NONE of NEMO's.  The
# check below is the one that matters: the built MY_SRC/trazdf.F90 must differ
# from NEMO's shipped trazdf.F90 by ADDITIONS ONLY, zero removed lines.
# Measured before this file was written: 0.
#
# THE ARM is unchanged from round 35 and is re-read off this run's own
# resolved namelist by the gate, which REFUSES a record from any other one:
# ln_zdfddm F, ln_zad_Aimp F, ln_zdfmfc F, ln_traldf_msc F, l_ldfslp TRUE
# (rotated laplacian, ln_traldf_iso T at ocean.output:655), ln_SEOS F so the
# DRAKKAR clamp RUNS.
#
# THE PATH MATTERS.  FCM's extract step parses Fortran with perl's
# Text::Balanced, which the system perl on this machine does not have;
# makenemo then fails with "Can't locate Text/Balanced.pm" long before it
# compiles anything.  The line below puts the conda build environment first,
# which is the PATH the round-33, round-35 and round-37 builds actually used.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R37TRAZDF
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
# The round-35 run: the namelists come from it and its record is the twin the
# new one is compared against, field by field, through the gate's own reader.
readonly SOURCE_RUN=$L2/round37_oracle_trazdf_matrix
# The ADMISSION baseline is round 29, exactly as round 35 used it, and NOT the
# round-35 run -- the admission cannot parse the round-35 trazdf record, which
# is the defect this acquisition removes.  Using round 29 keeps the admission
# comparing the 50 records both runs share.
readonly ADMISSION_BASE=$L2/round29_oracle_v2_zdf_matrix
readonly TARGET_RUN=$L2/round38_oracle_trazdf_kt2
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly KT1_RECORD=oracle_trazdf_matrix_kt00000001.bin
readonly NEW_RECORD=oracle_trazdf_matrix_kt00000002.bin

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly TRA_PATCH=$here/trazdf_round38_kt2.patch
# The gates import legoesm from THIS worktree.  Round 35 left the interpreter
# to find it, which works only if the operator's shell already has it; naming
# it here makes the acquisition self-contained and pins the same CPU/fp64
# regime every measurement in this campaign is taken at.
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
[[ -d "$REPO/packages/ocean" && -d "$REPO/src" ]]
export PYTHONPATH=$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
readonly GATE=$here/../nemo_testcase_l2_gyre_round35_trazdf_matrix.py
readonly R29_GATE=$here/../nemo_testcase_l2_gyre_round29_zdf_matrix.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly SHIPPED=$NEMO_ROOT/src/OCE/TRA/trazdf.F90

source_cfg=$NEMO_ROOT/cfgs/$SOURCE_CFG
target_cfg=$NEMO_ROOT/cfgs/$TARGET_CFG

[[ -d "$source_cfg/MY_SRC" && -d "$source_cfg/EXP00" ]]
[[ -f "$TRA_PATCH" && -f "$SHIPPED" ]]
[[ -f "$GATE" && -f "$R29_GATE" && -f "$ADMISSION" ]]
[[ -d "$SOURCE_RUN" && -d "$ADMISSION_BASE" ]]
if [[ -e "$target_cfg" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: target already exists: %s or %s\n' "$target_cfg" "$TARGET_RUN" >&2
  exit 64
fi
# The premise this delta rests on: the source card DOES carry the round-35
# instrument, and does NOT already carry the fix.
if [[ ! -e "$source_cfg/MY_SRC/trazdf.F90" ]]; then
  printf 'REFUSE: %s has no MY_SRC/trazdf.F90 to patch\n' "$SOURCE_CFG" >&2
  exit 66
fi
if ! grep -q 'NEMO_L2_TRAZD_1' "$source_cfg/MY_SRC/trazdf.F90"; then
  printf 'REFUSE: %s MY_SRC/trazdf.F90 is not the round-35 instrument\n' "$SOURCE_CFG" >&2
  exit 66
fi
if ! grep -q 'zl2_tmp(ntsi:ntei,ntsj:ntej,:) = avt' "$source_cfg/MY_SRC/trazdf.F90"; then
  printf 'REFUSE: %s MY_SRC/trazdf.F90 does not carry the round-37 avt staging\n' \
    "$SOURCE_CFG" >&2
  exit 66
fi
if grep -q 'kt <= nit000 + 1' "$source_cfg/MY_SRC/trazdf.F90"; then
  printf 'REFUSE: %s already writes two steps; nothing to acquire\n' \
    "$SOURCE_CFG" >&2
  exit 66
fi
if ! grep -q 'NEMO_L2_ZDFMX_1' "$source_cfg/MY_SRC/dynzdf.F90"; then
  printf 'REFUSE: %s MY_SRC/dynzdf.F90 is not the round-29 instrument\n' "$SOURCE_CFG" >&2
  exit 66
fi
dry=$(mktemp -d /tmp/gyre-r38-dryrun.XXXXXX)
cp "$source_cfg/MY_SRC/trazdf.F90" "$dry/trazdf.F90"
if ! patch -s "$dry/trazdf.F90" <"$TRA_PATCH" >/dev/null 2>&1; then
  printf 'REFUSE: the round-38 delta does not apply cleanly\n' >&2
  rm -rf "$dry"; exit 67
fi
# THE WRITE-ONLY INVARIANT, on the RESULT: additions only against NEMO's own
# shipped trazdf.F90.  `diff | grep -c` cannot inherit diff's exit status here
# because the count is what is tested, not the pipeline's status.
removed=$(diff "$SHIPPED" "$dry/trazdf.F90" | grep -c '^<' || true)
if [[ "$removed" -ne 0 ]]; then
  printf 'REFUSE: the patched instrument removes %s line(s) of the shipped trazdf.F90\n' \
    "$removed" >&2
  rm -rf "$dry"; exit 67
fi
rm -rf "$dry"
# Space.  The record is ~30 full-domain 3-D arrays plus the build tree.
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 2097152 ]]; then
    printf 'REFUSE: %s has %s kB free, under the 2 GB floor\n' "$mount" "$free_kb" >&2
    exit 68
  fi
done

work_manifest=$(mktemp -d /tmp/gyre-r38-source.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$work_manifest"
(
  cd "$source_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$SHIPPED" "$TRA_PATCH" \
  >"$work_manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath
# cp -r, NOT cp -a: preserved mtimes let fcm decide a patched file is already
# built.  MY_SRC is touched below for the same reason.
cp -r "$source_cfg/EXP00/." "$target_cfg/EXP00/"
cp -r "$source_cfg/MY_SRC/." "$target_cfg/MY_SRC/"
cp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
(
  cd "$target_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/copied_cfg_before_patch.sha256"
cmp "$work_manifest/source_cfg.sha256" "$work_manifest/copied_cfg_before_patch.sha256"
cmp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"

patch "$target_cfg/MY_SRC/trazdf.F90" <"$TRA_PATCH"
touch "$target_cfg/MY_SRC/"*.F90
sha256sum "$target_cfg/MY_SRC/trazdf.F90" "$target_cfg/MY_SRC/dynzdf.F90" \
  "$TRA_PATCH" >"$work_manifest/round38_instrument.sha256"

./makenemo -n "$TARGET_CFG" -m conda-scalarmath
target_binary=$target_cfg/BLD/bin/nemo.exe
[[ -x "$target_binary" ]]
# The build must have CARRIED the fix and every writer the card already had.
grep -q 'kt <= nit000 + 1' "$target_cfg/BLD/ppsrc/nemo/trazdf.f90" \
  || { printf 'REFUSE: the round-38 kt widening is absent from the compiled ppsrc (stale build)\n' >&2; exit 69; }
grep -q "oracle_trazdf_matrix_kt', kt" "$target_cfg/BLD/ppsrc/nemo/trazdf.f90" \
  || { printf 'REFUSE: the round-38 per-kt file name is absent from the compiled ppsrc (stale build)\n' >&2; exit 69; }
grep -q 'NEMO_L2_TRAZD_1' "$target_cfg/BLD/ppsrc/nemo/trazdf.f90" \
  || { printf 'REFUSE: the tra_zdf writer is absent from the compiled ppsrc\n' >&2; exit 69; }
grep -q 'NEMO_L2_ZDFMX_1' "$target_cfg/BLD/ppsrc/nemo/dynzdf.f90" \
  || { printf 'REFUSE: the round-29 dyn_zdf writer is absent from the compiled ppsrc\n' >&2; exit 69; }
grep -q 'r1_hu_0         ' "$target_cfg/BLD/ppsrc/nemo/dynzdf.f90" \
  || { printf 'REFUSE: the round-33 reference-geometry writer is absent from the compiled ppsrc\n' >&2; exit 69; }
for writer in stprk3 stprk3_stg; do
  grep -q 'oracle_' "$target_cfg/BLD/ppsrc/nemo/$writer.f90" \
    || { printf 'REFUSE: MY_SRC writer %s not compiled (stale build)\n' "$writer" >&2; exit 69; }
done
if nm -D "$target_binary" | grep -q '_ZGV'; then
  printf 'REFUSE: vector-math symbol present in %s\n' "$target_binary" >&2
  exit 65
fi
sha256sum "$target_binary" >"$work_manifest/binary.sha256"

mkdir "$TARGET_RUN"
readonly PREPARED="namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref \
namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
field_def_nemo-oce.xml field_def_nemo-pisces.xml"
for name in $PREPARED; do
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
cp "$target_binary" "$TARGET_RUN/nemo"
cp "$work_manifest"/*.sha256 "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  # Compare the COPIES against the ADMISSION BASELINE's originals, not against
  # the directory they were copied from: that is the comparison that can fail,
  # and it proves rounds 29, 35 and 37 are one namelist.
  for name in $PREPARED; do
    cmp "$ADMISSION_BASE/$name" "$name"
  done
  sha256sum $PREPARED >prepared_files.sha256
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export PATH=/home/dbalwada/miniconda3/envs/nemo-build/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.time.log
  TIMEFORMAT='wall_seconds %R\nuser_seconds %U\nsys_seconds %S'
  { time mpirun -np 1 --oversubscribe ./nemo 2>&1 | tee run.user.stdout.log ; } \
    2>>run.user.time.log
  test "${PIPESTATUS[0]}" -eq 0
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>run.user.time.log
)

# TWIN IDENTITY.  RAW byte identity against the ROUND-37 run is REPORTED for
# every record and REQUIRED for one -- the kt = nit000 trazdf record this arm
# reads.  Everything else is decided by the CONSUMED-FIELD admission below.
#
# ROUND 39: THE BLANKET RAW RULE WAS WRONG AND IT REFUSED A GOOD RUN.  NEMO's
# stream dumps write whole work arrays INCLUDING the nn_hls halo, which NEMO
# neither owns nor initialises, and one record carries a zFw slot the executed
# branch has not defined at the write point.  Those bytes are uninitialised
# memory: they differ between two runs of the SAME executable, so raw identity
# is not attainable and a rule that demands it refuses runs that changed no
# model state.  That is precisely why this campaign has a consumed-field
# admission gate, and this script already ran it four lines further down --
# the log line printed here even said "falls through to the consumed-field
# admission" while the next block refused on it.  Measured on this
# acquisition: 8 of 52 records differ by 4-52 bytes, EVERY differing element
# is halo or the registered undefined zFw slot, every value is subnormal
# garbage of order 1e-310, and the restart and mesh_mask are byte-identical.
: >"$TARGET_RUN/round38_raw_twin_cmp.log"
for source_record in "$SOURCE_RUN"/oracle_*.bin; do
  record=${source_record##*/}
  if [[ ! -e "$TARGET_RUN/$record" ]]; then
    printf 'REFUSE: the instrumented run did not write %s\n' "$record" >&2
    exit 71
  fi
  if cmp -s "$source_record" "$TARGET_RUN/$record"; then
    printf 'RAW_IDENTICAL %s\n' "$record"
  else
    printf 'RAW_DIFFERS   %s (falls through to the consumed-field admission)\n' \
      "$record"
  fi
done | tee -a "$TARGET_RUN/round38_raw_twin_cmp.log"
# THE ONE RECORD THAT MUST BE RAW-IDENTICAL is the kt = nit000 trazdf record:
# it is the record every round-35/37/38 arm scores against, so a single moved
# bit in it would invalidate them, and it carries no undefined slot.
if ! grep -q "^RAW_IDENTICAL $KT1_RECORD\$" \
     "$TARGET_RUN/round38_raw_twin_cmp.log"; then
  printf 'REFUSE: widening the arm moved the kt=1 trazdf record\n' >&2
  exit 71
fi
# EVERY OTHER round-37 record must come back with identical CONSUMED FIELDS --
# same instrument, same bar as the round-29 admission below, but against the
# run this delta was derived from.  An owned cell of a defined field that
# moved is a refusal; a halo or undefined-slot byte is admitted and PRINTED.
python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$NEW_RECORD" \
  --output "$TARGET_RUN/round38_source_admission.json" \
  || { printf 'REFUSE: a round-37 consumed field moved\n' >&2; exit 71; }
if python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$NEW_RECORD" --plant-consumed \
     >"$TARGET_RUN/round38_source_admission_plant.json" 2>&1; then
  printf 'REFUSE: the source-admission plant did not turn the gate red\n' >&2
  exit 72
fi
# AND IT MUST HAVE LANDED.  The gate also exits non-zero when the plant was
# never applied ("the requested plant was never applied"), so "exited non-zero"
# alone is satisfied by a control that proved nothing -- the same defeat round
# 38 removed from the operand gate's own plant, found here by an independent
# diff review.
grep -q '"plant_applied": true' "$TARGET_RUN/round38_source_admission_plant.json" \
  || { printf 'REFUSE: the source-admission plant never landed\n' >&2; exit 72; }
# And the whole point of the acquisition: the SECOND record must exist and
# must NOT be a copy of the first.
[[ -s "$TARGET_RUN/$NEW_RECORD" ]] \
  || { printf 'REFUSE: the run wrote no kt=2 record\n' >&2; exit 71; }
if cmp -s "$TARGET_RUN/$KT1_RECORD" "$TARGET_RUN/$NEW_RECORD"; then
  printf 'REFUSE: the kt=2 record is byte-identical to the kt=1 record\n' >&2
  exit 71
fi

# THE RECORD THIS ACQUISITION EXISTS FOR.  ah_wslp2 is identically 0.0 at
# kt = nit000, which is what makes that record unable to discriminate any
# isoneutral transcription.  If it is ALSO identically zero at kt = nit000+1
# the acquisition has not closed the gap, and that is a FINDING to report, not
# a silent pass -- so it is measured and printed here rather than assumed.
python - "$GATE" "$TARGET_RUN/$KT1_RECORD" "$TARGET_RUN/$NEW_RECORD" \
  >"$TARGET_RUN/round38_record_twin.txt" <<'PYTWIN'
import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("r35gate", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
one = module.read_trazdf_matrix(module.Path(sys.argv[2]))
two = module.read_trazdf_matrix(module.Path(sys.argv[3]))
fail = []
for label, rec in (("kt1", one), ("kt2", two)):
    if rec["tile_shaped_salvage"]:
        fail.append(f"{label} still needs the interior-extent salvage")
    if rec["order"] != one["order"]:
        fail.append(f"{label} array order moved")
if one["header"]["kt"] != two["header"]["kt"] - 1:
    fail.append(f"kt {one['header']['kt']} and {two['header']['kt']} are not consecutive")
for name in ("ah_wslp2", "akz", "avt", "zwt_mix"):
    for label, rec in (("kt1", one), ("kt2", two)):
        block = module._box(rec, name)
        print(f"{label.upper()}_ABSMAX {name} {float(np.abs(block).max()):.17g}")
a33 = float(np.abs(module._box(two, "ah_wslp2")).max())
print(f"KT2_AH_WSLP2_NONZERO {a33 > 0.0}")
if a33 == 0.0:
    print("FINDING the kt=2 record's ah_wslp2 is ALSO identically zero; this "
          "acquisition did not close the discrimination gap and a later step "
          "is needed")
for line in fail:
    print(f"TWIN_FAIL {line}")
print("TWIN_VERDICT", "PASS" if not fail else "FAIL")
sys.exit(0 if not fail else 1)
PYTWIN
tail -4 "$TARGET_RUN/round38_record_twin.txt"

python "$ADMISSION" --baseline "$ADMISSION_BASE" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$NEW_RECORD" \
  --output "$TARGET_RUN/round38_admission.json"
if python "$ADMISSION" --baseline "$ADMISSION_BASE" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$NEW_RECORD" --plant-consumed \
     >"$TARGET_RUN/round38_admission_plant.json" 2>&1; then
  printf 'REFUSE: the admission plant did not turn the gate red\n' >&2
  exit 72
fi
# AND IT MUST HAVE LANDED.  The gate also exits non-zero when the plant was
# never applied ("the requested plant was never applied"), so "exited non-zero"
# alone is satisfied by a control that proved nothing -- the same defeat round
# 38 removed from the operand gate's own plant, found here by an independent
# diff review.
grep -q '"plant_applied": true' "$TARGET_RUN/round38_admission_plant.json" \
  || { printf 'REFUSE: the admission plant never landed\n' >&2; exit 72; }
test -s "$TARGET_RUN/$NEW_RECORD"

# The round-29 gate on the momentum record: unchanged from round 35, so every
# row it scored must still be AT-BAR.
python "$R29_GATE" --record "$TARGET_RUN/oracle_zdf_matrix_kt00000001.bin" \
  --json "$TARGET_RUN/round38_r29_regression.json"

# The round-35 gate itself, and one plant per arm.  THE GATE'S VERDICT IS THE
# MEASUREMENT, NOT AN ACQUISITION CHECK: it exits non-zero on any row over the
# exact bar, which is correct and must never be relaxed -- but a DEBT row is a
# FINDING, not a reason to abandon the run before its plants have proved the
# gate can see anything.  So the exit code is captured and reported.
set +e
python "$GATE" --record "$TARGET_RUN/$KT1_RECORD" \
  --json "$TARGET_RUN/round38_trazdf_gate_kt1.json"
gate_status_kt1=$?
python "$GATE" --record "$TARGET_RUN/$NEW_RECORD" \
  --json "$TARGET_RUN/round38_trazdf_gate.json"
gate_status=$?
set -e
grep -q '"format": "nemo-testcase-l2-gyre-round35-trazdf-matrix-v1"' \
  "$TARGET_RUN/round38_trazdf_gate.json" \
  || { printf 'REFUSE: the gate produced no report; it crashed rather than judging\n' >&2; exit 73; }
arms=$(python - "$GATE" <<'PYARMS'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("r35gate", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(" ".join(module.PLANT_ARMS))
PYARMS
)
[[ -n "$arms" ]]
for arm in $arms; do
  out=$TARGET_RUN/round38_trazdf_plant_$arm.json
  if python "$GATE" --record "$TARGET_RUN/$NEW_RECORD" --plant "$arm" >"$out" 2>&1; then
    printf 'REFUSE: the %s plant did not turn the gate red\n' "$arm" >&2
    exit 70
  fi
  # A non-zero exit is NOT enough, twice over.  A plant that CRASHED the
  # reader exits non-zero and proves nothing; and this gate's BASELINE is
  # already red, so a plant that moved nothing would exit non-zero too.
  if ! grep -q '^STATUS DEBT' "$out"; then
    printf 'REFUSE: the %s plant exited non-zero without a DEBT verdict; it crashed rather than landing\n' "$arm" >&2
    exit 70
  fi
  if ! grep -q "^PLANT $arm landed=True" "$out"; then
    printf 'REFUSE: the %s plant moved no row; that control proves nothing\n' "$arm" >&2
    exit 70
  fi
done
(
  cd "$TARGET_RUN"
  sha256sum oracle_*.bin round38_admission.json round38_trazdf_gate.json \
    round38_trazdf_gate_kt1.json \
    round38_r29_regression.json round38_record_twin.txt \
    "$FINAL_RESTART" mesh_mask.nc >round38_outputs.sha256
)
printf 'ROUND38_GATE_VERDICT_KT1 exit=%s\n' "$gate_status_kt1"
printf 'ROUND38_GATE_VERDICT_KT2 exit=%s (0 = every row at bar; 1 = at least one row is DEBT, which is a finding)\n' "$gate_status"
printf 'ROUND38_GYRE_TRAZDF_KT2_ORACLE_READY %s\n' "$TARGET_RUN"
