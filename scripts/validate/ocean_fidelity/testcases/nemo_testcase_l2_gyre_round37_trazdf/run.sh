#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED acquisition only.  The agent must not run NEMO, makenemo or
# mpirun from its sandbox; it wrote this file and stopped.
#
#   Usage:  run.sh
#
# WHAT THIS ACQUIRES, AND WHY.  Round 36 found the round-35 record MISLABELLED,
# and NEMO's own ALLOCATE says why.  zdf_oce.F90 allocates
#
#    avs(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk)
#    avt(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk)
#
# over the INTERIOR box, while avm -- on the SAME ALLOCATE statement -- is
# (jpi,jpj,jpk).  The round-35 instrument wrote `WRITE(il2_unit) avt` under a
# header declaring jpi,jpj,jpk, so each of those two payloads is 32*22*31
# values where its header claims 36*26*31: 57536 bytes short each, and every
# array written after avs lands at the wrong offset.  Round 36's READER
# recovers them where the byte stream proves it, and that recovery is not the
# fix: the record should say what it is.  Two arrays, staged into the
# full-domain scratch every neighbouring array already uses.
#
# WHY THIS MATTERS BEYOND TIDINESS.  The shared admission gate cannot read the
# round-35 record at all -- it has no interior-extent recovery and raises
# `array '...' rank 1896932436`.  That is harmless only while the round-35 run
# is nobody's BASELINE.  It becomes a hard blocker the moment a later round
# inherits it, so the instrument is fixed at the source rather than every
# reader taught the salvage.
#
# ONE DELTA PATCH, ON THE ROUND-35 CARD'S OWN MY_SRC.
#
#   trazdf_round37_avt_extent.patch replaces the two raw WRITEs by the staging
#   idiom used four lines below them for ah_wslp2 and akz.  ln_tile is F on
#   this card (ocean.output "Tiling (T) or not (F)  ln_tile = F"), so
#   ntsi:ntei,ntsj:ntej IS Nis0:Nie0,Njs0:Nje0 and the copy conforms exactly.
#   The staged halo ring is the scratch's zeros: avt has no halo to carry, and
#   the gate's scored box is the interior.
#
# THE INVARIANT IS CHECKED ON THE FINAL FILE, NOT ON THE PATCH.  Round 35's
# guard counted removed lines in the patch, which is right for a patch against
# the SHIPPED source and wrong for a delta against an already-instrumented
# copy: this delta legitimately removes two of the INSTRUMENT's own lines.  So
# the check below is the stronger one it was standing in for -- the built
# MY_SRC/trazdf.F90 must differ from NEMO's shipped trazdf.F90 by ADDITIONS
# ONLY, zero removed lines.  A WRITE-only instrument that deletes a shipped
# line is not WRITE-only.
#
# THE ARM is unchanged from round 35 and is re-read off this run's own
# resolved namelist by the gate, which REFUSES a record from any other one:
# ln_zdfddm F, ln_zad_Aimp F, ln_zdfmfc F, ln_traldf_msc F, l_ldfslp TRUE
# (rotated laplacian), ln_SEOS F so the DRAKKAR clamp RUNS.
#
# THE PATH MATTERS.  FCM's extract step parses Fortran with perl's
# Text::Balanced, which the system perl on this machine does not have;
# makenemo then fails with "Can't locate Text/Balanced.pm" long before it
# compiles anything.  The line below puts the conda build environment first,
# which is the PATH the round-33 and round-35 builds actually used.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R35TRAZDF
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R37TRAZDF
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
# The round-35 run: the namelists come from it and its record is the twin the
# new one is compared against, field by field, through the gate's own reader.
readonly SOURCE_RUN=$L2/round35_oracle_trazdf_matrix
# The ADMISSION baseline is round 29, exactly as round 35 used it, and NOT the
# round-35 run -- the admission cannot parse the round-35 trazdf record, which
# is the defect this acquisition removes.  Using round 29 keeps the admission
# comparing the 50 records both runs share.
readonly ADMISSION_BASE=$L2/round29_oracle_v2_zdf_matrix
readonly TARGET_RUN=$L2/round37_oracle_trazdf_matrix
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly NEW_RECORD=oracle_trazdf_matrix_kt00000001.bin

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly TRA_PATCH=$here/trazdf_round37_avt_extent.patch
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
if grep -q 'zl2_tmp(ntsi:ntei,ntsj:ntej,:) = avt' "$source_cfg/MY_SRC/trazdf.F90"; then
  printf 'REFUSE: %s already carries the round-37 staging; nothing to acquire\n' \
    "$SOURCE_CFG" >&2
  exit 66
fi
if ! grep -q 'NEMO_L2_ZDFMX_1' "$source_cfg/MY_SRC/dynzdf.F90"; then
  printf 'REFUSE: %s MY_SRC/dynzdf.F90 is not the round-29 instrument\n' "$SOURCE_CFG" >&2
  exit 66
fi
dry=$(mktemp -d /tmp/gyre-r37-dryrun.XXXXXX)
cp "$source_cfg/MY_SRC/trazdf.F90" "$dry/trazdf.F90"
if ! patch -s "$dry/trazdf.F90" <"$TRA_PATCH" >/dev/null 2>&1; then
  printf 'REFUSE: the round-37 delta does not apply cleanly\n' >&2
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

work_manifest=$(mktemp -d /tmp/gyre-r37-source.XXXXXX)
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
  "$TRA_PATCH" >"$work_manifest/round37_instrument.sha256"

./makenemo -n "$TARGET_CFG" -m conda-scalarmath
target_binary=$target_cfg/BLD/bin/nemo.exe
[[ -x "$target_binary" ]]
# The build must have CARRIED the fix and every writer the card already had.
grep -q 'zl2_tmp(ntsi:ntei,ntsj:ntej,:) = avt' "$target_cfg/BLD/ppsrc/nemo/trazdf.f90" \
  || { printf 'REFUSE: the round-37 avt staging is absent from the compiled ppsrc (stale build)\n' >&2; exit 69; }
grep -q 'zl2_tmp(ntsi:ntei,ntsj:ntej,:) = avs' "$target_cfg/BLD/ppsrc/nemo/trazdf.f90" \
  || { printf 'REFUSE: the round-37 avs staging is absent from the compiled ppsrc (stale build)\n' >&2; exit 69; }
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

# TWIN IDENTITY, IN TWO STEPS.  RAW byte identity is tried FIRST against the
# ROUND-35 run and every result is LOGGED, because it is the strongest claim
# and the one to prefer.  It is not the CRITERION: NEMO's stream dumps write
# whole work arrays including the nn_hls halo, which NEMO neither owns nor
# initialises, and THIS run legitimately changes one record -- the two staged
# arrays make oracle_trazdf_matrix_kt00000001.bin 115072 bytes LONGER.
: >"$TARGET_RUN/round37_raw_twin_cmp.log"
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
done | tee -a "$TARGET_RUN/round37_raw_twin_cmp.log"
# The one record that MUST differ, and the 50 that must not.
grep -q "^RAW_DIFFERS   $NEW_RECORD" "$TARGET_RUN/round37_raw_twin_cmp.log" \
  || { printf 'REFUSE: %s is byte-identical to round 35; the fix did not take\n' \
       "$NEW_RECORD" >&2; exit 71; }

# THE FIELD-BY-FIELD TWIN on the changed record, through the GATE'S OWN
# READER -- the only reader that can decode both, because round 35's needs the
# interior-extent recovery and round 37's must not.  Every array must be bit
# for bit what round 35 wrote, avt and avs INCLUDED: round 36 proved the
# recovery reconstructs them correctly, so a difference here is the fix
# changing a number, which it must not do.
python - "$GATE" "$SOURCE_RUN/$NEW_RECORD" "$TARGET_RUN/$NEW_RECORD" \
  >"$TARGET_RUN/round37_record_twin.txt" <<'PYTWIN'
import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("r35gate", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
old = module.read_trazdf_matrix(module.Path(sys.argv[2]))
new = module.read_trazdf_matrix(module.Path(sys.argv[3]))
fail = []
if not old["tile_shaped_salvage"]:
    fail.append("the round-35 record needed NO salvage; it is not the mislabelled one")
if new["tile_shaped_salvage"]:
    fail.append(f"the round-37 record STILL needs salvage: {sorted(new['tile_shaped_salvage'])}")
if old["order"] != new["order"]:
    fail.append("the array order moved")
for key in sorted(set(old["header"]) | set(new["header"])):
    if old["header"].get(key) != new["header"].get(key):
        fail.append(f"header {key}: {old['header'].get(key)} -> {new['header'].get(key)}")
for name in old["order"]:
    a = np.asarray(old["arrays"][name], dtype=np.float64)
    b = np.asarray(new["arrays"][name], dtype=np.float64)
    if a.shape != b.shape:
        fail.append(f"{name}: shape {a.shape} -> {b.shape}")
    elif a.tobytes() != b.tobytes():
        n = int((a != b).sum())
        fail.append(f"{name}: {n} cells differ, max abs "
                    f"{float(np.abs(a - b).max()):.17g}")
    else:
        print(f"TWIN_IDENTICAL {name}")
print(f"TWIN_SALVAGED_IN_ROUND35 {sorted(old['tile_shaped_salvage'])}")
for line in fail:
    print(f"TWIN_FAIL {line}")
print("TWIN_VERDICT", "PASS" if not fail else "FAIL")
sys.exit(0 if not fail else 1)
PYTWIN
tail -3 "$TARGET_RUN/round37_record_twin.txt"

python "$ADMISSION" --baseline "$ADMISSION_BASE" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$NEW_RECORD" \
  --output "$TARGET_RUN/round37_admission.json"
if python "$ADMISSION" --baseline "$ADMISSION_BASE" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$NEW_RECORD" --plant-consumed \
     >"$TARGET_RUN/round37_admission_plant.json" 2>&1; then
  printf 'REFUSE: the admission plant did not turn the gate red\n' >&2
  exit 72
fi
test -s "$TARGET_RUN/$NEW_RECORD"

# The round-29 gate on the momentum record: unchanged from round 35, so every
# row it scored must still be AT-BAR.
python "$R29_GATE" --record "$TARGET_RUN/oracle_zdf_matrix_kt00000001.bin" \
  --json "$TARGET_RUN/round37_r29_regression.json"

# The round-35 gate itself, and one plant per arm.  THE GATE'S VERDICT IS THE
# MEASUREMENT, NOT AN ACQUISITION CHECK: it exits non-zero on any row over the
# exact bar, which is correct and must never be relaxed -- but a DEBT row is a
# FINDING, not a reason to abandon the run before its plants have proved the
# gate can see anything.  So the exit code is captured and reported.
set +e
python "$GATE" --record "$TARGET_RUN/$NEW_RECORD" \
  --json "$TARGET_RUN/round37_trazdf_gate.json"
gate_status=$?
set -e
grep -q '"format": "nemo-testcase-l2-gyre-round35-trazdf-matrix-v1"' \
  "$TARGET_RUN/round37_trazdf_gate.json" \
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
  out=$TARGET_RUN/round37_trazdf_plant_$arm.json
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
  sha256sum oracle_*.bin round37_admission.json round37_trazdf_gate.json \
    round37_r29_regression.json round37_record_twin.txt \
    "$FINAL_RESTART" mesh_mask.nc >round37_outputs.sha256
)
printf 'ROUND37_GATE_VERDICT exit=%s (0 = every row at bar; 1 = at least one row is DEBT, which is a finding)\n' "$gate_status"
printf 'ROUND37_GYRE_TRAZDF_MATRIX_ORACLE_READY %s\n' "$TARGET_RUN"
