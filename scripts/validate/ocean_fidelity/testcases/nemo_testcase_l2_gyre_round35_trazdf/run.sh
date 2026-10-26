#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED acquisition only.  The agent must not run NEMO, makenemo or
# mpirun from its sandbox; it wrote this file and stopped.
#
#   Usage:  run.sh
#
# WHAT THIS ACQUIRES, AND WHY.  Round 34 localised GYRE's kt=2 T/S failure to
# ONE routine.  At kt=1 stage 3 the accumulator handed to tra_zdf is AT-BAR on
# both tracers -- T 6.05e-16, S 5.79e-16 -- while tra_zdf's OUTPUT is
# 1.36e-12 on T; advection, tra_sbc_RK3, tra_qsr and tra_ldf are all inside
# that at-bar accumulator and are therefore exonerated, tra_ldf twice over
# because its own increment is exactly 0.  What no record carries is the
# routine's own tridiagonal: round 29 instrumented dyn_zdf's MOMENTUM matrix,
# not this one.  This run adds it.
#
# TWO PATCHES, STACKED IN ONE CONFIG COPY:
#
#  1. trazdf_round35.patch, on the SHIPPED src/OCE/TRA/trazdf.F90 -- the new
#     instrument.  It dumps, at kt = nit000, the tracer RHS as tra_zdf
#     RECEIVES it, the before/now tracer fields, the mixing coefficient
#     zwt = avt + ah_wslp2 AS ASSEMBLED, zwi/zwd/zws AS BUILT, the LU diagonal
#     after the first recurrence, the per-level right-hand side, the forward
#     sweep, the solved column BEFORE and AFTER NEMO's own negative-salinity
#     clamp, and every operand the solve reads -- avt, avs, ah_wslp2, akz,
#     tmask, the four e3t/e3w time-level substitutions AND their raw
#     ingredients e3t_0/e3w_0/r3t, plus rDt and the branch flags.
#
#  2. dynzdf_round33_refgeom.patch, on the round-29-patched MY_SRC/dynzdf.F90
#     this config INHERITS -- NEMO's own e3u_0/e3v_0, hu_0/hv_0 and
#     r1_hu_0/r1_hv_0 appended to the existing momentum record.  It is the
#     SAME file the two tanks were acquired with, not a copy.  Reason: the
#     independent review of rounds 33-34 measured that the round-32 barotropic
#     correction becomes exactly bit-identical on both tanks when legoESM's
#     sum/hu_0 is replaced by NEMO's literal sum * r1_hu_0 -- which is what
#     stprk3_stg.F90:440-441 writes -- and GYRE's discharge currently REBUILDS
#     that divisor because no GYRE record carries r1_hu_0.  This run makes the
#     record carry it; the transcription change itself is round 36's.
#
# THE ARM, read off the run's own resolved namelist rather than assumed
# (round29_oracle_v2_zdf_matrix/ocean.output), because the reader REFUSES a
# record from any other one:
#   * ln_zdfddm = F (:568)      -- ONE matrix, built for T and reused for S
#   * ln_zad_Aimp = F (:553)    -- the plain diagonal, no implicit vert. adv.
#   * ln_zdfmfc = F (:561)      -- no mass-flux terms in matrix or RHS
#   * ln_traldf_msc = F (:657)  -- the ah_wslp2 fold, not the akz one
#   * ln_traldf_lap = T (:650) + ln_traldf_iso = T (:655), resolved as
#     "Rotated laplacian operator (standard)" (:667), so l_ldfslp is TRUE and
#     the a33 fold IS inside the matrix
#   * ln_SEOS = F (:158)        -- the DRAKKAR clamp at trazdf.F90:89-91 RUNS
#
# THE PATH MATTERS.  FCM's extract step parses Fortran with perl's
# Text::Balanced, which the system perl on this machine does not have;
# makenemo then fails with "Can't locate Text/Balanced.pm" long before it
# compiles anything.  The line below puts the conda build environment first,
# which is the PATH the round-33 builds actually used.
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly SOURCE_CFG=GYRE_OMIP_L2_P3_SM_R29ZDF
readonly TARGET_CFG=GYRE_OMIP_L2_P3_SM_R35TRAZDF
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3
readonly SOURCE_RUN=$L2/round29_oracle_v2_zdf_matrix
readonly TARGET_RUN=$L2/round35_oracle_trazdf_matrix
readonly FINAL_RESTART=GYRE_OMIP_L2_P3_00000010_restart.nc
readonly NEW_RECORD=oracle_trazdf_matrix_kt00000001.bin

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly TRA_PATCH=$here/trazdf_round35.patch
# NOT a copy: the same file the two tanks were acquired with, so if the
# reference-geometry instrument ever changes, all three cards change together.
readonly REFGEOM_PATCH=$here/../nemo_testcase_l1_tanks_round33_zdf/dynzdf_round33_refgeom.patch
readonly GATE=$here/../nemo_testcase_l2_gyre_round35_trazdf_matrix.py
readonly R29_GATE=$here/../nemo_testcase_l2_gyre_round29_zdf_matrix.py
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly SHIPPED=$NEMO_ROOT/src/OCE/TRA/trazdf.F90

source_cfg=$NEMO_ROOT/cfgs/$SOURCE_CFG
target_cfg=$NEMO_ROOT/cfgs/$TARGET_CFG

[[ -d "$source_cfg/MY_SRC" && -d "$source_cfg/EXP00" ]]
[[ -f "$TRA_PATCH" && -f "$REFGEOM_PATCH" && -f "$SHIPPED" ]]
[[ -f "$GATE" && -f "$R29_GATE" && -f "$ADMISSION" ]]
[[ -d "$SOURCE_RUN" ]]
if [[ -e "$target_cfg" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: target already exists: %s or %s\n' "$target_cfg" "$TARGET_RUN" >&2
  exit 64
fi
# The premise patch 1 rests on: this card compiles the SHIPPED trazdf body and
# does not override it.  Checked, not assumed.
if [[ -e "$source_cfg/MY_SRC/trazdf.F90" ]]; then
  printf 'REFUSE: %s overrides trazdf.F90 in MY_SRC; the shipped-source premise is false\n' \
    "$SOURCE_CFG" >&2
  exit 66
fi
# The premise patch 2 rests on: this card DOES carry the round-29 dynzdf
# instrument, so the reference-geometry addition stacks on it.
if [[ ! -e "$source_cfg/MY_SRC/dynzdf.F90" ]]; then
  printf 'REFUSE: %s has no MY_SRC/dynzdf.F90 to stack the round-33 addition on\n' \
    "$SOURCE_CFG" >&2
  exit 66
fi
if ! grep -q 'NEMO_L2_ZDFMX_1' "$source_cfg/MY_SRC/dynzdf.F90"; then
  printf 'REFUSE: %s MY_SRC/dynzdf.F90 is not the round-29 instrument\n' "$SOURCE_CFG" >&2
  exit 66
fi
# A WRITE-only instrument may ADD lines; it may not delete or change one.
# `diff | grep -q` would return diff's own exit status under pipefail and could
# never fire, which is how an earlier round shipped a check that proved
# nothing; count the removed lines instead.  '^-[^-]' would MISS a deleted
# BLANK line, which a unified diff emits as a bare '-'.
for patch_file in "$TRA_PATCH" "$REFGEOM_PATCH"; do
  if [[ $(grep -c '^-' "$patch_file") -ne $(grep -c '^---' "$patch_file") ]]; then
    printf 'REFUSE: %s deletes or changes a shipped line; it must only ADD\n' \
      "$patch_file" >&2
    exit 67
  fi
done
dry=$(mktemp -d /tmp/gyre-r35-dryrun.XXXXXX)
cp "$SHIPPED" "$dry/trazdf.F90"
cp "$source_cfg/MY_SRC/dynzdf.F90" "$dry/dynzdf.F90"
if ! patch -s "$dry/trazdf.F90" <"$TRA_PATCH" >/dev/null 2>&1 \
   || ! patch -s "$dry/dynzdf.F90" <"$REFGEOM_PATCH" >/dev/null 2>&1; then
  printf 'REFUSE: the instrument patches do not apply cleanly\n' >&2
  rm -rf "$dry"; exit 67
fi
rm -rf "$dry"
# Space.  The new record is jpi*jpj*jpk*8 over ~30 3-D arrays, about 7 MB, and
# the build tree wants a few hundred MB; refuse rather than fail half way.
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 2097152 ]]; then
    printf 'REFUSE: %s has %s kB free, under the 2 GB floor\n' "$mount" "$free_kb" >&2
    exit 68
  fi
done

work_manifest=$(mktemp -d /tmp/gyre-r35-source.XXXXXX)
printf 'temporary provenance directory (retained): %s\n' "$work_manifest"
(
  cd "$source_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$SHIPPED" "$TRA_PATCH" "$REFGEOM_PATCH" \
  >"$work_manifest/toolchain.sha256"

cd "$NEMO_ROOT"
# No del_key: the cpp key file is copied wholesale from the source card below,
# so the target inherits exactly its keys (key_qco key_vco_1d3d key_RK3).
./makenemo -r GYRE_PISCES -n "$TARGET_CFG" -m conda-scalarmath
# cp -r, NOT cp -a: preserved mtimes let fcm decide a patched file is already
# built.  MY_SRC is touched below for the same reason.
cp -r "$source_cfg/EXP00/." "$target_cfg/EXP00/"
cp -r "$source_cfg/MY_SRC/." "$target_cfg/MY_SRC/"
cp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
# trazdf has no MY_SRC copy on this card; create it from the SHIPPED source.
# Copying the shipped file is allowed; editing NEMO's own tree is not.
[[ ! -e "$target_cfg/MY_SRC/trazdf.F90" ]]
cp "$SHIPPED" "$target_cfg/MY_SRC/trazdf.F90"
(
  cd "$target_cfg"
  find EXP00 MY_SRC -type f ! -name trazdf.F90 -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/copied_cfg_before_patch.sha256"
cmp "$work_manifest/source_cfg.sha256" "$work_manifest/copied_cfg_before_patch.sha256"
cmp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
cmp "$SHIPPED" "$target_cfg/MY_SRC/trazdf.F90"

patch "$target_cfg/MY_SRC/trazdf.F90" <"$TRA_PATCH"
patch "$target_cfg/MY_SRC/dynzdf.F90" <"$REFGEOM_PATCH"
touch "$target_cfg/MY_SRC/"*.F90
sha256sum "$target_cfg/MY_SRC/trazdf.F90" "$target_cfg/MY_SRC/dynzdf.F90" \
  "$TRA_PATCH" "$REFGEOM_PATCH" >"$work_manifest/round35_instrument.sha256"

./makenemo -n "$TARGET_CFG" -m conda-scalarmath
target_binary=$target_cfg/BLD/bin/nemo.exe
[[ -x "$target_binary" ]]
# The build must have CARRIED both instruments, not reused a stale object, and
# the card's OWN pre-existing writers must still be compiled or the twin
# identity check below would compare against records that were never written.
grep -q 'NEMO_L2_TRAZD_1' "$target_cfg/BLD/ppsrc/nemo/trazdf.f90" \
  || { printf 'REFUSE: the tra_zdf writer is absent from the compiled ppsrc (stale build)\n' >&2; exit 69; }
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
for name in namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref \
  namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
  iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
  field_def_nemo-oce.xml field_def_nemo-pisces.xml; do
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
cp "$target_binary" "$TARGET_RUN/nemo"
cp "$work_manifest"/*.sha256 "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  # Verifying a manifest against the files it was just generated from could
  # never fail.  Compare the COPIES against the source run's originals, which
  # is the thing that can actually be wrong.
  for name in namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref \
    namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
    iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
    field_def_nemo-oce.xml field_def_nemo-pisces.xml; do
    cmp "$SOURCE_RUN/$name" "$name"
  done
  sha256sum namelist_cfg namelist_ref namelist_top_cfg namelist_top_ref \
    namelist_pisces_cfg namelist_pisces_ref context_nemo.xml file_def_nemo.xml \
    iodef.xml axis_def_nemo.xml domain_def_nemo.xml grid_def_nemo.xml \
    field_def_nemo-oce.xml field_def_nemo-pisces.xml >prepared_files.sha256
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

# TWIN IDENTITY, IN TWO STEPS.  RAW byte identity is tried FIRST and every
# result is LOGGED, because it is the strongest claim and the one to prefer.
# It is not, however, the CRITERION, and a bare `cmp` under `set -e` would
# take the whole round down with it: NEMO's stream dumps write whole work
# arrays including the nn_hls halo, which NEMO neither owns nor initialises,
# and THIS run legitimately grows one inherited record -- the round-33
# reference-geometry addition appends six arrays to oracle_zdf_matrix.
#
# So a raw difference falls through to the shared CONSUMED-FIELD ADMISSION,
# which admits a difference only when every differing ELEMENT is in the halo
# or in a slot the writer has not defined at the write point, compares a
# self-describing record over the INTERSECTION of the two field sets so the
# legitimate growth passes while a state change in the shared fields does
# not, and PRINTS each admitted difference with its value.
: >"$TARGET_RUN/round35_raw_twin_cmp.log"
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
done | tee -a "$TARGET_RUN/round35_raw_twin_cmp.log"
if grep -q '^RAW_DIFFERS' "$TARGET_RUN/round35_raw_twin_cmp.log"; then
  printf 'raw twin identity is NOT exact; running the consumed-field admission\n'
else
  printf 'raw twin identity is EXACT on every inherited record\n'
fi
python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new "$NEW_RECORD" \
  --output "$TARGET_RUN/round35_admission.json"
if python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new "$NEW_RECORD" --plant-consumed \
     >"$TARGET_RUN/round35_admission_plant.json" 2>&1; then
  printf 'REFUSE: the admission plant did not turn the gate red\n' >&2
  exit 72
fi
test -s "$TARGET_RUN/$NEW_RECORD"

# The round-29 gate, re-run on the GROWN momentum record: the reference
# geometry is appended, so every row it already scored must be unchanged.
python "$R29_GATE" --record "$TARGET_RUN/oracle_zdf_matrix_kt00000001.bin" \
  --json "$TARGET_RUN/round35_r29_regression.json"

# The round-35 gate itself, and one plant per arm.  Each plant MUST exit
# non-zero or the arm it belongs to proves nothing.
# THE GATE'S VERDICT IS THE MEASUREMENT, NOT AN ACQUISITION CHECK.  It exits
# non-zero on any row over the exact bar, which is correct and must never be
# relaxed -- but a DEBT row is this round's FINDING, not a reason to abandon
# the run before its plants have proved the gate can see anything.  So the
# exit code is captured and reported at the end rather than tripping set -e.
set +e
python "$GATE" --record "$TARGET_RUN/$NEW_RECORD" \
  --json "$TARGET_RUN/round35_trazdf_gate.json"
gate_status=$?
set -e
grep -q '"format": "nemo-testcase-l2-gyre-round35-trazdf-matrix-v1"' \
  "$TARGET_RUN/round35_trazdf_gate.json" \
  || { printf 'REFUSE: the gate produced no report; it crashed rather than judging\n' >&2; exit 73; }
# The arm list comes from the GATE, never a copy here: a hardcoded list would
# silently stop testing an arm the gate later grows.
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
  out=$TARGET_RUN/round35_trazdf_plant_$arm.json
  if python "$GATE" --record "$TARGET_RUN/$NEW_RECORD" --plant "$arm" >"$out" 2>&1; then
    printf 'REFUSE: the %s plant did not turn the gate red\n' "$arm" >&2
    exit 70
  fi
  # A non-zero exit is NOT enough, twice over.  A plant that CRASHED the
  # reader exits non-zero and proves nothing; and this gate's BASELINE may
  # already be red, so a plant that moved nothing would exit non-zero too.
  # Require the gate's own verdict AND its own statement that the plant
  # landed on at least one row.
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
  sha256sum oracle_*.bin round35_admission.json round35_trazdf_gate.json \
    round35_r29_regression.json "$FINAL_RESTART" mesh_mask.nc \
    >round35_outputs.sha256
)
printf 'ROUND35_GATE_VERDICT exit=%s (0 = every row at bar; 1 = at least one row is DEBT, which is a finding)\n' "$gate_status"
printf 'ROUND35_GYRE_TRAZDF_MATRIX_ORACLE_READY %s\n' "$TARGET_RUN"
