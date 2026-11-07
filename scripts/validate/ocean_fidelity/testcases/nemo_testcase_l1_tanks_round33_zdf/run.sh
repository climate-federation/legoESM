#!/usr/bin/env bash
set -euo pipefail

# USER-EXECUTED acquisition only.  The agent must not run NEMO, makenemo or
# mpirun from its sandbox; it wrote this file and stopped.
#
#   Usage:  run.sh LOCK_EXCHANGE
#           run.sh OVERFLOW
#
# WHAT THIS ACQUIRES, AND WHY.  Round 32 moved legoESM's RK3 stage-3 barotropic
# correction from before the implicit solve to after it, matching
# stprk3_stg.F90:430 followed by the correction block at :439-446.  Rule 12
# says such a change lands only if the changed operator is bit-exact GIVEN
# NEMO'S OWN INPUTS on every card that EXECUTES it.  Both tanks execute it --
# ln_drgimp = T and ln_dynspg_ts = T on each (lock_kt1_10/ocean.output:560,
# :752; overflow_kt1_10/ocean.output:672, :869) -- and neither has ever been
# discharged.  What ran on them in round 32 was the oracle-relative TRAJECTORY
# MOVE gate, which is a different thing, and it FAILED on OVERFLOW.
#
# The discharge needs arrays NEMO holds only inside dyn_zdf and never writes:
# puu(:,:,:,Kaa) as dyn_zdf LEAVES it, uu_b(:,:,Kaa), the masks, and -- added
# this round -- NEMO's own e3u_0, hu_0 and r1_hu_0.  GYRE already has the first
# group from the round-29 instrument.  This run applies THAT SAME patch file to
# the two tanks, which compile the same shipped dynzdf.F90 body, plus one
# round-33 addition on top of it for the reference geometry.
#
# ARM DIFFERENCES, checked rather than assumed (the patch is arm-agnostic; this
# is here so the reader knows what the record will contain):
#   * ln_dynadv_vec = F on both tanks, and neither compiles key_linssh, so
#     dynzdf.F90:119 takes the ELSE arm at :127-132 -- the key_qco form.  GYRE
#     takes the vector arm at :121-122.
#   * ln_zad_Aimp = T on both tanks and F on GYRE, so the tanks' matrix carries
#     the implicit-vertical-advection terms.  The instrument copies zwi/zwd/zws
#     AS BUILT, after every contribution, so it records whatever the branch did.
#   * ln_drgimp = T and ln_dynspg_ts = T on both, so :148-171 runs and
#     uu_Kaa_pre is the post-removal, post-bottom-stress vector.
#   * key_RK3 on both, so the stage-3 record is the RK3 stage-3 one.
#
# THE PATH MATTERS, AND ATTEMPT 1 DIED ON IT.  FCM's extract step parses Fortran
# with perl's Text::Balanced, which the system perl on this machine does not
# have; makenemo then fails with "Can't locate Text/Balanced.pm" long before it
# compiles anything.  Run with the conda build environment FIRST on PATH:
#
#   PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:$PATH \
#     scripts/.../nemo_testcase_l1_tanks_round33_zdf/run.sh LOCK_EXCHANGE
#
# The line below sets it so the next operator does not have to remember, and it
# is the SAME PATH the two round-33 builds actually used
# (round33/tanks_run_{LOCK_EXCHANGE,OVERFLOW}.attempt2.log).
export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly L1=/data/abyssal/dbalwada/nemo-testcases-l1/phase3
readonly L2=/data/abyssal/dbalwada/nemo-testcases-l2/phase3

card=${1:-}
case "$card" in
  LOCK_EXCHANGE)
    SOURCE_CFG=LOCK_EXCHANGE_OMIP_L1_P3
    TARGET_CFG=LOCK_EXCHANGE_OMIP_L1_P3_R33ZDF
    TEST_CASE=LOCK_EXCHANGE
    SOURCE_RUN=$L1/lock_kt1_10
    TARGET_RUN=$L2/round33_lock_zdf_matrix
    FINAL_RESTART=LOCK_EXCHANGE_OMIP_L1_ZCO_00000010_restart.nc
    ;;
  OVERFLOW)
    SOURCE_CFG=OVERFLOW_OMIP_L1_P3
    TARGET_CFG=OVERFLOW_OMIP_L1_P3_R33ZDF
    TEST_CASE=OVERFLOW
    SOURCE_RUN=$L1/overflow_kt1_10
    TARGET_RUN=$L2/round33_overflow_zdf_matrix
    FINAL_RESTART=OVERFLOW_OMIP_L1_ZPS_00000010_restart.nc
    ;;
  *)
    printf 'Usage: %s LOCK_EXCHANGE|OVERFLOW\n' "$0" >&2
    exit 64
    ;;
esac
readonly SOURCE_CFG TARGET_CFG TEST_CASE SOURCE_RUN TARGET_RUN FINAL_RESTART

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
# The SAME patch GYRE's round-29 record was written with.  Not a copy: if the
# instrument ever changes, all three cards change together or none does.
readonly ZDF_PATCH=$here/../nemo_testcase_l2_gyre_round29_zdf/dynzdf_round29.patch
# ... plus ONE round-33 addition, applied ON TOP of it: NEMO's own e3u_0/e3v_0
# (preprocessor MACROS, materialised elementwise), hu_0/hv_0 and r1_hu_0/r1_hv_0.
# Without it the discharge would have to REBUILD the column divisor, and on
# LOCK it could not even do that -- LOCK's mesh_mask.nc carries no e3*_0 at
# all, because under key_vco_1d the vertical coordinate IS the 1-D ladder
# (domzgr_substitute.h90:89).
readonly REFGEOM_PATCH=$here/dynzdf_round33_refgeom.patch
readonly GATE=$here/../nemo_testcase_l1_tanks_round33_zdf_rule12.py
# The SHARED consumed-field admission, not a copy: the same gate GYRE's round-21
# and round-29 acquisitions were admitted with, generalised in round 34 so the
# dimensions come from each record's own header.
readonly ADMISSION=$here/../nemo_testcase_l2_gyre_round21_admission.py
readonly SHIPPED=$NEMO_ROOT/src/OCE/DYN/dynzdf.F90

source_cfg=$NEMO_ROOT/tests/$SOURCE_CFG
target_cfg=$NEMO_ROOT/tests/$TARGET_CFG

[[ -d "$source_cfg/MY_SRC" && -d "$source_cfg/EXP00" ]]
[[ -f "$ZDF_PATCH" && -f "$REFGEOM_PATCH" && -f "$SHIPPED" && -f "$GATE" ]]
[[ -f "$ADMISSION" ]]
[[ -d "$SOURCE_RUN" ]]
if [[ -e "$target_cfg" || -e "$TARGET_RUN" ]]; then
  printf 'REFUSE: target already exists: %s or %s\n' "$target_cfg" "$TARGET_RUN" >&2
  exit 64
fi
# The premise the shared patch rests on: this card compiles the SHIPPED
# dynzdf body and does not override it.  Checked, not assumed.
if [[ -e "$source_cfg/MY_SRC/dynzdf.F90" ]]; then
  printf 'REFUSE: %s overrides dynzdf.F90 in MY_SRC; the shared-patch premise is false\n' \
    "$SOURCE_CFG" >&2
  exit 66
fi
if [[ "$(readlink -f "$source_cfg/WORK/dynzdf.F90")" != "$(readlink -f "$SHIPPED")" ]]; then
  printf 'REFUSE: %s does not compile the shipped dynzdf body\n' "$SOURCE_CFG" >&2
  exit 66
fi
# A WRITE-only instrument may ADD lines; it may not delete or change one.
# `diff | grep -q` would return diff's own exit status under pipefail and could
# never fire, which is how an earlier round shipped a check that proved nothing;
# count the removed lines instead.
dry=$(mktemp -d /tmp/tanks-r33-dryrun.XXXXXX)
cp "$SHIPPED" "$dry/dynzdf.F90"
if ! patch -s "$dry/dynzdf.F90" <"$ZDF_PATCH" >/dev/null 2>&1 \
   || ! patch -s "$dry/dynzdf.F90" <"$REFGEOM_PATCH" >/dev/null 2>&1; then
  printf 'REFUSE: the instrument patches do not stack cleanly on the shipped dynzdf\n' >&2
  rm -rf "$dry"; exit 67
fi
rm -rf "$dry"
for patch_file in "$ZDF_PATCH" "$REFGEOM_PATCH"; do
  # '^-[^-]' would MISS a deleted BLANK line, which a unified diff emits as a
  # bare '-'.  Match every removal line and exclude only the '---' file header.
  if [[ $(grep -c '^-' "$patch_file") -ne $(grep -c '^---' "$patch_file") ]]; then
    printf 'REFUSE: %s deletes or changes a shipped line; it must only ADD\n' \
      "$patch_file" >&2
    exit 67
  fi
done
# Space.  The OVERFLOW record is jpi*jpj*jpk*8 per array over ~25 arrays, about
# 30 MB, and the build tree wants a few hundred MB; refuse rather than fail
# half way through a build.
for mount in /tmp "$(dirname "$TARGET_RUN")" "$NEMO_ROOT"; do
  free_kb=$(df -Pk "$mount" | awk 'NR==2 {print $4}')
  if [[ "$free_kb" -lt 2097152 ]]; then
    printf 'REFUSE: %s has %s kB free, under the 2 GB floor\n' "$mount" "$free_kb" >&2
    exit 68
  fi
done

work_manifest=$(mktemp -d "/tmp/tanks-r33-${card}.XXXXXX")
printf 'temporary provenance directory (retained): %s\n' "$work_manifest"
(
  cd "$source_cfg"
  find EXP00 MY_SRC -type f -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/source_cfg.sha256"
sha256sum "$NEMO_ROOT/arch/arch-conda-scalarmath.fcm" \
  "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$SHIPPED" "$ZDF_PATCH" \
  >"$work_manifest/toolchain.sha256"

cd "$NEMO_ROOT"
./makenemo -a "$TEST_CASE" -n "$TARGET_CFG" -m conda-scalarmath del_key 'key_xios'
# cp -r, NOT cp -a: preserved mtimes let fcm decide a patched file is already
# built.  MY_SRC is touched below for the same reason.
cp -r "$source_cfg/EXP00/." "$target_cfg/EXP00/"
cp -r "$source_cfg/MY_SRC/." "$target_cfg/MY_SRC/"
cp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
# dynzdf has no MY_SRC copy on either tank; create it from the SHIPPED source.
# Copying the shipped file is allowed; editing NEMO's own tree is not.
[[ ! -e "$target_cfg/MY_SRC/dynzdf.F90" ]]
cp "$SHIPPED" "$target_cfg/MY_SRC/dynzdf.F90"
(
  cd "$target_cfg"
  find EXP00 MY_SRC -type f ! -name dynzdf.F90 -print0 | sort -z | xargs -0 sha256sum
) >"$work_manifest/copied_cfg_before_patch.sha256"
cmp "$work_manifest/source_cfg.sha256" "$work_manifest/copied_cfg_before_patch.sha256"
cmp "$source_cfg/cpp_${SOURCE_CFG}.fcm" "$target_cfg/cpp_${TARGET_CFG}.fcm"
cmp "$SHIPPED" "$target_cfg/MY_SRC/dynzdf.F90"

patch "$target_cfg/MY_SRC/dynzdf.F90" <"$ZDF_PATCH"
patch "$target_cfg/MY_SRC/dynzdf.F90" <"$REFGEOM_PATCH"
touch "$target_cfg/MY_SRC/"*.F90
sha256sum "$target_cfg/MY_SRC/dynzdf.F90" "$ZDF_PATCH" "$REFGEOM_PATCH" \
  >"$work_manifest/round33_instrument.sha256"

./makenemo -n "$TARGET_CFG" -m conda-scalarmath
target_binary=$target_cfg/BLD/bin/nemo.exe
[[ -x "$target_binary" ]]
# The build must have CARRIED the instrument, not reused a stale object, and
# the card's OWN pre-existing writers must still be compiled or the twin
# identity check below would compare against records that were never written.
grep -q 'NEMO_L2_ZDFMX_1' "$target_cfg/BLD/ppsrc/nemo/dynzdf.f90" \
  || { printf 'REFUSE: the dyn_zdf writer is absent from the compiled ppsrc (stale build)\n' >&2; exit 69; }
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
for name in namelist_cfg namelist_ref context_nemo.xml file_def_nemo-oce.xml \
  iodef.xml; do
  cp -L "$SOURCE_RUN/$name" "$TARGET_RUN/$name"
done
cp "$target_binary" "$TARGET_RUN/nemo"
cp "$work_manifest"/*.sha256 "$TARGET_RUN/"
(
  cd "$TARGET_RUN"
  sha256sum namelist_cfg namelist_ref context_nemo.xml file_def_nemo-oce.xml \
    iodef.xml >prepared_files.sha256
  # Verifying that manifest against the files it was just generated from could
  # never fail.  Compare the COPIES against the source run's originals, which
  # is the thing that can actually be wrong.
  for name in namelist_cfg namelist_ref context_nemo.xml \
    file_def_nemo-oce.xml iodef.xml; do
    cmp "$SOURCE_RUN/$name" "$name"
  done
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
# result is logged, because it is the strongest claim and the one to prefer.
# It is not, however, the CRITERION: NEMO's stream dumps write whole work
# arrays including the nn_hls = 2 halo, which NEMO neither owns nor
# initialises, so two runs of the same executable can differ in bytes that are
# uninitialised memory.  Round 33's attempt died here under set -e on exactly
# that, and everything below it -- the mesh and restart checks, the Rule-12
# discharge, its plant and the outputs manifest -- never ran.
#
# So a raw difference falls through to the shared CONSUMED-FIELD ADMISSION,
# which admits a difference only when every differing ELEMENT is in the halo or
# in a slot the writer has not defined at the write point, and PRINTS each
# admitted difference with its value.  One differing bit in an OWNED cell still
# stops the round.
: >"$TARGET_RUN/round33_raw_twin_cmp.log"
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
done | tee -a "$TARGET_RUN/round33_raw_twin_cmp.log"
if grep -q '^RAW_DIFFERS' "$TARGET_RUN/round33_raw_twin_cmp.log"; then
  printf 'raw twin identity is NOT exact; running the consumed-field admission\n'
else
  printf 'raw twin identity is EXACT on every inherited record\n'
fi
# The admission checks the mesh and the final restart too, so they are named
# here rather than compared separately.
python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
  --twin /nonexistent \
  --identical "$FINAL_RESTART" mesh_mask.nc \
  --allowed-new oracle_zdf_matrix_kt00000001.bin \
  --output "$TARGET_RUN/round33_admission.json"
if python "$ADMISSION" --baseline "$SOURCE_RUN" --candidate "$TARGET_RUN" \
     --twin /nonexistent --identical "$FINAL_RESTART" mesh_mask.nc \
     --allowed-new oracle_zdf_matrix_kt00000001.bin --plant-consumed \
     >"$TARGET_RUN/round33_admission_plant.json" 2>&1; then
  printf 'REFUSE: the admission plant did not turn the gate red\n' >&2
  exit 72
fi
test -s "$TARGET_RUN/oracle_zdf_matrix_kt00000001.bin"

# The Rule-12 discharge itself, and its plant.  One command each, and the plant
# MUST exit non-zero or the discharge proves nothing.
python "$GATE" --card "$card" --output "$TARGET_RUN/round33_rule12_correction.json"
if python "$GATE" --card "$card" --plant \
     >"$TARGET_RUN/round33_rule12_correction_plant.json" 2>&1; then
  printf 'REFUSE: the planted control did not turn the gate red\n' >&2
  exit 70
fi
(
  cd "$TARGET_RUN"
  sha256sum oracle_*.bin round33_admission.json \
    round33_rule12_correction.json "$FINAL_RESTART" mesh_mask.nc \
    >round33_outputs.sha256
)
printf 'ROUND33_%s_ZDF_MATRIX_ORACLE_READY %s\n' "$card" "$TARGET_RUN"
