#!/bin/bash
# Build NEMO per-substep SEQ-DUMP reference lanes at the 10-day twin restarts.
#
# #1455 PHASE 1 (the time axis).  The barotropic substep walk
# (scripts/validate/ocean_fidelity/dino_1226/substep_traj_compare.py) measures a
# one-step ACC deposit difference against NEMO's own per-substep dump.  That dump
# existed at ONE state only (day 180, RUN_SEQDUMP_D180_1R), so the walk was n=1
# in time and could not tell an oscillation from an accumulation.  This script
# builds the SAME 1-rank instrumented lane at every 10-day restart the 90-day
# twin carries (day 190..270), so the deposit becomes a time series.
#
# For each NEMO step kt it:
#   1. stitches RUN_90D_TWIN's 16 per-rank restart tiles into one file with the
#      shipped REBUILD_NEMO tool (NEMO's 1-rank read needs a single file; the
#      legoESM bridge reads the SAME tiles in memory via rebuild_nemo_restart),
#   2. clones RUN_SEQDUMP_D180_1R's namelist with only nn_it000/nn_itend/
#      cn_ocerst_in changed -- one variable per lane, everything else identical,
#   3. runs the SAME instrumented binary (nemo.exe.seqdump_ea0c113c, the one
#      whose day-180 dumps the walk already used) for 4 steps.
#
# Day 180 is NOT rebuilt: RUN_TRAJ already carries its stitched restart and
# RUN_SEQDUMP_D180_1R already carries its dumps.  Re-running it would change the
# reference the three committed walk commits were measured against.
#
# Idempotent: a lane whose substep_dump.bin already exists is skipped.
set -euo pipefail

DINO="${DINO_ORACLE_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO}"
REBUILD_EXE="${DINO}/../../tools/REBUILD_NEMO/rebuild_nemo.exe"
BIN="${DINO}/BLD/bin/nemo.exe.seqdump_ea0c113c"
# Restart source.  Default = the 90-day twin's 16 per-rank tiles (stitched
# below).  DINO_SEQDUMP_SRC_1R names a directory that already holds SINGLE-FILE
# 1-rank restarts, in which case no stitching happens and the file is used
# as-is -- that is the consecutive-step lane (RUN_D180_STEP1_1R).
TWIN="${DINO_SEQDUMP_SRC:-${DINO}/RUN_90D_TWIN}"
SRC_1R="${DINO_SEQDUMP_SRC_1R:-}"
# Lane naming: by DAY for the 10-day grid, by KT when the states are closer
# together than a day (kt*2700/86400 would collide).
NAME_BY_KT="${DINO_SEQDUMP_NAME_BY_KT:-0}"
TEMPLATE="${DINO}/RUN_SEQDUMP_D180_1R"
STAGE="${DINO}/REBUILD_TWIN"

# kt -> day: kt * 2700 s / 86400.  6080=d190 ... 8640=d270.
KTS="${DINO_SEQDUMP_KTS:-6080 6400 6720 7040 7360 7680 8000 8320 8640}"

for f in "$REBUILD_EXE" "$BIN" "$TEMPLATE/namelist_cfg" "$TEMPLATE/namelist_ref"; do
  [ -e "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
mkdir -p "$STAGE"

for KT in $KTS; do
  DAY=$(( KT * 2700 / 86400 ))
  if [ "$NAME_BY_KT" = "1" ]; then
    RUNDIR="${DINO}/RUN_SEQDUMP_KT${KT}_1R"
  else
    RUNDIR="${DINO}/RUN_SEQDUMP_D${DAY}_1R"
  fi
  BASE=$(printf "DINO_%08d_restart" "$KT")
  # The day-180 lane is the reference the three committed walk commits were
  # measured against; rebuilding it would move that reference.  Guard on the
  # RESOLVED DIRECTORY, not on the kt literal: with day-based naming every kt
  # in 5760..5791 integer-divides to day 180, so a kt-only refusal leaves the
  # reference lane exposed to the `rm -rf` below.  (Found by adversarial
  # review; the kt-only version of this very guard had that hole.)
  if [ "$RUNDIR" = "$TEMPLATE" ]; then
    # SKIP, do not exit: a batch list that happens to contain such a kt must
    # not kill every lane after it.
    echo "SKIPPING kt=${KT}: it resolves to ${RUNDIR}, the reference lane the" \
         "committed walk was measured against. Set DINO_SEQDUMP_NAME_BY_KT=1" \
         "for sub-daily states." >&2
    continue
  fi
  if [ -s "${RUNDIR}/substep_dump.bin" ]; then
    echo "[skip] day ${DAY} (kt=${KT}): ${RUNDIR}/substep_dump.bin present"
    continue
  fi

  # ---- 1. stitch the 16 tiles (or take an existing 1-rank restart) ------------
  if [ -n "$SRC_1R" ]; then
    [ -s "${SRC_1R}/${BASE}.nc" ] || { echo "FATAL: no ${SRC_1R}/${BASE}.nc" >&2; exit 1; }
    # Never silently replace a staged REBUILD_NEMO stitch with a 1-rank file at
    # the same path: a later day-grid run would then read the wrong restart.
    if [ -e "${STAGE}/${BASE}.nc" ] && \
       [ "$(readlink -f "${STAGE}/${BASE}.nc")" != "$(readlink -f "${SRC_1R}/${BASE}.nc")" ]; then
      echo "FATAL: ${STAGE}/${BASE}.nc already exists and points elsewhere" >&2
      exit 1
    fi
    ln -sf "${SRC_1R}/${BASE}.nc" "${STAGE}/${BASE}.nc"
  elif [ ! -s "${STAGE}/${BASE}.nc" ]; then
    ntile=$(ls "${TWIN}/${BASE}"_[0-9][0-9][0-9][0-9].nc 2>/dev/null | wc -l)
    [ "$ntile" -eq 16 ] || { echo "FATAL: kt=${KT} has ${ntile} tiles, expected 16" >&2; exit 1; }
    ( cd "$STAGE"
      for i in $(seq 0 15); do ln -sf "../RUN_90D_TWIN/${BASE}_$(printf %04d "$i").nc" .; done
      printf "&nam_rebuild\nfilebase='%s'\nndomain=16\n/\n" "$BASE" > nam_rebuild
      "$REBUILD_EXE" nam_rebuild > "rebuild_${KT}.log" 2>&1
      grep -q "rebuild completed successfully" "rebuild_${KT}.log" \
        || { echo "FATAL: rebuild failed for kt=${KT}" >&2; exit 1; } )
  fi

  # ---- 2. clone the day-180 lane, one variable changed ------------------------
  rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"
  cp "${TEMPLATE}/namelist_ref" "$RUNDIR/"
  sed -e "s/nn_it000    =       5761/nn_it000    =       $((KT+1))/" \
      -e "s/nn_itend    =       5764/nn_itend    =       $((KT+4))/" \
      -e "s/cn_ocerst_in = \"DINO_00005760_restart\"/cn_ocerst_in = \"${BASE}\"/" \
      "${TEMPLATE}/namelist_cfg" > "${RUNDIR}/namelist_cfg"
  grep -q "nn_it000    =       $((KT+1))" "${RUNDIR}/namelist_cfg" \
    || { echo "FATAL: nn_it000 substitution missed for kt=${KT}" >&2; exit 1; }
  grep -q "nn_itend    =       $((KT+4))" "${RUNDIR}/namelist_cfg" \
    || { echo "FATAL: nn_itend substitution missed for kt=${KT}" >&2; exit 1; }
  grep -q "cn_ocerst_in = \"${BASE}\"" "${RUNDIR}/namelist_cfg" \
    || { echo "FATAL: cn_ocerst_in substitution missed for kt=${KT}" >&2; exit 1; }
  ln -sf "../REBUILD_TWIN/${BASE}.nc" "${RUNDIR}/"
  ln -sf "$BIN" "${RUNDIR}/nemo"

  # ---- 3. run the instrumented binary ----------------------------------------
  # Retries: this binary intermittently segfaults during INITIALISATION (in
  # dia_obs_init/dia_detide_init, before any dump is written or any physics is
  # stepped) and runs clean on an immediate retry from the same directory --
  # observed at day 260 (1 retry) and day 270 (2 retries).  The retry loop
  # breaks as soon as substep_dump.bin EXISTS, which is a start signal, not a
  # completion signal; a run that died mid-trajectory is therefore NOT retried
  # -- it falls through to the completeness check below and FATALs on the
  # missing final restart.  That is the intended behaviour (a mid-run failure
  # is a real failure and must not be papered over by a retry), and it is
  # written down here because the loop reads like a general retry and is not.
  for try in 1 2 3 4; do
    ( cd "$RUNDIR" && ./nemo > "run_try${try}.log" 2>&1 ) || true
    [ -s "${RUNDIR}/substep_dump.bin" ] && break
    echo "[retry] day ${DAY} (kt=${KT}): launch ${try} died in init, retrying"
  done
  # The final restart proves the run REACHED nn_itend.  Without it, a blowup
  # after substep 1 would still leave a non-empty substep_dump.bin and pass a
  # size-only check -- the dumps are overwritten every step, so their existence
  # proves only that the run STARTED.
  for want in substep_dump.bin spg_dump_puu_b_final.bin spg_dump_zu_frc.bin \
              spg_dump_zv_frc.bin spg_dump_ssh_frc.bin spg_dump_un_adv_final.bin \
              mesh_mask.nc "$(printf 'DINO_%08d_restart.nc' $((KT+4)))"; do
    [ -s "${RUNDIR}/${want}" ] || { echo "FATAL: kt=${KT} produced no ${want}" >&2; exit 1; }
  done
  echo "[done] day ${DAY} (kt=${KT}) -> ${RUNDIR}"
done
echo "ALL LANES BUILT"
