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
TWIN="${DINO}/RUN_90D_TWIN"
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
  RUNDIR="${DINO}/RUN_SEQDUMP_D${DAY}_1R"
  BASE=$(printf "DINO_%08d_restart" "$KT")
  if [ -s "${RUNDIR}/substep_dump.bin" ]; then
    echo "[skip] day ${DAY} (kt=${KT}): ${RUNDIR}/substep_dump.bin present"
    continue
  fi

  # ---- 1. stitch the 16 tiles -------------------------------------------------
  if [ ! -s "${STAGE}/${BASE}.nc" ]; then
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
  grep -q "cn_ocerst_in = \"${BASE}\"" "${RUNDIR}/namelist_cfg" \
    || { echo "FATAL: cn_ocerst_in substitution missed for kt=${KT}" >&2; exit 1; }
  ln -sf "../REBUILD_TWIN/${BASE}.nc" "${RUNDIR}/"
  ln -sf "$BIN" "${RUNDIR}/nemo"

  # ---- 3. run the instrumented binary ----------------------------------------
  # Retries: this binary intermittently segfaults during INITIALISATION (in
  # dia_obs_init/dia_detide_init, before any dump is written or any physics is
  # stepped) and runs clean on an immediate retry from the same directory --
  # observed at day 260 (1 retry) and day 270 (2 retries).  Retrying is safe
  # precisely because the completeness check below is on the DUMPS, not on the
  # exit code: a partial run cannot pass it, and a crash this early cannot
  # leave a half-written dump behind.
  for try in 1 2 3 4; do
    ( cd "$RUNDIR" && ./nemo > "run_try${try}.log" 2>&1 ) || true
    [ -s "${RUNDIR}/substep_dump.bin" ] && break
    echo "[retry] day ${DAY} (kt=${KT}): launch ${try} died in init, retrying"
  done
  for want in substep_dump.bin spg_dump_puu_b_final.bin spg_dump_zu_frc.bin \
              spg_dump_zv_frc.bin spg_dump_ssh_frc.bin spg_dump_un_adv_final.bin \
              mesh_mask.nc; do
    [ -s "${RUNDIR}/${want}" ] || { echo "FATAL: kt=${KT} produced no ${want}" >&2; exit 1; }
  done
  echo "[done] day ${DAY} (kt=${KT}) -> ${RUNDIR}"
done
echo "ALL LANES BUILT"
