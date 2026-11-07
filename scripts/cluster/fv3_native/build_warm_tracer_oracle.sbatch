#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=warmtrorc
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/warm_tracer_oracle_%j.log
# THE MISSING MULTI-TRACER ORACLE. The pinned deck cold-starts from its
# analytic test case, which sets liq_wat = 0 everywhere, so a two-tracer
# comparison against it perturbs a zero. Three WARM-STARTED decks from the
# zero-step restart, one edit each beyond warm_start=.T. and the run length:
#
#   run_hydro_1step_warm_gfs           control: unmodified restart, one step.
#                                      Its fv_core MUST match the cold-start
#                                      run_hydro_1step_gfs -- that is the
#                                      whole warm-start chain, certified.
#   run_hydro_zerostep_warmtracer_gfs  liq_wat := port tracer 1, zero steps:
#                                      the tracer deck's IC, in RESTART/,
#                                      where the parity harness reads it.
#   run_hydro_1step_warmtracer_gfs     the same, one step.  Its u/v/pt/delp
#                                      and sphum MUST equal the control's
#                                      (nwat=0: liq_wat is a passenger with
#                                      no feedback), and liq_wat must move.
#
# Every assertion is against the runs' OWN restarts and logfile; a deck is
# published only after all of them hold (build_nh_moist_oracle.sbatch's
# staging protocol).
set -uo pipefail
MPI=/burg/opt/openmpi-4.1.1_ucx_cuda_11.0.3_support
export PATH=$MPI/bin:$PATH
export LD_LIBRARY_PATH=$MPI/lib:/burg/opt/netcdf-fortran-4.5.3/lib:/burg/opt/netcdf-c-4.7.4/lib:/cm/shared/apps/hdf5/1.10.1/lib:${LD_LIBRARY_PATH:-}
PIN=/burg-archive/glab/users/pg2328/fv3_oracle_pinned
EXE=${EXE:-$PIN/build_hydro_gfsconst/fv3_solo_hydro.exe}
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
SRC=$PIN/run_hydro_1step_gfs
ZS=$PIN/run_hydro_zerostep_consv_gfs
WRITER=$REPO/scripts/validate/fv3_native/write_tracer_oracle_ic.py
CMP=$REPO/scripts/validate/fv3_native/compare_oracle_restarts.py
TRIC=/burg-archive/glab/users/pg2328/fv3_duo_gaps/warm_tracer_ic_${SLURM_JOB_ID}
[ -x "$EXE" ] || { echo "PREFLIGHT FAILED: no exe at $EXE"; exit 5; }
ldd "$EXE" | grep -q "not found" && { echo "PREFLIGHT FAILED: unresolved libs"; ldd "$EXE" | grep "not found"; exit 6; }
for f in "$SRC/input.nml" "$ZS/RESTART/fv_core.res.tile6.nc" "$ZS/RESTART/fv_tracer.res.tile6.nc" "$WRITER" "$CMP"; do
  [ -e "$f" ] || { echo "PREFLIGHT FAILED: missing $f"; exit 5; }
done
cd "$REPO"
echo "host: $(hostname)  exe: $EXE  repo sha: $(git rev-parse HEAD) dirty: $(git status --porcelain | wc -l)"

echo "=== STEP 0: write the modulated liq_wat into a copy of the zero-step tracer restarts ==="
"$PY" "$WRITER" --zerostep-run "$ZS" --out-dir "$TRIC" || { echo "WRITER FAILED"; exit 7; }

STAGED=()
run_deck () {  # name minutes tracer_dir
  local W=$PIN/$1 MIN=$2 TR=$3 STAGE
  STAGE=$PIN/$1.staging.$$
  if [ -e "$W/PROVENANCE" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "REFUSING to overwrite $W (carries PROVENANCE); FORCE=1 to override"; exit 9
  fi
  rm -rf "$STAGE"; mkdir -p "$STAGE/RESTART" "$STAGE/INPUT" || exit 8
  cp "$SRC"/{input.nml,diag_table,field_table} "$STAGE/" || exit 8
  [ -d "$SRC/INPUT" ] && cp -r "$SRC/INPUT/." "$STAGE/INPUT/"
  cp "$ZS"/RESTART/* "$STAGE/INPUT/" || exit 8
  if [ -n "$TR" ]; then cp "$TR"/fv_tracer.res.tile*.nc "$STAGE/INPUT/" || exit 8; fi
  cd "$STAGE" || exit 8
  # exactly one edit each, asserted
  [ "$(grep -cE '^ *warm_start *= *\.F\.' input.nml)" = 1 ] || { echo "nml: warm_start line not unique"; exit 8; }
  sed -i 's/^\( *warm_start *= *\)\.F\./\1.T./' input.nml || exit 8
  [ "$(grep -cE '^ *minutes *= *32' input.nml)" = 1 ] || { echo "nml: minutes line not unique"; exit 8; }
  sed -i "s/^\( *minutes *= *\)32/\1$MIN/" input.nml || exit 8
  echo "=== RUN $1 (warm_start=.T., minutes=$MIN, tracer=${TR:-original}) ==="
  mpirun -np 6 "$EXE" > run_out.txt 2>&1; local RC=$?
  local FAIL=""
  [ "$RC" -eq 0 ] || FAIL="mpirun rc=$RC"
  for t in 1 2 3 4 5 6; do
    [ -s "RESTART/fv_core.res.tile${t}.nc" ] || FAIL="$FAIL; missing core tile$t"
    [ -s "RESTART/fv_tracer.res.tile${t}.nc" ] || FAIL="$FAIL; missing tracer tile$t"
  done
  if [ -s logfile.000000.out ]; then
    grep -q "FMSConstants: GFS" logfile.000000.out || FAIL="$FAIL; not GFS constants"
    grep -qiE "^ *WARM_START *= *T" logfile.000000.out || FAIL="$FAIL; warm_start not T in log"
  else
    FAIL="$FAIL; no logfile"
  fi
  if [ -n "$FAIL" ]; then
    echo "RUN FAILED for $1 ($FAIL):"; head -20 run_out.txt; grep -iE 'error|fatal' logfile.000000.out | head; cd /; unstage_all; rm -rf "$STAGE"; exit 7
  fi
  {
    echo "generated_by  scripts/cluster/fv3_native/build_warm_tracer_oracle.sbatch"
    echo "slurm_job     ${SLURM_JOB_ID:-none}"
    echo "source_deck   $SRC"
    echo "restart_from  $ZS/RESTART"
    echo "tracer_ic     ${TR:-original zero-step fv_tracer.res}"
    echo "patch         warm_start .F. -> .T.; minutes 32 -> $MIN (sole edits, asserted)"
    echo "exe           $EXE"
    echo "exe_sha256    $(sha256sum "$EXE" | cut -d' ' -f1)"
    echo "repo_sha      $(git -C "$REPO" rev-parse HEAD)"
    grep -i "FMSConstants" logfile.000000.out | head -1
    grep -iE "^ *(WARM_START|HYDROSTATIC|ADIABATIC|NPZ|K_SPLIT|N_SPLIT|NWAT|DNATS|CONSV_TE) " logfile.000000.out | head -12
  } > PROVENANCE
  cd "$PIN" || exit 8
  STAGED+=("$1")
  echo "--- staged $1 (publish after the cross-deck controls) ---"
}
publish_all () {  # codex 2026-09-14 (x2): whole family or nothing, and TRANSACTIONAL
  # an existing deck is moved aside, never deleted, until every deck of
  # the family is in place; any mv failure rolls the family back
  local done=()
  for d in "${STAGED[@]}"; do
    if [ -e "$PIN/$d" ]; then
      mv "$PIN/$d" "$PIN/$d.old.$$" || { echo "PUBLISH FAILED moving old $d aside"; rollback "${done[@]}"; exit 11; }
    fi
    if ! mv "$PIN/$d.staging.$$" "$PIN/$d"; then
      echo "PUBLISH FAILED for $d -- rolling the family back"
      [ -e "$PIN/$d.old.$$" ] && mv "$PIN/$d.old.$$" "$PIN/$d"
      rollback "${done[@]}"; exit 11
    fi
    done+=("$d")
    echo "--- published $PIN/$d ---"
  done
  for d in "${STAGED[@]}"; do rm -rf "$PIN/$d.old.$$"; done
}
rollback () {  # undo published decks of this family: staging back, old back
  local d
  for d in "$@"; do
    mv "$PIN/$d" "$PIN/$d.staging.$$" 2>/dev/null
    [ -e "$PIN/$d.old.$$" ] && mv "$PIN/$d.old.$$" "$PIN/$d"
  done
  unstage_all
}
unstage_all () { for d in "${STAGED[@]}"; do rm -rf "$PIN/$d.staging.$$"; done; }

run_deck run_hydro_1step_warm_gfs 32 ""
run_deck run_hydro_zerostep_warmtracer_gfs 0 "$TRIC"
run_deck run_hydro_1step_warmtracer_gfs 32 "$TRIC"

echo "=== CONTROL 1: warm start reproduces the cold start (fv_core + sphum) ==="
"$PY" "$CMP" --a "$PIN/run_hydro_1step_warm_gfs.staging.$$" --b "$SRC" --file fv_core.res --vars u,v,T,delp --max-rel "${WARM_MAX_REL:-0}" || { unstage_all; exit 10; }
"$PY" "$CMP" --a "$PIN/run_hydro_1step_warm_gfs.staging.$$" --b "$SRC" --file fv_tracer.res --vars sphum,liq_wat --max-rel "${WARM_MAX_REL:-0}" || { unstage_all; exit 10; }
echo "=== CONTROL 2: the tracer deck's zero step IS the IC it was given ==="
"$PY" "$CMP" --a "$PIN/run_hydro_zerostep_warmtracer_gfs.staging.$$" --b "$ZS" --file fv_core.res --vars u,v,T,delp || { unstage_all; exit 10; }
echo "=== CONTROL 3: a passenger changes nothing else -- tracer deck dynamics == control ==="
"$PY" "$CMP" --a "$PIN/run_hydro_1step_warmtracer_gfs.staging.$$" --b "$PIN/run_hydro_1step_warm_gfs.staging.$$" --file fv_core.res --vars u,v,T,delp || { unstage_all; exit 10; }
"$PY" "$CMP" --a "$PIN/run_hydro_1step_warmtracer_gfs.staging.$$" --b "$PIN/run_hydro_1step_warm_gfs.staging.$$" --file fv_tracer.res --vars sphum || { unstage_all; exit 10; }
echo "=== SIGNAL: liq_wat is nonzero and moved over the step ==="
"$PY" "$CMP" --a "$PIN/run_hydro_zerostep_warmtracer_gfs.staging.$$" --b "$PIN/run_hydro_1step_warmtracer_gfs.staging.$$" --file fv_tracer.res --vars sphum,liq_wat --moved sphum,liq_wat --max-rel inf || { unstage_all; exit 10; }
publish_all
echo WARM_TRACER_ORACLE_DONE
