#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=ksplitorc
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/ksplit_oracle_%j.log
# THE k_split = 2 ORACLE. Every pinned deck runs k_split = 1; the port
# accepts k_split > 1 and nothing compares it to the oracle. ONE namelist
# edit on the certified cold-start deck: k_split 1 -> 2 (the acoustic
# loop runs twice per step at half the time step, n_split = 8 each).
# The zero-step twin is the IC and must equal the pinned zero-step IC
# bitwise (k_split does not enter the IC); the one-step deck is the
# oracle row. Published only after those hold and the run's own log
# resolves K_SPLIT=2 (asserted from the logfile, not from this sed).
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
CMP=$REPO/scripts/validate/fv3_native/compare_oracle_restarts.py
[ -x "$EXE" ] || { echo "PREFLIGHT FAILED: no exe at $EXE"; exit 5; }
ldd "$EXE" | grep -q "not found" && { echo "PREFLIGHT FAILED: unresolved libs"; ldd "$EXE" | grep "not found"; exit 6; }
[ -f "$SRC/field_table" ] && [ -f "$CMP" ] || { echo "PREFLIGHT FAILED: missing inputs"; exit 5; }
cd "$REPO"
echo "host: $(hostname)  exe: $EXE  repo sha: $(git rev-parse HEAD) dirty: $(git status --porcelain | wc -l)"

STAGED=()
run_deck () {  # name minutes
  local W=$PIN/$1 MIN=$2 STAGE
  STAGE=$PIN/$1.staging.$$
  if [ -e "$W/PROVENANCE" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "REFUSING to overwrite $W (carries PROVENANCE); FORCE=1 to override"; exit 9
  fi
  rm -rf "$STAGE"; mkdir -p "$STAGE/RESTART" "$STAGE/INPUT" || exit 8
  cp "$SRC"/{input.nml,diag_table,field_table} "$STAGE/" || exit 8
  cd "$STAGE" || exit 8
  # ONE edit to the namelist: k_split 1 -> 2, asserted unique here and
  # resolved from the run's own logfile below.
  [ "$(grep -cE '^ *k_split *= *1' input.nml)" = 1 ] || { echo "nml: k_split line not unique"; exit 8; }
  sed -i 's/^\( *k_split *= *\)1/\12/' input.nml || exit 8
  [ "$(grep -cE '^ *minutes *= *32' input.nml)" = 1 ] || { echo "nml: minutes line not unique"; exit 8; }
  sed -i "s/^\( *minutes *= *\)32/\1$MIN/" input.nml || exit 8
  echo "=== RUN $1 (cold start, minutes=$MIN, k_split=2) ==="
  mpirun -np 6 "$EXE" > run_out.txt 2>&1; local RC=$?
  local FAIL=""
  [ "$RC" -eq 0 ] || FAIL="mpirun rc=$RC"
  for t in 1 2 3 4 5 6; do
    [ -s "RESTART/fv_core.res.tile${t}.nc" ] || FAIL="$FAIL; missing core tile$t"
    [ -s "RESTART/fv_tracer.res.tile${t}.nc" ] || FAIL="$FAIL; missing tracer tile$t"
  done
  if [ -s logfile.000000.out ]; then
    grep -q "FMSConstants: GFS" logfile.000000.out || FAIL="$FAIL; not GFS constants"
    grep -qiE "^ *WARM_START *= *F" logfile.000000.out || FAIL="$FAIL; warm_start not F"
    grep -qiE "^ *ADIABATIC *= *T" logfile.000000.out || FAIL="$FAIL; adiabatic not T"
    grep -qiE "^ *K_SPLIT *= *2" logfile.000000.out || FAIL="$FAIL; log does not resolve K_SPLIT=2"
  else
    FAIL="$FAIL; no logfile"
  fi
  # the tracer restart must carry the pinned deck's three tracers in order
  ORDER=$("$PY" -c "import netCDF4; ds=netCDF4.Dataset('RESTART/fv_tracer.res.tile1.nc'); print(','.join(v for v in ds.variables if ds.variables[v].ndim==4))" 2>/dev/null)
  [ "$ORDER" = "sphum,liq_wat,rainwat" ] || FAIL="$FAIL; tracer order is '$ORDER'"
  if [ -n "$FAIL" ]; then
    echo "RUN FAILED for $1 ($FAIL):"; head -20 run_out.txt; grep -iE 'error|fatal' logfile.000000.out | head; cd /; unstage_all; rm -rf "$STAGE"; exit 7
  fi
  {
    echo "generated_by  scripts/cluster/fv3_native/build_ksplit_oracle.sbatch"
    echo "slurm_job     ${SLURM_JOB_ID:-none}"
    echo "source_deck   $SRC"
    echo "patch         k_split 1 -> 2; minutes 32 -> $MIN (sole edits, asserted)"
    echo "tracer_order  $ORDER"
    echo "exe           $EXE"
    echo "exe_sha256    $(sha256sum "$EXE" | cut -d' ' -f1)"
    echo "repo_sha      $(git -C "$REPO" rev-parse HEAD)"
    grep -i "FMSConstants" logfile.000000.out | head -1
    grep -iE "^ *(WARM_START|HYDROSTATIC|ADIABATIC|NPZ|K_SPLIT|N_SPLIT|NWAT|DNATS|CONSV_TE|DO_TERMINATOR) " logfile.000000.out | head -12
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

run_deck run_hydro_zerostep_ksplit2_gfs 0
run_deck run_hydro_1step_ksplit2_gfs 32

echo "=== CONTROL: the IC deck's state is the pinned zero-step IC (k_split does not enter the IC) ==="
"$PY" "$CMP" --a "$PIN/run_hydro_zerostep_ksplit2_gfs.staging.$$" --b "$PIN/run_hydro_zerostep_consv_gfs" --file fv_core.res --vars u,v,T,delp || { unstage_all; exit 10; }
"$PY" "$CMP" --a "$PIN/run_hydro_zerostep_ksplit2_gfs.staging.$$" --b "$PIN/run_hydro_zerostep_consv_gfs" --file fv_tracer.res --vars sphum || { unstage_all; exit 10; }
echo "=== SIGNAL: the k_split=2 step differs from the k_split=1 step (a no-op edit would be vacuous) and moved the state ==="
"$PY" "$CMP" --a "$PIN/run_hydro_1step_ksplit2_gfs.staging.$$" --b "$SRC" --file fv_core.res --vars u,v,T,delp --moved u,v,T,delp --max-rel inf || { unstage_all; exit 10; }
"$PY" "$CMP" --a "$PIN/run_hydro_zerostep_ksplit2_gfs.staging.$$" --b "$PIN/run_hydro_1step_ksplit2_gfs.staging.$$" --file fv_core.res --vars u,v,T,delp --moved u,v,T,delp --max-rel inf || { unstage_all; exit 10; }
publish_all
echo KSPLIT_ORACLE_DONE
