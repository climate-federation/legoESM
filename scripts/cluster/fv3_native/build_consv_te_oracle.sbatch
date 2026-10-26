#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=consvorc
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --mem=24G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/consv_te_oracle_%j.log
# DOES AN ORACLE FOR THE ENERGY FIXER EXIST? I claimed it did not --
# "no deck resolves consv_te non-zero, so there is no oracle" -- which
# was true of the SHIPPED decks and false as a statement about what is
# possible, now that deck generation is a tested tool.
#
# ONE VARIABLE: the certified hydrostatic deck with consv_te 0.0 -> 1.0.
# That is the only edit, asserted. If the run succeeds and its E_Flux /
# dtmp are non-zero, the fixer engaged and the port has something to be
# scored against; if consv_te silently stays under consv_min, it did
# not and the claim stands.
set -uo pipefail
MPI=/burg/opt/openmpi-4.1.1_ucx_cuda_11.0.3_support
export PATH=$MPI/bin:$PATH
export LD_LIBRARY_PATH=$MPI/lib:/burg/opt/netcdf-fortran-4.5.3/lib:/burg/opt/netcdf-c-4.7.4/lib:/cm/shared/apps/hdf5/1.10.1/lib:${LD_LIBRARY_PATH:-}
PIN=/burg-archive/glab/users/pg2328/fv3_oracle_pinned
EXE=${EXE:-$PIN/build_hydro_gfsconst/fv3_solo_hydro.exe}
[ -x "$EXE" ] || { echo "PREFLIGHT FAILED: no exe at $EXE"; exit 5; }
if ldd "$EXE" | grep -q "not found"; then
  echo "PREFLIGHT FAILED: unresolved libraries"; ldd "$EXE" | grep "not found"; exit 6
fi
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
SETK=$REPO/scripts/validate/fv3_native/flip_adiabatic_flag.py
[ -f "$SETK" ] || { echo "PREFLIGHT FAILED: no $SETK"; exit 5; }
echo "host: $(hostname)"; echo "exe: $EXE"

CONSV=${CONSV:-1.0}
for ARM in zerostep 1step; do
  SRC=$PIN/run_hydro_${ARM}_gfs
  [ -d "$SRC" ] || SRC=$PIN/run_hydro_zerostep       # the dry IC deck has no _gfs suffix
  W=$PIN/run_hydro_${ARM}_consv_gfs
  STAGE=${W}.staging.$$
  [ -d "$SRC" ] || { echo "PREFLIGHT FAILED: missing $SRC"; exit 5; }
  if [ -e "$W/PROVENANCE" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "REFUSING to overwrite $W (carries PROVENANCE). FORCE=1 to override."; exit 9
  fi
  rm -rf "$STAGE"; mkdir -p "$STAGE/RESTART" || exit 8
  cp -r "$SRC/INPUT" "$STAGE/" || { rm -rf "$STAGE"; exit 8; }
  for f in input.nml diag_table field_table; do
    /bin/cp -f "$SRC/$f" "$STAGE/" || { rm -rf "$STAGE"; exit 8; }
  done
  cd "$STAGE" || { rm -rf "$STAGE"; exit 8; }
  python3 "$SETK" input.nml consv_te "$CONSV" 0.0 \
    || { echo "EDIT FAILED"; cd /; rm -rf "$STAGE"; exit 8; }

  echo "=== RUN $ARM consv_te=$CONSV (np=6) ==="
  mpirun -np 6 $EXE > run_out.txt 2>&1
  RC=$?; echo "run rc=$RC"
  FAIL=""
  [ "$RC" -eq 0 ] || FAIL="mpirun rc=$RC"
  for t in 1 2 3 4 5 6; do
    [ -s "RESTART/fv_core.res.tile${t}.nc" ] || FAIL="$FAIL; missing core tile$t"
  done
  if [ -s logfile.000000.out ]; then
    grep -q "FMSConstants: GFS" logfile.000000.out || FAIL="$FAIL; not GFS constants"
    grep -qiE "^ *CONSV_TE *= *${CONSV%.*}" logfile.000000.out || echo "NOTE: consv_te not echoed as ${CONSV}"
  else
    FAIL="$FAIL; no logfile"
  fi
  if [ -n "$FAIL" ]; then
    echo "RUN FAILED for $ARM ($FAIL)"; head -5 run_out.txt
    cd /; rm -rf "$STAGE"; exit 7
  fi
  echo "--- resolved ---"
  grep -iE "^ *(CONSV_TE|ADIABATIC|HYDROSTATIC|NPZ|K_SPLIT|N_SPLIT) |FMSConstants" logfile.000000.out | head -8
  # DID THE FIXER ENGAGE? E_Flux is dtmp scaled; a non-zero value means
  # `consv > consv_min` was taken and te_2d/dtmp are live.
  echo "--- fixer evidence ---"
  grep -iE "E_Flux|Total Energy|dtmp" run_out.txt | head -5 || echo "(no E_Flux line printed)"
  {
    echo "generated_by  scripts/cluster/fv3_native/build_consv_te_oracle.sbatch"
    echo "slurm_job     ${SLURM_JOB_ID:-none}"
    echo "source_deck   $SRC"
    echo "patch         consv_te 0.0 -> $CONSV (sole edit, asserted)"
    echo "exe_sha256    $(sha256sum $EXE | cut -d' ' -f1)"
    grep -i "FMSConstants" logfile.000000.out | head -1
    grep -iE "^ *(CONSV_TE|ADIABATIC|HYDROSTATIC|NPZ|K_SPLIT|N_SPLIT) " logfile.000000.out | head -8
  } > PROVENANCE
  cd "$PIN"; rm -rf "$W" && mv "$STAGE" "$W"
  echo "--- published $W ---"
done
echo CONSV_TE_ORACLE_DONE
