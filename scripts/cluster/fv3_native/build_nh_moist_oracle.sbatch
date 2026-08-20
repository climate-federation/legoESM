#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=nhmoistorc
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --mem=24G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/nh_moist_oracle_%j.log
# THE MISSING ORACLE: a NON-hydrostatic MOIST duo run, so the NH moist
# arm can be scored the way the hydrostatic one now is.
#
# ONE VARIABLE. The existing run_nh_{zerostep,1step}_gfs decks and the
# certified run_hydro_1step_moist_gfs deck differ in exactly three
# lines -- hydrostatic, phys_hydrostatic, adiabatic -- and their
# field_tables are byte-identical (checked). So the NH moist deck is
# the NH deck with `adiabatic = .true.` -> `.false.`, and nothing else.
# In the solo driver that single flag is what sets zvir = rvgas/rdgas-1
# and moist_phys = .true. (atmosphere.F90:156-161).
#
# The binary is the SAME one the existing NH oracle runs used: the
# no-SW_DYNAMICS, GFS-constants build. Both source decks' logfiles say
# "FMSConstants: GFS" and the parity harness asserts that from the
# run's own log, so a mis-linked binary is caught rather than assumed.
set -uo pipefail
MPI=/burg/opt/openmpi-4.1.1_ucx_cuda_11.0.3_support
export PATH=$MPI/bin:$PATH
export LD_LIBRARY_PATH=$MPI/lib:/burg/opt/netcdf-fortran-4.5.3/lib:/burg/opt/netcdf-c-4.7.4/lib:/cm/shared/apps/hdf5/1.10.1/lib:${LD_LIBRARY_PATH:-}
PIN=/burg-archive/glab/users/pg2328/fv3_oracle_pinned
# build_hydro_gfsconst, NOT build_hydro_serialnc: the existing _gfs runs
# (run_nh_1step_gfs, run_hydro_1step_gfs) report "FMSConstants: GFS" and
# the harness pins that set, while the serialnc/nc474 runs report GFDL.
# gfsconst also links libnetcdf.so.18 (netcdf-c-4.7.4), matching the
# LD_LIBRARY_PATH above; serialnc wants so.19 and died rc=127 on exactly
# that mismatch in job 9444375, leaving a deck with no output.
EXE=${EXE:-$PIN/build_hydro_gfsconst/fv3_solo_hydro.exe}
[ -x "$EXE" ] || { echo "PREFLIGHT FAILED: no exe at $EXE"; exit 5; }
# PREFLIGHT THE DYNAMIC LINK, so a mis-paired exe/lib fails HERE rather
# than as six silent per-rank loader errors inside a redirected log.
if ldd "$EXE" | grep -q "not found"; then
  echo "PREFLIGHT FAILED: unresolved libraries for $EXE"
  ldd "$EXE" | grep "not found"
  exit 6
fi
echo "host: $(hostname)"; echo "exe: $EXE"

for ARM in zerostep 1step; do
  SRC=$PIN/run_nh_${ARM}_gfs
  W=$PIN/run_nh_${ARM}_moist_gfs
  [ -d "$SRC" ] || { echo "PREFLIGHT FAILED: missing $SRC"; exit 5; }
  rm -rf $W && mkdir -p $W/RESTART
  cp -r $SRC/INPUT $W/
  for f in input.nml diag_table field_table; do /bin/cp -f $SRC/$f $W/; done
  cd $W
  python3 - <<'PY'
import re
s = open('input.nml').read()
# The ONE edit. Assert it is the one line and that it was actually
# .true. before -- a silent no-op here would produce a DRY deck with a
# moist name, which is the worst available outcome.
hits = re.findall(r'^\s*adiabatic\s*=\s*\.(\w+)\.', s, re.M)
assert hits == ['true'], f"expected exactly one adiabatic=.true., got {hits}"
s = re.sub(r'(^\s*adiabatic\s*=\s*)\.true\.', r'\1.false.', s, flags=re.M)
assert re.findall(r'^\s*adiabatic\s*=\s*\.(\w+)\.', s, re.M) == ['false']
open('input.nml','w').write(s)
print("adiabatic -> .false.")
PY
  echo "=== RUN $ARM (np=6) ==="
  mpirun -np 6 $EXE > run_out.txt 2>&1
  RC=$?
  echo "run rc=$RC"
  if [ "$RC" -ne 0 ] || [ ! -s RESTART/fv_core.res.tile1.nc ]; then
    echo "RUN FAILED for $ARM -- refusing to leave a named deck with no"
    echo "output, which a later run would read as a valid oracle:"
    head -5 run_out.txt
    cd /; rm -rf $W
    exit 7
  fi
  grep -iE "FATAL|SIGABRT|MPI_ERR" run_out.txt | head -3
  grep -i "FMSConstants" logfile.000000.out | head -1
  grep -iE "^ *(HYDROSTATIC|ADIABATIC|NPZ|K_SPLIT|N_SPLIT|DUOGRID) " logfile.000000.out | head -8
  echo "--- restarts ---"; ls RESTART/ | head -14
  echo "--- state ---"; grep -E "PS\*|U  max|V  max|T  max|sphum" run_out.txt | tail -6
done
echo NH_MOIST_ORACLE_DONE
