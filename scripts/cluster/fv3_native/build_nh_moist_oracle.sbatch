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

# A GENERATED ORACLE IS PUBLISHED ATOMICALLY OR NOT AT ALL.
# The first version ran under `set -uo pipefail` with no `-e`, so a
# failed mkdir/cp/edit fell through into mpirun and could publish a DRY
# deck under a moist name; it validated one tile of six; and its final
# greps printed evidence rather than asserting it (codex BLOCKER, job
# 9444413). Now: stage under a temp name, assert everything from the
# run's OWN logfile and restarts, and only then rename into place.
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
FLIP=$REPO/scripts/validate/fv3_native/flip_adiabatic_flag.py
[ -f "$FLIP" ] || { echo "PREFLIGHT FAILED: no $FLIP"; exit 5; }

for ARM in zerostep 1step; do
  SRC=$PIN/run_nh_${ARM}_gfs
  W=$PIN/run_nh_${ARM}_moist_gfs
  STAGE=${W}.staging.$$
  [ -d "$SRC" ] || { echo "PREFLIGHT FAILED: missing $SRC"; exit 5; }
  if [ -e "$W/PROVENANCE" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "REFUSING to overwrite $W: it carries PROVENANCE, i.e. bits a"
    echo "published number was measured against. FORCE=1 to override."
    exit 9
  fi
  rm -rf "$STAGE"
  mkdir -p "$STAGE/RESTART" || exit 8
  cp -r "$SRC/INPUT" "$STAGE/" || { rm -rf "$STAGE"; exit 8; }
  for f in input.nml diag_table field_table; do
    /bin/cp -f "$SRC/$f" "$STAGE/" || { rm -rf "$STAGE"; exit 8; }
  done
  cd "$STAGE" || { rm -rf "$STAGE"; exit 8; }
  # THE ONE EDIT, in a tested module. Its exit status is CHECKED: an
  # assertion firing here used to be ignored, and the UNEDITED dry deck
  # was run instead.
  python3 "$FLIP" input.nml || { echo "EDIT FAILED"; cd /; rm -rf "$STAGE"; exit 8; }

  echo "=== RUN $ARM (np=6) ==="
  mpirun -np 6 $EXE > run_out.txt 2>&1
  RC=$?
  echo "run rc=$RC"
  FAIL=""
  [ "$RC" -eq 0 ] || FAIL="mpirun rc=$RC"
  for t in 1 2 3 4 5 6; do
    [ -s "RESTART/fv_core.res.tile${t}.nc" ] || FAIL="$FAIL; missing core tile$t"
  done
  if [ -s logfile.000000.out ]; then
    # ASSERT what the run RESOLVED, from its own log, not what we set.
    grep -q "FMSConstants: GFS" logfile.000000.out || FAIL="$FAIL; not GFS constants"
    grep -qiE "^ *ADIABATIC *= *F" logfile.000000.out || FAIL="$FAIL; adiabatic not F"
    grep -qiE "^ *HYDROSTATIC *= *F" logfile.000000.out || FAIL="$FAIL; hydrostatic not F"
    grep -qiE "^ *PHYS_HYDROSTATIC *= *F" logfile.000000.out || FAIL="$FAIL; phys_hydrostatic not F"
  else
    FAIL="$FAIL; no logfile"
  fi
  if [ -n "$FAIL" ]; then
    echo "RUN FAILED for $ARM ($FAIL) -- refusing to publish a deck a"
    echo "later job would read as a valid oracle:"
    head -5 run_out.txt
    cd /; rm -rf "$STAGE"; exit 7
  fi
  # PROVENANCE: which bits a number was measured against, and enough to
  # detect the twins silently diverging from their source deck later.
  {
    echo "generated_by  scripts/cluster/fv3_native/build_nh_moist_oracle.sbatch"
    echo "slurm_job     ${SLURM_JOB_ID:-none}"
    echo "source_deck   $SRC"
    echo "patch         adiabatic .true. -> .false. (sole edit, asserted)"
    echo "exe           $EXE"
    echo "exe_sha256    $(sha256sum $EXE | cut -d' ' -f1)"
    for f in input.nml field_table diag_table; do
      echo "src_${f}_sha256 $(sha256sum $SRC/$f | cut -d' ' -f1)"
    done
    grep -i "FMSConstants" logfile.000000.out | head -1
    grep -iE "^ *(HYDROSTATIC|ADIABATIC|PHYS_HYDROSTATIC|NPZ|K_SPLIT|N_SPLIT|DUOGRID|CONSV_TE|FV_SG_ADJ) " logfile.000000.out | head -12
  } > PROVENANCE
  cd "$PIN"
  rm -rf "$W" && mv "$STAGE" "$W"
  echo "--- published $W ---"; cat "$W/PROVENANCE"
done
echo NH_MOIST_ORACLE_DONE
