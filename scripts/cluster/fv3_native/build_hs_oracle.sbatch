#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=build_hs
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/build_hs_%j.log
# Held-Suarez physics-coupling oracle: the certified run_hydro_1step_gfs deck
# (hydrostatic, GFS constants) with adiabatic .true.->.false. AND
# do_Held_Suarez added .true., nothing else. fv_phys runs whenever
# adiabatic=.false. (solo/atmosphere.F90:474); its Held-Suarez tendencies are
# gated on flagstruct%do_Held_Suarez (solo/fv_phys.F90:533) -- a namelist flag.
# Published ATOMICALLY: stage -> assert from the run's OWN logfile -> rename.
set -uo pipefail
MPI=/burg/opt/openmpi-4.1.1_ucx_cuda_11.0.3_support
export PATH=$MPI/bin:$PATH
export LD_LIBRARY_PATH=$MPI/lib:/burg/opt/netcdf-fortran-4.6.2/lib:/burg/opt/netcdf-c-4.9.3/lib:${LD_LIBRARY_PATH:-}
PIN=/burg-archive/glab/users/pg2328/fv3_oracle_pinned
EXE=${EXE:-$PIN/build_hydro_gfsconst/fv3_solo_hydro.exe}
[ -x "$EXE" ] || { echo "PREFLIGHT FAILED: no exe at $EXE"; exit 5; }
if ldd "$EXE" | grep -q "not found"; then echo "PREFLIGHT: unresolved libs"; ldd "$EXE" | grep "not found"; exit 6; fi
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
FLIP=$REPO/scripts/validate/fv3_native/flip_adiabatic_flag.py
echo "host: $(hostname)"; echo "exe: $EXE"

SRC=$PIN/run_hydro_1step_gfs
W=$PIN/run_hs_1step_gfs
STAGE=${W}.staging.$$
[ -d "$SRC" ] || { echo "PREFLIGHT: missing $SRC"; exit 5; }
if [ -e "$W/PROVENANCE" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "REFUSING to overwrite $W (carries PROVENANCE); FORCE=1 to override"; exit 9; fi
rm -rf "$STAGE"; mkdir -p "$STAGE/RESTART" || exit 8
cp -r "$SRC/INPUT" "$STAGE/" || { rm -rf "$STAGE"; exit 8; }
for f in input.nml diag_table field_table; do
  /bin/cp -f "$SRC/$f" "$STAGE/" || { rm -rf "$STAGE"; exit 8; }
done
cd "$STAGE" || { rm -rf "$STAGE"; exit 8; }

# TWO edits, both checked: flip adiabatic (tested set_key module) and ADD
# do_Held_Suarez into &fv_core_nml (set_key only CHANGES existing keys).
python3 "$FLIP" input.nml || { echo "EDIT FAILED (adiabatic)"; cd /; rm -rf "$STAGE"; exit 8; }
python3 - <<'PY' || { echo "EDIT FAILED (do_Held_Suarez)"; cd /; rm -rf "$STAGE"; exit 8; }
import re, sys
t = open("input.nml").read()
if re.search(r'^\s*do_Held_Suarez', t, re.M):
    raise SystemExit("do_Held_Suarez already present -- unexpected base deck")
t2 = re.sub(r'(&fv_core_nml\s*\n)', r'\1       do_Held_Suarez = .true.\n', t, count=1)
if t2 == t or 'do_Held_Suarez = .true.' not in t2:
    raise SystemExit("insert did not take")
open("input.nml", "w").write(t2)
PY
grep -qiE "^\s*adiabatic\s*=\s*\.false\." input.nml || { echo "assert: adiabatic not false in nml"; cd /; rm -rf "$STAGE"; exit 8; }
grep -qiE "^\s*do_Held_Suarez\s*=\s*\.true\." input.nml || { echo "assert: HS not true in nml"; cd /; rm -rf "$STAGE"; exit 8; }

echo "=== RUN HS 1step (np=6) ==="
mpirun -np 6 $EXE > run_out.txt 2>&1; RC=$?
echo "run rc=$RC"
FAIL=""
[ "$RC" -eq 0 ] || FAIL="mpirun rc=$RC"
for t in 1 2 3 4 5 6; do [ -s "RESTART/fv_core.res.tile${t}.nc" ] || FAIL="$FAIL; missing core tile$t"; done
if [ -s logfile.000000.out ]; then
  grep -q "FMSConstants: GFS" logfile.000000.out || FAIL="$FAIL; not GFS constants"
  grep -qiE "^ *ADIABATIC *= *F" logfile.000000.out || FAIL="$FAIL; adiabatic not F"
  grep -qiE "^ *HYDROSTATIC *= *T" logfile.000000.out || FAIL="$FAIL; hydrostatic not T"
else
  FAIL="$FAIL; no logfile"
fi
if [ -n "$FAIL" ]; then echo "RUN FAILED ($FAIL):"; head -8 run_out.txt; cd /; rm -rf "$STAGE"; exit 7; fi

{
  echo "generated_by  scripts/cluster/fv3_native/build_hs_oracle.sbatch"
  echo "slurm_job     ${SLURM_JOB_ID:-none}"
  echo "source_deck   $SRC"
  echo "edits         adiabatic .true.->.false. ; do_Held_Suarez added .true."
  echo "exe           $EXE"
  echo "built_utc     see slurm log timestamp"
} > "$STAGE/PROVENANCE"
cd /
rm -rf "$W"; mv "$STAGE" "$W" || { echo "PUBLISH mv failed"; exit 8; }
echo "PUBLISHED $W"
echo "BUILD_HS_DONE"
