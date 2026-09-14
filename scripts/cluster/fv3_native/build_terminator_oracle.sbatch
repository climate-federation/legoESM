#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=termorc
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/terminator_oracle_%j.log
# THE MULTI-TRACER ORACLE ON THE CERTIFIED COLD-START PATH. The pinned
# deck's second tracer (liq_wat) is zero everywhere, so scoring it proves
# nothing; the warm-start route (build_warm_tracer_oracle.sbatch) left the
# certified path by ~1e-3 after one step for a reason not yet found.
# test_cases.F90:6746-6750 fills the DCMIP16 TERMINATOR pair cl/cl2 --
# nonzero, longitude-dependent, Cl + 2 Cl2 = qcly -- on the SAME cold
# start whenever the field_table lists them; the reaction lives in fv_phys
# and never runs on this adiabatic deck. So: two cold decks that differ
# from run_hydro_{zerostep,1step}_gfs by two field_table entries inserted
# BEFORE rainwat (dnats=1 keeps rainwat the one inert tracer):
#   run_hydro_zerostep_term_gfs   the IC, in RESTART/, as the harness reads it
#   run_hydro_1step_term_gfs      one step
# Published only after: the 1-step deck's u/v/T/delp equal the pinned
# cold-start deck's BITWISE (passengers feed back nothing); sphum is
# compared and its result RECORDED (the terminator branch adds an init-time
# halo update of every tracer -- whether that moves sphum is a finding,
# not an assumption); cl/cl2 nonzero and moved; Cl + 2 Cl2 stays qcly.
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
  # ONE edit to the field_table: cl and cl2 inserted before rainwat, so
  # the dnats=1 tail is still rainwat alone.  Asserted below from the
  # run's own tracer restart, not from this sed.
  [ "$(grep -c '"rainwat"' field_table)" = 1 ] || { echo "field_table: rainwat entry not unique"; exit 8; }
  "$PY" - <<'PYEOF' || exit 8
import re
s = open("field_table").read()
block = ''' "TRACER", "atmos_mod", "cl"
           "longname",     "terminator Cl"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=0.0" /
 "TRACER", "atmos_mod", "cl2"
           "longname",     "terminator Cl2"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=0.0" /
'''
i = s.index(' "TRACER", "atmos_mod", "rainwat"')
open("field_table", "w").write(s[:i] + block + s[i:])
PYEOF
  [ "$(grep -cE '^ *minutes *= *32' input.nml)" = 1 ] || { echo "nml: minutes line not unique"; exit 8; }
  sed -i "s/^\( *minutes *= *\)32/\1$MIN/" input.nml || exit 8
  echo "=== RUN $1 (cold start, minutes=$MIN, field_table + cl, cl2) ==="
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
  else
    FAIL="$FAIL; no logfile"
  fi
  # the tracer restart must carry exactly sphum, liq_wat, cl, cl2, rainwat in that order
  ORDER=$("$PY" -c "import netCDF4; ds=netCDF4.Dataset('RESTART/fv_tracer.res.tile1.nc'); print(','.join(v for v in ds.variables if ds.variables[v].ndim==4))" 2>/dev/null)
  [ "$ORDER" = "sphum,liq_wat,cl,cl2,rainwat" ] || FAIL="$FAIL; tracer order is '$ORDER'"
  if [ -n "$FAIL" ]; then
    echo "RUN FAILED for $1 ($FAIL):"; head -20 run_out.txt; grep -iE 'error|fatal' logfile.000000.out | head; cd /; unstage_all; rm -rf "$STAGE"; exit 7
  fi
  {
    echo "generated_by  scripts/cluster/fv3_native/build_terminator_oracle.sbatch"
    echo "slurm_job     ${SLURM_JOB_ID:-none}"
    echo "source_deck   $SRC"
    echo "patch         field_table += cl, cl2 before rainwat; minutes 32 -> $MIN (sole edits, asserted)"
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

run_deck run_hydro_zerostep_term_gfs 0
run_deck run_hydro_1step_term_gfs 32

echo "=== CONTROL 1: passengers change no dynamics -- 1-step term deck == pinned cold 1-step deck ==="
"$PY" "$CMP" --a "$PIN/run_hydro_1step_term_gfs.staging.$$" --b "$SRC" --file fv_core.res --vars u,v,T,delp || { unstage_all; exit 10; }
echo "=== RECORD: does the terminator branch's init-time halo update move sphum? (finding, not gate) ==="
"$PY" "$CMP" --a "$PIN/run_hydro_1step_term_gfs.staging.$$" --b "$SRC" --file fv_tracer.res --vars sphum,liq_wat --max-rel inf
echo "=== CONTROL 2: the IC deck's dynamics are the pinned zero-step IC ==="
"$PY" "$CMP" --a "$PIN/run_hydro_zerostep_term_gfs.staging.$$" --b "$PIN/run_hydro_zerostep_consv_gfs" --file fv_core.res --vars u,v,T,delp || { unstage_all; exit 10; }
echo "=== SIGNAL: cl, cl2 nonzero and moved; sphum moved ==="
"$PY" "$CMP" --a "$PIN/run_hydro_zerostep_term_gfs.staging.$$" --b "$PIN/run_hydro_1step_term_gfs.staging.$$" --file fv_tracer.res --vars sphum,cl,cl2 --moved sphum,cl,cl2 --max-rel inf || { unstage_all; exit 10; }
echo "=== CONSERVATION: Cl + 2 Cl2 == qcly on every tile at the IC (gate); after the step (RECORD: limited transport is not linear, and this deviation is the DCMIP terminator diagnostic itself) ==="
STAGE_PID=$$ "$PY" - <<'PYEOF' || { unstage_all; exit 10; }
import netCDF4, numpy as np, sys
sys.path.insert(0, "/burg-archive/glab/users/pg2328/legoESM_codex_m8/packages/core")
from legoesm.core.fv3_native_dcmip16_ic import TERM_QCLY
P = "/burg-archive/glab/users/pg2328/fv3_oracle_pinned"
import os; PID = os.environ["STAGE_PID"]
res = {}
for deck in ("run_hydro_zerostep_term_gfs", "run_hydro_1step_term_gfs"):
    worst = 0.0
    any_cl = False
    for t in range(1, 7):
        with netCDF4.Dataset(f"{P}/{deck}.staging.{PID}/RESTART/fv_tracer.res.tile{t}.nc") as ds:
            cl = np.asarray(ds["cl"][0]); cl2 = np.asarray(ds["cl2"][0])
            # a tile wholly on the night side has cl == 0 (tile 5 at C48);
            # cl2 is then qcly/2 there, never zero
            assert cl2.max() > 0, (deck, t)
            any_cl = any_cl or cl.max() > 0
            worst = max(worst, float(np.abs(cl + 2 * cl2 - TERM_QCLY).max() / TERM_QCLY))
    assert any_cl, f"{deck}: cl is zero on every tile"
    res[deck] = worst
    print(f"  {deck}: worst |Cl + 2 Cl2 - qcly| / qcly over 6 tiles = {worst:.3e}")
ok = res["run_hydro_zerostep_term_gfs"] < 1e-12
print("IC CONSERVATION", "OK" if ok else "VIOLATED", "; post-step deviation recorded above")
sys.exit(0 if ok else 1)
PYEOF
publish_all
echo TERMINATOR_ORACLE_DONE
