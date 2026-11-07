#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scm_twins
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scm_twins_%j.out

# Ocean column twins: legoESM dynamics-free SCM vs NEMO ORCA1 columns at the
# 4 diagnostic points (siberian_shelf / barents / nh_midlat / subtrop_pac),
# tiers T0 (no dynamics: f_c=0, no wind) + T1 (Ekman), vmix kpp (run
# `--vmix kpp,constant` manually for the A/B), 90 days from the NEMO woce
# January IC under CORE-II NYF NCAR bulk fluxes.
#
# CAVEAT (also printed by the driver on every output): the referee is NEMO's
# 3-D monthly output — those columns INCLUDE advection (+ sea ice, SSS
# restoring, runoff) that the SCM does not; a NEMO C1D twin comes later.
#
# Job 8889618 postmortem (4 h walltime, CANCELLED, zero driver output):
#   1. stdout was piped through `tail -120`, which buffers EVERYTHING until
#      process exit, and python ran without -u — no progress was visible at
#      all.  Now: `python -u`, NO output pipe (stdout lands in this file
#      live), and the driver prints a per-simulated-day progress line plus a
#      day-1 steps/s self-check (loud RETRACE-SUSPECTED warning < 5 steps/s).
#   2. the per-step eager loop re-traced + re-compiled the implicit-solver
#      lax.fori_loop closures EVERY step (~seconds/step).  The driver now
#      drives ONE jitted step per run (record index = traced arg).
#   3. the job kept integrating after tests_rc=1.  Now: red tests exit
#      immediately, the driver never starts.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
echo "=== HOST ==="; hostname; date
echo "=== worktree ==="; git -C $WT log --oneline -1

echo "=== unit tests: tests/ocean/unit/test_scm_column_twins.py ==="
$PY -u -m pytest tests/ocean/unit/test_scm_column_twins.py -q
tests_rc=$?
echo "tests_rc=$tests_rc"
if [ "$tests_rc" -ne 0 ]; then
  echo "=== unit tests FAILED — skipping the driver (gate) ==="; date
  exit 1
fi

echo "=== SCM column twins: 4 points x {T0,T1} x {kpp} ==="
$PY -u scripts/validate/ocean_fidelity/run_scm_column_twins.py \
  --days 90 --dt 3600 --start-month 1 \
  --vmix kpp --tiers T0,T1 \
  --sw-mode penetrate --convection none \
  --output-dir results/omip_nemo/scm_twins
twins_rc=$?
echo "twins_rc=$twins_rc"

echo "=== outputs ==="
ls -la results/omip_nemo/scm_twins/ 2>/dev/null | head -50
echo "=== DONE ==="; date
exit $twins_rc
