#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=ab_prefail
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/ab_prefail_%j.out
set +e
# Are these 16 failures MINE, or were they already red?
#
# The wide hydrostatic unit tier was run because two new PhysicsState fields
# are a change to every consumer of the carry.  It came back 16 failed / 1869
# passed, all in areas the change does not touch (convective autoconversion,
# GWD, RRTMGP aerosol, the C-D grid solver, spectral-PE convection / tracer
# advection / filters).  "Untouched area" is an argument, not a measurement:
# run the SAME files at the parent commit and at HEAD, on ONE node, in ONE
# job, so the only difference is the commit.
#
#   IDENTICAL failure sets  -> pre-existing, not this change (CI has been dark
#                              since 2026-05-27, so drift is expected).
#   MORE failures at HEAD   -> mine; the extra names are the finding.
#   FEWER at HEAD           -> report it, do not celebrate it; something in
#                              this change is masking a failure.
: "${BASE_WT:?set BASE_WT to a worktree at the PARENT commit}"
: "${HEAD_WT:?set HEAD_WT to a worktree at HEAD}"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
FILES="tests/atmosphere/hydrostatic/unit/test_convective_autoconversion_dispatch.py
tests/atmosphere/hydrostatic/unit/test_gwd_composite.py
tests/atmosphere/hydrostatic/unit/test_hines_gwd_faithful.py
tests/atmosphere/hydrostatic/unit/test_rrtmgp_stratosphere.py
tests/atmosphere/hydrostatic/unit/test_semi_implicit_cdgrid.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_flux_form_and_filters.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py"

for ARM in BASE HEAD; do
  WT=$([ "$ARM" = BASE ] && echo "$BASE_WT" || echo "$HEAD_WT")
  cd "$WT" || exit 1
  echo "===== $ARM  sha=$(git rev-parse --short HEAD)  host=$(hostname) ====="
  export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')
  export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
  $PY -m pytest -q $FILES 2>&1 | grep -E "^FAILED|passed|failed" \
      | sort > /tmp/ab_$ARM.$SLURM_JOB_ID
  cat /tmp/ab_$ARM.$SLURM_JOB_ID
done

echo "===== diff BASE vs HEAD (empty = identical failure sets) ====="
diff /tmp/ab_BASE.$SLURM_JOB_ID /tmp/ab_HEAD.$SLURM_JOB_ID
echo "diff rc=$?  (0 = identical, i.e. PRE-EXISTING)"
echo "DONE $(date +%T)"
