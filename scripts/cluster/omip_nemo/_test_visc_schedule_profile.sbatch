#!/bin/bash
#SBATCH --job-name=viscprof
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/viscprof_%j.out
#
# Does the viscosity schedule still halve the viscosity the profile file names?
#
# The test file carries its own vacuity proof (test_reverting_the_fix_halves_it
# asserts the PRE-FIX expression fails the invariant), so a green run here is
# evidence the assertion can fail, not only that it passed.
#
# x64 because the invariant is an exact product identity -- at float32 the
# rescale-then-multiply round trip would leave a 1e-7 residue and the test
# would be measuring rounding rather than the fix.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge
cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=== HEAD $(git -C $ROOT log --oneline -1) ==="
$PY -m pytest -q \
    tests/unit/test_run_omip_core2_visc_schedule_profile.py \
    tests/unit/test_run_omip_core2_eq_visc_shaping.py
echo "=== pytest rc=$? ==="
