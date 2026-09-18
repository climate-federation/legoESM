#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fesomsss
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/fesomsss_%j.out
#
# Gate for the FESOM SSS-restoring tracer channel: the new applicator, the
# surface-salinity setter it writes through, the CLI gate that used to refuse
# the channel, and the two neighbouring suites the edits could have broken.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
FJX=/burg-archive/glab/users/pg2328/fesom_jax_repo
echo "fesom_jax: $(git -C $FJX log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$FJX:$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
# The FESOM units are worthless if fesom_jax is not importable -- the whole
# file skips and the gate reports success.  Fail loudly instead.
$PY -c "import fesom_jax" || { echo "FATAL: fesom_jax not importable"; exit 1; }

echo "=== 1. the new FESOM restoring units (x64) ==="
# -p no:cacheprovider keeps the run read-only-ish; --no-skip is not a pytest
# flag, so the skip count is asserted below instead.
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q -p no:randomly \
  tests/ocean/unit/test_sss_apply_fesom.py 2>&1 | tee /tmp/t1.$$ | tail -25
grep -qE "[0-9]+ passed" /tmp/t1.$$ || { echo "FATAL: section 1 did not PASS anything (all skipped?)"; }
rm -f /tmp/t1.$$

echo "=== 2. CLI gate + the lat-lon/MPAS applicator it shares a module with ==="
JAX_PLATFORMS=cpu $PY -m pytest -q -p no:randomly \
  tests/unit/test_run_omip_core2_fesom_cli.py \
  tests/unit/test_run_omip_core2_sss_restoring_cli.py \
  tests/unit/test_sss_apply.py 2>&1 | tail -25
echo "rc2=$?"

echo "=== 3. neighbouring FESOM suites (regression) ==="
JAX_PLATFORMS=cpu $PY -m pytest -q -p no:randomly \
  tests/ocean/unit/test_ocean_model_fesom.py \
  tests/ocean/unit/test_fesom_b4.py 2>&1 | tail -25
echo "rc3=$?"

echo "=== 4. the driver still imports and its parser still builds ==="
JAX_PLATFORMS=cpu $PY -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('rom', 'scripts/run/run_omip_core2.py')
m = importlib.util.module_from_spec(spec); sys.modules['rom'] = m
spec.loader.exec_module(m)
p = m._build_arg_parser()
a = p.parse_args(['--grid','fesom','--fesom-mesh-dir','/some/mesh',
                  '--output','/some/out','--no-emp',
                  '--sss-restore','--sss-restore-file','/x.nc'])
print('channel default ->', repr(a.sss_restore_channel), '(None MEANS tracer)')
m.validate_fesom_stage(a, p)
print('validate_fesom_stage: ACCEPTED the tracer channel')
" 2>&1 | tail -12
echo "rc4=$?"
echo "DONE"
