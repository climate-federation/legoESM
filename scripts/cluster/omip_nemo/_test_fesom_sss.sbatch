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
# the channel, and the neighbouring suites the edits could have broken.
#
# THIS SCRIPT MUST BE ABLE TO FAIL.  Its first revision could not: it had no
# `set -e`, its FATAL only echoed, and its "did anything pass" predicate was
# satisfied by "3 failed, 9 passed".  A green SLURM exit meant nothing.  Every
# section now feeds `fail()`, and the job exits non-zero if any did.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
FJX=/burg-archive/glab/users/pg2328/fesom_jax_repo
echo "fesom_jax: $(git -C $FJX log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$FJX:$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

RC=0
fail() { echo "FAIL[$1]: $2"; RC=1; }

# The FESOM units are worthless if fesom_jax is not importable -- the whole
# file skips and the gate reports success.  Fail loudly instead.
$PY -c "import fesom_jax" || { echo "FATAL: fesom_jax not importable"; exit 1; }

# Run a pytest selection and judge it by pytest's OWN summary line, not by the
# shell status alone: an OOM kill or a collection error can still exit 0.
run_pytest() {   # run_pytest <label> <extra-env> <targets...>
  local label="$1"; shift
  local log=/tmp/fesomsss.$$.$label
  JAX_PLATFORMS=cpu $@ 2>&1 | tee "$log" | tail -25
  local st=${PIPESTATUS[0]}
  grep -qE "[0-9]+ (failed|error)" "$log" && fail "$label" "pytest reported failures/errors"
  grep -qE "[0-9]+ passed" "$log" || fail "$label" "nothing PASSED (all skipped? collection error?)"
  [ "$st" -eq 0 ] || fail "$label" "pytest exited $st"
  rm -f "$log"
}

echo "=== 1. the new FESOM restoring units (x64) ==="
run_pytest s1 env JAX_ENABLE_X64=1 $PY -m pytest -q -p no:randomly \
  tests/ocean/unit/test_sss_apply_fesom.py

echo "=== 2. CLI gate + the siblings this module ships with ==="
run_pytest s2 $PY -m pytest -q -p no:randomly \
  tests/unit/test_run_omip_core2_fesom_cli.py \
  tests/unit/test_run_omip_core2_sss_restoring_cli.py \
  tests/unit/test_sss_apply.py \
  tests/ocean/unit/test_sss_restoring.py

echo "=== 3. neighbouring FESOM suites (regression) ==="
run_pytest s3 $PY -m pytest -q -p no:randomly \
  tests/ocean/unit/test_ocean_model_fesom.py \
  tests/ocean/unit/test_fesom_b4.py

echo "=== 4. the fesom lane REFUSES an unset channel and ACCEPTS both explicit ones ==="
JAX_PLATFORMS=cpu $PY -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('rom', 'scripts/run/run_omip_core2.py')
m = importlib.util.module_from_spec(spec); sys.modules['rom'] = m
spec.loader.exec_module(m)
p = m._build_arg_parser()
BASE = ['--grid','fesom','--fesom-mesh-dir','/some/mesh','--output','/some/out',
        '--no-emp','--sss-restore','--sss-restore-file','/x.nc']

a = p.parse_args(BASE)
assert a.sss_restore_channel is None, a.sss_restore_channel
try:
    m.validate_fesom_stage(a, p)
except SystemExit as e:
    assert 'explicit' in str(e), str(e)
    print('unset channel: REFUSED, as intended')
else:
    raise AssertionError('unset channel was ACCEPTED -- an unasked choice')

for ch in ('tracer', 'water_flux'):
    extra = ['--sss-restore-channel', ch]
    if ch == 'water_flux':
        extra += ['--sss-restore-normalization', 'live_s']
    m.validate_fesom_stage(p.parse_args(BASE + extra), p)
    print(f'{ch}: ACCEPTED')
" 2>&1 | tail -12
[ ${PIPESTATUS[0]} -eq 0 ] || fail s4 "the CLI channel contract is broken"

echo "DONE rc=$RC"
exit $RC
