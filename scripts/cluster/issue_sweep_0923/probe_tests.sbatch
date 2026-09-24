#!/bin/bash
#SBATCH --job-name=sweep0923_t2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/sweep0923_tests.%j.out
# Direct tests for the two probes this sweep added, plus a non-vacuity check on
# the matrix-runner seed override: the AST guard must go RED if the helper is
# bypassed at any one of the four grid lanes.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

echo "########## the new tests ##########"
"$PY" -m pytest -q -rA --tb=short tests/unit/test_issue_sweep_0923_probes.py 2>&1 | tail -60

echo "########## NON-VACUITY: drop the argmax guard, the trap test must go red ##########"
cp scripts/matrix/run_atmosphere_test_matrix.py /tmp/_matrix_argmax_$$.py
python3 - <<'ARGMAXPY'
import pathlib
p = pathlib.Path("scripts/matrix/run_atmosphere_test_matrix.py")
s = p.read_text()
old = ("    if not bool(jnp.isfinite(jnp.max(field))):\n"
       "        return None\n")
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, "", 1))
print("SABOTAGED: argmax all-NaN guard removed")
ARGMAXPY
"$PY" -m pytest -q tests/unit/test_issue_sweep_0923_probes.py -k "argmax or nonfinite" 2>&1 | tail -8
cp -f /tmp/_matrix_argmax_$$.py scripts/matrix/run_atmosphere_test_matrix.py
rm -f /tmp/_matrix_argmax_$$.py
echo "RESTORED argmax guard"
"$PY" -m pytest -q tests/unit/test_issue_sweep_0923_probes.py -k "argmax or nonfinite" 2>&1 | tail -4
echo "########## NON-VACUITY: break one lane, the AST guard must go red ##########"
cp scripts/matrix/run_atmosphere_test_matrix.py /tmp/_matrix_backup_$$.py
python3 - <<'PY'
import pathlib
p = pathlib.Path("scripts/matrix/run_atmosphere_test_matrix.py")
s = p.read_text()
old = "state = held_suarez_topo_init(grid, sigma, **_topo_init_kwargs())"
assert s.count(old) == 1
p.write_text(s.replace(old, "state = held_suarez_topo_init(grid, sigma, h_0=_topo_h0)", 1))
print("SABOTAGED: cube lane bypasses the helper")
PY
"$PY" -m pytest -q tests/unit/test_issue_sweep_0923_probes.py -k routes_through 2>&1 | tail -6
cp -f /tmp/_matrix_backup_$$.py scripts/matrix/run_atmosphere_test_matrix.py
rm -f /tmp/_matrix_backup_$$.py
echo "RESTORED: $(git -C "$WT" diff --stat scripts/matrix/run_atmosphere_test_matrix.py | tail -1)"

echo "########## the guard must be GREEN again after the restore ##########"
"$PY" -m pytest -q tests/unit/test_issue_sweep_0923_probes.py -k routes_through 2>&1 | tail -4
echo DONE
