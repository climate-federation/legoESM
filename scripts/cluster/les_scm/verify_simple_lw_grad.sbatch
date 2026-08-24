#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=lwgrad
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/lwgrad_%j.out
set +e
# Acceptance for the simple_lw AD fix. Three things, in the order that makes a
# partial run still worth reading.
#
#  1. NON-VACUITY: the new tests must FAIL at the pre-fix pin. A regression
#     test for a NaN that passes either way is decoration.
#  2. They pass at the post-fix pin, along with the kernel's existing suite.
#  3. The arms that actually failed -- astex and dycoms -- now tune.
: "${OLD_WT:?}"; : "${NEW_WT:?}"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEW_TESTS="tests/atmosphere/test_simple_lw.py::test_gradient_through_the_COLUMN_GEOMETRY_is_finite
           tests/atmosphere/test_simple_lw.py::test_gradient_through_the_STATE_is_finite
           tests/atmosphere/test_simple_lw.py::test_gradient_through_density_is_finite"
echo "OLD $(git -C "$OLD_WT" rev-parse --short HEAD)  NEW $(git -C "$NEW_WT" rev-parse --short HEAD)"

echo "=============== 1. NON-VACUITY at the pre-fix pin ==============="
cd "$OLD_WT" || exit 1
export PYTHONPATH=$OLD_WT/src:$(ls -d "$OLD_WT"/packages/*/ | tr '\n' ':')
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
# The tests do not exist at the old pin, so copy just that file in. The KERNEL
# stays the old one, which is the variable under test.
cp -f "$NEW_WT/tests/atmosphere/test_simple_lw.py" tests/atmosphere/test_simple_lw.py
$PY -m pytest -q $NEW_TESTS 2>&1 | tail -12
echo "pre-fix rc=${PIPESTATUS[0]} (expect NON-ZERO)"
git checkout -- tests/atmosphere/test_simple_lw.py
git status --porcelain | head -3

echo "=============== 2. the kernel's full suite at the post-fix pin ==============="
cd "$NEW_WT" || exit 1
export PYTHONPATH=$NEW_WT/src:$(ls -d "$NEW_WT"/packages/*/ | tr '\n' ':')
$PY -m pytest -q tests/atmosphere/test_simple_lw.py 2>&1 | tail -12
echo "post-fix rc=${PIPESTATUS[0]} (expect 0)"

echo "=============== 3. the two arms that failed, end to end ==============="
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
R=/burg-archive/glab/users/pg2328/legoESM/results
OUT=$R/scm_les_turbulence/_lwgrad
rm -rf "$OUT"
for c in astex dycoms; do
  echo "--- $c ---"
  $PY scripts/run/run_scm_les_turbulence_tuning.py \
      --cases "$c:$R/les_ref/$c" --outdir "$OUT/$c" \
      --nlev 48 --steps 2 --schemes louis \
      --joint-aggregation default_relative --surface-flux-to-closure \
      2>&1 | grep -E "joint|step|status|FAILED" | tail -6
  $PY - "$OUT/$c/tuned_parameters.json" <<'PYEOF'
import json, sys
p = json.load(open(sys.argv[1]))
for s in p["schemes"]:
    fr = s.get("frozen") or {}
    nf = [k for k, v in fr.items() if "non-finite" in v]
    print(f"  {s['scheme']}: status={s['status']} trained={s['n_trained']} "
          f"nonfinite_grad={len(nf)}")
PYEOF
done
echo "DONE $(date +%T)"
