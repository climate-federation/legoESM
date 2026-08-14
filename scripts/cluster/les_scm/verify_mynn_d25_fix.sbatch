#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=mynnfix
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/mynnfix_%j.out
set +e
# Acceptance for the two-sided D25 floor. Three things have to hold before the
# eight-case campaign is relaunched with mynn25 in it:
#
#   1. cbl RUNS. It is the case the pole killed at step 10.
#   2. The other SEVEN cases are UNCHANGED -- byte-identical dumped columns
#      against the pre-fix pin, which is the whole claim of a two-sided
#      sign-preserving floor. This is a diff of arrays, not of summaries.
#   3. The unit gate is green, including the NN09 oracle-parity tests
#      (GABLS1 / Wangara / Ekman) that pin the closure's faithfulness.
#
# BOTH pins are run here, in one job, so the comparison cannot drift on
# hardware or environment.
: "${OLD_WT:?set OLD_WT to the PRE-fix worktree}"
: "${NEW_WT:?set NEW_WT to the POST-fix worktree}"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
P=scripts/validate/diag_scheme_blowup.py
DUMPS=/burg-archive/glab/users/pg2328/legoESM/logs/mynnfix_dumps_$SLURM_JOB_ID
mkdir -p "$DUMPS"
echo "host: $(hostname)"
echo "OLD pin: $(git -C "$OLD_WT" rev-parse HEAD)"
echo "NEW pin: $(git -C "$NEW_WT" rev-parse HEAD)"
CASES="ekman gabls1 cbl wangara bomex rico astex dycoms"

run_pin () {                 # $1 = worktree, $2 = tag
  local WT=$1 tag=$2
  cd "$WT" || exit 1
  export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')
  export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
  for c in $CASES; do
    # 120 steps is well past the step-10 death and long enough for the seven
    # healthy cases to accumulate a difference if the floor ever fires.
    $PY $P --case "$c" --scheme mynn25 --nlev 48 --steps 120 \
        --print-every 60 --dump-profiles > "$DUMPS/${tag}_$c.txt" 2>&1
    echo "  $tag $c rc=$?  $(grep -c NON-FINITE "$DUMPS/${tag}_$c.txt") nonfinite-lines"
  done
}

echo "=============== 1. run both pins ==============="
run_pin "$OLD_WT" old
run_pin "$NEW_WT" new

echo "=============== 2. cbl must now survive ==============="
for tag in old new; do
  echo "--- $tag / cbl (tail) ---"
  grep -E "^ +[0-9]+ +[0-9.]+|NON-FINITE|finite through" "$DUMPS/${tag}_cbl.txt" | tail -8
done

echo "=============== 3. the other seven must be UNCHANGED ==============="
# Strip the jax CUDA-absent banner (it is host state, not a result) and diff
# the rest, which includes every dumped column value.
for c in $CASES; do
  for tag in old new; do
    grep -vE "Traceback|^  File|^    |RuntimeError|cuInit|plugin|\^\^\^|^Jax plugin" \
        "$DUMPS/${tag}_$c.txt" > "$DUMPS/${tag}_$c.clean"
  done
  if diff -q "$DUMPS/old_$c.clean" "$DUMPS/new_$c.clean" >/dev/null; then
    echo "  $c IDENTICAL"
  else
    echo "  $c DIFFERS:"
    diff "$DUMPS/old_$c.clean" "$DUMPS/new_$c.clean" | head -14
  fi
done

echo "=============== 4. unit gate at the NEW pin ==============="
cd "$NEW_WT" || exit 1
export PYTHONPATH=$NEW_WT/src:$(ls -d "$NEW_WT"/packages/*/ | tr '\n' ':')
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
$PY -m pytest -q \
    tests/atmosphere/hydrostatic/unit/test_mynn25_d25_pole.py \
    tests/atmosphere/hydrostatic/unit/test_mynn25_faithful.py \
    tests/atmosphere/hydrostatic/unit/test_mynn25.py \
    tests/test_param_specs.py tests/test_physics_contracts.py \
    tests/test_no_inline_physics_coeffs.py \
    tests/unit/test_diag_scheme_blowup.py \
    tests/unit/test_plot_scm_les_turbulence_case_select.py \
    tests/atmosphere/hydrostatic/test_run_scm_les_turbulence_tuning.py \
    2>&1 | tail -25
echo "unit rc=${PIPESTATUS[0]}"

echo "=============== 5. NON-VACUITY: revert the floor, cbl must die ==============="
# The acceptance above is only worth something if the fix is what produced it.
$PY - <<'PYEOF' 2>&1 | tail -6
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
from legoesm.atmosphere.physics.turbulence.config import MYNN25Config
from legoesm.atmosphere.physics.turbulence.mynn25 import _compute_SM_SH
import jax.numpy as jnp
for floor, label in ((1e-30, "pre-fix"), (1e-2, "post-fix")):
    cfg = MYNN25Config(d25_floor=floor)
    slope = 3*cfg.A2*cfg.B2*(1-cfg.C3) + 12*cfg.A1*cfg.A2*(1-cfg.C2)
    root = 1.0/slope
    worst = 0.0
    for gh in np.linspace(0.5*root, 1.5*root, 2001):
        sm, sh = _compute_SM_SH(jnp.zeros((1,1)), jnp.full((1,1), gh),
                                jnp.full((1,1), 4.0), jnp.full((1,1), 1e-5), cfg)
        worst = max(worst, abs(float(sm.ravel()[0])), abs(float(sh.ravel()[0])))
    print(f"{label}: phi_4 root G_H={root:.6g}  max|SM25|,|SH25| near it = {worst:.6g}")
PYEOF
echo "DONE $(date +%T)"
