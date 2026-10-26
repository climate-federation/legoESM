#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=wangara_bc
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/wangara_bc_%j.out
set +e
# Does the per-step surface-flux boundary condition actually repair Wangara?
#
# PRE-REGISTERED, before the job ran. The number is the per-case normalized
# profile score for Wangara, evaluated with --skip-tuning (defaults only) so
# nothing can be absorbed into a fitted parameter.
#
#   The two BROKEN arms measured, on every closure:
#     8.6 - 9.9   the diurnal cycle frozen at its 09:00 instant
#     13.0 - 15.5 the cycle driven correctly but the closure handed ZERO
#   Every OTHER case scores 0.06 - 3.4 at defaults.
#
#   CONFIRMS the fix: Wangara lands in that same 0.06 - 3.4 band, i.e. it
#     becomes an ordinary case rather than an outlier, and the spread ACROSS
#     closures is comparable to the other dry cases.
#   REFUTES it: still >= 8 on every closure -- a scheme-independent score is
#     the arm, not the closures, so the boundary condition is still wrong and
#     Wangara must stay out of the campaign.
#   AMBIGUOUS (report as such, do not spin): 3.4 - 8. Better, not fixed.
#
# The second arm is the double-count control. With --no-surface-flux-to-closure
# Wangara falls back to the column-tendency route, which is the OLD 13.0-15.5
# behaviour; if the new route were ALSO leaving the column injection on, the
# two arms would not differ in the direction expected (BC-only should warm the
# surface layer less than BC + column injection).
: "${E_WT:?set E_WT to the pinned worktree}"
WT=$E_WT
cd "$WT" || exit 1
PIN=$(git rev-parse HEAD)
echo "PINNED sha: $PIN"; echo "host: $(hostname)"; nvidia-smi -L
if [ -n "$EXPECT_SHA" ] && [ "$PIN" != "$EXPECT_SHA" ]; then
  echo "worktree is at $PIN, expected $EXPECT_SHA -- refusing"; exit 4
fi
export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
OUT=${OUT:-$R/scm_les_turbulence/_wangara_bc}

echo "=== unit: the new boundary-condition path ==="
$PY -m pytest -q \
    tests/atmosphere/test_prescribed_surface_flux_to_closure.py \
    tests/atmosphere/hydrostatic/unit/test_scm_forcing.py \
    tests/atmosphere/hydrostatic/test_run_scm_les_turbulence_tuning.py \
    2>&1 | tail -25
echo "  pytest rc=$?"

echo "=== unit: nothing else moved (PhysicsState is cross-cutting) ==="
# Two new PhysicsState fields is a change to every consumer of the carry, so
# run the carry/migration gates and the whole hydrostatic unit tier -- the
# new tests alone would not catch a broken restart or a stale-carry gate.
$PY -m pytest -q \
    tests/unit/test_physics_state_carry.py \
    tests/unit/test_physics_state_migration.py \
    tests/atmosphere/hydrostatic/unit \
    2>&1 | tail -14
echo "  pytest rc=$?"

[ -d "$R/les_ref/wangara" ] || { echo "MISSING wangara reference"; exit 2; }
rm -rf "$OUT"
for MODE in on off; do
  if [ "$MODE" = on ]; then FLAG="--surface-flux-to-closure"; else FLAG=""; fi
  echo "=== wangara, defaults only, flux-to-closure=$MODE ==="
  $PY scripts/run/run_scm_les_turbulence_tuning.py \
      --cases "wangara:$R/les_ref/wangara" --outdir "$OUT/$MODE" \
      --nlev 48 --skip-tuning $FLAG 2>&1 | grep -Ev "^E[0-9]{4}|slow_operation|^$"
  echo "  rc=$?"
  cat "$OUT/$MODE/ranking.csv"
done
echo "DONE $(date +%T)"
