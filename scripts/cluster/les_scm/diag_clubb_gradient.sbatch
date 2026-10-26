#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=clubb_grad
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/clubb_grad_%j.out
set +e
# WHICH clubb parameter carries the 1e15 gradient?
#
# The 8-case campaign measured, at step 1 and the DEFAULT parameters:
#   clubb        dL = -9.95e14, best rejected 1.004651 at scale 0.001
#   clubb_lite   dL = -1.31e12
#   smagorinsky  dL = -1.82e17
# A loss of order 1 with a gradient of 1e15 is not a physical sensitivity. It
# is a near-singular denominator or a clip boundary, and it is why no scale in
# the line-search ladder finds a reducing step: the linear model is valid over
# a neighbourhood far smaller than the smallest step tried.
#
# PRE-REGISTERED. The driver now prints the per-parameter |grad| spectrum.
#   CONFIRMS "one broken term": the spectrum spans many orders of magnitude
#     and a small number of parameters carry the 1e14-1e15 entries, naming the
#     closure term to inspect.
#   REFUTES it: the spectrum is FLAT at ~1e15, i.e. the whole rollout is
#     ill-conditioned rather than one denominator, which is a different (and
#     worse) problem and must be reported as such.
# Also prints the preflight loss: under default_relative it must be EXACTLY 1
# at the defaults, and the campaign measured 1.00033 for clubb and 1.00052 for
# mynn25 -- a lossy physical->unconstrained->physical round trip that is a
# second, independent defect.
#
# Two cases only (cheap, and enough for a gradient at the default point):
# bomex is where clubb scores BEST (0.075) and wangara where it scores worst.
: "${E_WT:?set E_WT}"
cd "$E_WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)  host: $(hostname)"
export PYTHONPATH=$E_WT/src:$(ls -d "$E_WT"/packages/*/ | tr '\n' ':')
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
for CASES in "bomex" "wangara"; do
  echo "===== $CASES ====="
  $PY scripts/run/run_scm_les_turbulence_tuning.py \
      --cases "$CASES:$R/les_ref/$CASES" \
      --outdir "$R/scm_les_turbulence/_clubb_grad_$CASES" \
      --nlev 48 --steps 2 --schemes clubb,clubb_lite \
      --joint-aggregation default_relative --surface-flux-to-closure \
      2>&1 | grep -Ev "^E[0-9]{4}|slow_operation|XLA has|If you|^$"
done
echo "DONE $(date +%T)"
