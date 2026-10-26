#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=grad_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/grad_sweep_%j.out
set +e
# CHAOS or ONE ILL-CONDITIONED DENOMINATOR? The length sweep separates them.
#
# Both hypotheses fit every number measured so far: |grad| ~ 1e15 against a
# loss of order 1, no line-search scale reducing the loss, and an objective
# that reads 1.00033 at the DEFAULT parameters where it must read exactly 1.
#
# The chaos story says identical coefficients arriving by two code paths
# (static Python floats vs traced arrays) diverge through rounding alone and
# amplify as exp(lambda*T). Its weak point, which an independent review found:
# SMAGORINSKY carries the LARGEST gradient (-1.82e17) and yet returns EXACTLY
# 1.0. Under chaos its discrepancy should be the biggest, not zero.
#
# The alternative says a single parameter-dependent quantity passes near zero
# (catastrophic cancellation into a denominator). That needs no time
# amplification and predicts smagorinsky's exact 1.0 naturally.
#
# PRE-REGISTERED on |grad|max, printed per scheme by the driver:
#   CHAOS          -> |grad|max grows ~log-linearly with rollout length. At
#                     0.5 h (30 steps at dt=60) it must be MANY orders below
#                     its 6 h value; halving T should divide it by ~exp(lambda*T/2).
#   ILL-CONDITIONED -> |grad|max is roughly FLAT in length and already ~1e15 at
#                     0.5 h. Arithmetically, 30 steps cannot amplify 1e-16 to
#                     1e15, so a large gradient at 0.5 h REFUTES chaos outright.
#   Secondary tell: chaos spreads the magnitude across many parameters and
#   rotates the direction with T; a bad denominator keeps the SAME one or two
#   parameters dominant at every length.
#
# bomex on purpose: it is the case where clubb is the BEST of all nine closures
# (0.075), so whatever is wrong is not masked by a bad arm. --skip-tuning is
# NOT used: the preflight gradient print happens on the tuning path.
: "${E_WT:?set E_WT}"
cd "$E_WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)  host: $(hostname)"
export PYTHONPATH=$E_WT/src:$(ls -d "$E_WT"/packages/*/ | tr '\n' ':')
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
for H in 0.5 1 2 6; do
  echo "===== rollout ${H} h ====="
  $PY scripts/run/run_scm_les_turbulence_tuning.py \
      --cases "bomex:$R/les_ref/bomex" \
      --outdir "$R/scm_les_turbulence/_gradsweep_${H}h" \
      --nlev 48 --steps 1 --schemes clubb,smagorinsky \
      --hours "$H" --analysis-hours 0.25 \
      --joint-aggregation default_relative --surface-flux-to-closure \
      2>&1 | grep -E "preflight loss|largest:|no reducing|accepted|joint |NOTE|refus|Error"
done
echo "DONE $(date +%T)"
