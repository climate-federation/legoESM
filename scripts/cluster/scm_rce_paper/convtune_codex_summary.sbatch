#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=convtune_cdxs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_cdxs_%j.out
set +e
# ---------------------------------------------------------------------------
# Adversarial review of the two instruments added for the anchor-off arm's
# results: the tuning-summary plotter and the ZM cost bench.  Both are
# INSTRUMENTS, which is exactly the class this repo has been burned by --
# a bug here manufactures a confident wrong conclusion rather than an error.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
# CODEX_HOME must be node-local AND must still exist when codex reads it: on
# g059 (job 9361561) the /tmp dir vanished between `cp` and codex start.  The
# location is therefore chosen by TESTING it, and codex's exit status is
# checked rather than assumed (a bad CODEX_HOME has historically exited 0
# having reviewed nothing).
for _CH in "${SLURM_TMPDIR:-}" "/tmp" "/burg-archive/glab/users/pg2328/_codexhome"; do
  [ -n "$_CH" ] || continue
  CANDIDATE="$_CH/codex_home_${SLURM_JOB_ID:-$$}"
  mkdir -p "$CANDIDATE" 2>/dev/null || continue
  cp /burg-archive/home/pg2328/.codex/auth.json "$CANDIDATE/" 2>/dev/null || continue
  [ -f "$CANDIDATE/auth.json" ] || continue
  export CODEX_HOME="$CANDIDATE"
  break
done
[ -n "${CODEX_HOME:-}" ] || { echo "FATAL: no usable CODEX_HOME"; exit 90; }
echo "CODEX_HOME=$CODEX_HOME"
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version || exit 91

cd $WT || exit 92
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 93; }

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of two new INSTRUMENTS. Assume every number they produce
will be published; your job is to find the ways they produce a WRONG number
that looks right, not to praise them.

FILES:
  scripts/plot/plot_scm_rce_convection_tuning_summary.py
  tests/unit/test_plot_scm_rce_convection_tuning_summary.py
  scripts/bench/bench_scm_convection_cost.py
  tests/unit/test_bench_scm_convection_cost.py

CONTEXT (measured, do not re-derive):
  * 10 convection schemes were tuned in a single-column RCEMIP1 SCM against a
    SAM CRM reference, 100 d, dt 600 s, last 5 d averaged. Anchor-off arm,
    results in results/scm_rce_convtune/arm_implicit_flux_morrison_noanchor.
  * The combined score is
      sqrt((T_rmse^2 + qv_rmse^2 + cloud_rmse^2 + w*precip_rmse^2)/(3+w)), w=1
    (packages/ml/legoesm/training/scm_rce_metrics.py), each component
    normalized by the reference profile's own mass-weighted std.
  * `kuo` is INACTIVE in a single column (convection/integration.py needs a
    large-scale moisture-convergence operator) and scores 5.736693549527354,
    bit-identical to convection=none. `dca` exposes 0 tunable parameters.
  * zhang_mcfarlane cost 1114 s/eval vs 220-312 s for the other nine.

REVIEW THESE SPECIFIC RISKS, and say file:line for each finding:

A. PLOTTER. (1) Can `_structural_note` label a scheme that genuinely failed
   its search as "flat by construction", or miss one that IS structural?
   Its rule is `prior score == tuned score` plus a name table -- what breaks
   it? (2) `_ff` maps None to NaN; trace EVERY consumer and say whether a NaN
   can reach a median, an axis limit, or `np.nanargmin` in a way that throws
   or silently mis-ranks. (3) The jitter is drawn with size = before.size and
   indexed by scheme index -- is it consistent between panels and between the
   before/after points of the SAME scheme? (4) `figure_profiles` reads
   `precip_ref_mm_day` from `recs[0]` only -- is that safe if schemes carry
   different reference values? (5) Is anything about the figure capable of
   showing "improved" when the scheme degraded, or vice versa?

B. BENCH. (1) The vector-width control claims to vary ONLY ncol. Read
   `build_state` and say whether ncol changes anything else that could alter
   per-column cost (broadcasting, memory layout, a scheme that reduces across
   columns). (2) `time_scheme` compiles inside the timed function and then
   excludes the first call -- is one warmup call enough on CPU, and can XLA
   cache across the ON/OFF and _NEWTON_ITERS arms in a way that makes an arm
   secretly reuse another's executable? Note the code calls
   `jax.clear_caches()` between scaling arms but NOT between ON/OFF arms --
   is that a defect? (3) Is `statistics.median` over `repeats` calls the right
   reduction, given the campaign quantity of interest is total wall clock?
   (4) The `_NEWTON_ITERS` monkeypatch: is patching the module attribute
   sufficient, or could a `from ... import _NEWTON_ITERS` elsewhere, a
   closure, or a default argument capture the old value? Grep for it.
   (5) Does the probe's synthetic column actually TRIGGER convection in every
   scheme? If a scheme's trigger never fires, its timing is the cost of the
   inactive branch and the "pack" it is compared against is not a fair
   baseline. This is the single most likely way this bench lies -- say
   whether it does, and how to check.

C. TESTS. For each test file: name any test that CANNOT FAIL, and any claim
   in a docstring that the test does not actually establish.

Answer with a numbered list of concrete defects, each with file:line and the
one-line fix. If a risk above is NOT a defect, say so explicitly and why --
do not pad.
PROMPT
)" 2>&1
CODEX_RC=$?
echo "CODEX_RC=$CODEX_RC"
[ $CODEX_RC -eq 0 ] || echo "REVIEW DID NOT RUN — do not treat this job as a review"
echo ALLDONE
