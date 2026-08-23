#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_tol
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_tol_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of the tolerance-closure campaign on the FV3 duo-grid
JAX port's gate files. Scope: the two most recent commits touching
tests/grids/ (git log --oneline -5; diff them). ~292 provisional
TOL-PENDING bounds across twelve gate files were replaced by
bound = measured x 10, where "measured" comes from an env-gated
measurement mode added to the comparison helpers.

THE DOCTRINE UNDER REVIEW, not just the diff: a tolerance derived from
one measurement run on one host certifies nothing about the mechanism --
it certifies "no bigger than this that day". The campaign's defence is
(a) structural asserts stay live in measure mode, (b) exact-0.0
measurements get an eps guard rather than 10x0, (c) every >1e-9
measurement is NAMED as a potential defect at its site rather than
silently bounded. Attack whether those defences actually hold in the
code.

ATTACK IN THIS ORDER, file:line each:

1. THE INSTRUMENTATION (tests/grids/fv3_gate_helpers.py). gate_scalar,
   gated_check_grads, and the measure-mode branches added to
   check_adjoint and the per-file _cmp helpers: can measure mode
   suppress an assert it should not (a structural check accidentally
   downstream of the early return)? Is gated_check_grads' "smallest
   passing power-of-ten" search sound -- does it call check_grads with
   both atol and rtol, and can the search mask a REAL gradient defect
   by walking up to a huge tolerance and reporting it as "measured"?
   Is anything nondeterministic between the measurement run and the
   verification run (RNG, host libm, thread count)?
2. THE BOUNDS THEMSELVES. Spot-check ten sites across different files
   against the harvest (/burg-archive/glab/users/pg2328/fv3_duo_gaps/
   tolmeasure_9425294/harvest.txt): does the written bound equal the
   printed measurement x 10 (one-digit round-up), does the cited value
   match the harvest line, and is the WORST parametrisation used where
   one bound covers a loop? Name any site where the comment's number
   and the harvest disagree.
3. THE EPS GUARDS. Every exactly-0.0 site now asserts <= 1e-15. For
   comparisons whose operands are O(1e5), a 1e-15 RELATIVE guard is
   ~1e-10 absolute -- is each guard actually relative, and is 1e-15 the
   right scale for a bitwise-expected comparison (vs array_equal)? Any
   site where the 0.0 was VACUOUS (comparing empty/masked-out arrays)
   and the guard certifies nothing?
4. THE NAMED DEFECTS. The full-step jit-vs-eager wind gap (7.5e-07,
   now on u AND v, localised to acoustic_step_sixface), cgrid jit
   nh.ws3 1.6e-09, nh_core one-sided FD 4.6e-06: is each recorded at
   its site as UNEXPLAINED/PLAUSIBLE with enough evidence pointers that
   a future reader can pick it up, or did any get a bound that reads
   as if it were understood? Is 7.5e-07 on winds inside a bound that a
   REGRESSION to 5e-06 would still pass? State what the bound would and
   would not catch.
5. THE TWO RED-TEST FIXES in the follow-up commit: the spec _out_like
   pkc entry (is the mapping true of the spec -- check :253's
   allocation) and the nh_core check_grads at atol=rtol=2.2e-4 (is
   bounding an order-1 FD at 22x its default defensible given the
   quoted adjoint-zero + eps^2-ladder evidence, or is it hiding a real
   fwd/rev defect the adjoint cannot see? Be specific about what the
   adjoint identity CAN and CANNOT certify for this operator).
6. ANY COMMENT ADDED that is not true of the code.

Output BLOCKER / MAJOR / MINOR with the concrete failure each enables.
Say explicitly where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"
echo CODEX_TOL_REVIEW_DONE
