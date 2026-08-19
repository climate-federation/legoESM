#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_gates
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_gate_triage_%j.log
# Adversarial review of the INSTRUMENTS, not the model: the shared 3-D phase
# gates, the shared test helpers, the triage probe, and the five gate
# recalibrations it justified.  The repo's rule is explicit that the
# diagnostic tooling gets reviewed too -- a bug there manufactures a
# confident wrong conclusion.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; echo "CODEX_HOME=$CODEX_HOME"; codex --version
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"
echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW OF DIAGNOSTIC INSTRUMENTS, not of model physics.

CONTEXT. This branch ports a loop-faithful NumPy fp64 FV3 duo-grid dycore
into JAX. The contract is docs/atmosphere/fv3_duo_jax_lane_strategy.md and
the live state is docs/atmosphere/fv3_duo_jax_lane_STATE.md -- read both.
The commits under review do NOT change model numerics. They change the
things that decide what we BELIEVE about the model, which in this repo has
historically been where the expensive mistakes live.

REVIEW THESE COMMITS (git log --oneline main...HEAD; git show <sha>):
  a111bf998  promote 7 duplicated private entry gates/assembly helpers into
             packages/core/legoesm/core/fv3_phase3d_common.py, taking the
             SUPERSET where two copies had diverged
  dbbc46628  two gates that asserted false premises about JAX itself
  13b60c41e  sys.path fix for the triage probe
  a7f5a5a1d  tests/grids/fv3_gate_helpers.py -- one home for the comparison
             and gradient helpers, plus its own gates
  7bc183a9b  five gradient/fixture gate recalibrations, each keyed to a
             number printed by scripts/validate/fv3_gradient_gate_triage.py
             (job 9417326)

ATTACK IN THIS ORDER, cite file:line, and be concrete.

1. THE PROMOTION MUST BE BEHAVIOUR-PRESERVING. fv3_phase3d_common.py claims
   each function is the UNION of two previously divergent private copies,
   with the superset taken. Diff it against what a111bf998 deleted from
   fv3_cgrid_phase_3d.py and fv3_dsw_phase_3d.py. Find any case where the
   promoted version accepts something a caller's old copy rejected, or
   rejects something it accepted -- in particular:
   - stack_levels' want2d=None mode versus the old want2d-required copy:
     is any call site now checking LESS than before?
   - validate_stacked now maps names through CSW_OUT_LIKE. Is that mapping
     really the identity for every name the C-grid module passes, or does
     some name now resolve to a DIFFERENT field_shape than it used to?
   - require_nord gained a `name` parameter; find any call site left at the
     old arity or passing the wrong name.
   - CSW_OUT_LIKE moved and is re-exported. Is the re-export the SAME object,
     and does any test still pin it against the NumPy spec's private copy?

2. THE FIVE RECALIBRATIONS ARE CLAIMS ABOUT INSTRUMENTS. For each, decide
   whether the new gate can still FAIL on a real defect, or whether it was
   widened until it cannot:
   - one_grad_p affine gate now bounds the SECOND DIFFERENCE as a multiple of
     u_mach*|f|_inf instead of bounding the ratio. Could a genuine quadratic
     term hide under that bound? What size of curvature would it take, and is
     that size physically reachable in one_grad_p?
   - nh_p_grad's FD ladder moved to eps=(1.6e-3, 8e-4, 4e-4). Is the step now
     large enough that the eps^2 model itself breaks (higher-order terms), so
     the 4.0 ratio is arriving for the wrong reason?
   - update_dz_d and the mapz iv=-2 BC gate replaced check_grads(order=2)
     with a tolerance-free adjoint identity. The identity tests jvp against
     vjp of the SAME jacobian -- name what class of defect it therefore
     CANNOT see, and say whether the order=1 check_grads left beside it
     covers that class.
   - the smt5 gate now bisects on a step in d flux/dt instead of a boolean.
     Is "piecewise linear in t at iord=5" actually true of the code path, or
     only of the arm the fixture happens to take? What happens if the scan
     interval contains TWO switches?

3. THE TRIAGE PROBE ITSELF (scripts/validate/fv3_gradient_gate_triage.py).
   Its numbers are now cited in five test docstrings and a commit message.
   - Does each block measure the quantity its header claims?
   - P3 sweeps the probe SCALE and concludes affine-plus-cancellation from a
     1/s trend. Is there any non-affine f that would also produce 1/s?
   - P2 reports an EXACTLY zero adjoint residual. Is that credible, or a
     sign the two sides share a computation (e.g. the same lowered program)
     and the identity is trivially satisfied?
   - P5 concluded the old boolean detector was tracking a zero crossing.
     Check that conclusion against the printed margins.
   - Are the pre-registered CONFIRMS/REFUTES lines actually discriminating,
     or could both be true at once?

4. THE SHARED TEST HELPERS (tests/grids/fv3_gate_helpers.py). cmp_fields is
   the comparison every future gate will use. Find a case where it PASSES on
   two fields that differ materially, or FAILS on two that do not:
   - the median scale floor `|b| + median|b|`;
   - fill classification by exact value;
   - the n_over threshold at a hard-coded 1e-13;
   - NaN mask equality when one lane has MORE NaN than the other.
   Then check its own test file actually pins those properties.

5. ANY CLAIM IN A COMMENT OR COMMIT MESSAGE THAT IS NOT TRUE OF THE CODE.
   Quote it and say what the code does instead. Especially the CORRECTION
   and RETRACTION blocks -- three of them describe measurements; verify the
   described mechanism matches the code that was changed.

Output: findings ranked BLOCKER / MAJOR / MINOR, each with file:line, the
concrete failure it enables, and the smallest fix. If a section has nothing,
say so explicitly rather than inventing a finding.
PROMPT
)"
echo "CODEX RC=$?"
echo CODEX_GATE_REVIEW_DONE
