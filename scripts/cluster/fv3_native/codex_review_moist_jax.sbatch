#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_moistjax
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_moistjax_%j.log
# Adversarial review of the moist (zvir) coupling ported into the JAX lane.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"
echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of ONE commit on the FV3 duo-grid JAX port: the HEAD
commit "feat(fv3): moist zvir coupling in the JAX lane". Read it with
`git show HEAD`. Two files change:
packages/core/legoesm/core/fv3_dynamics.py (the JAX PORT) and
tests/grids/test_fv3_dynamics.py (its gate).

THE AUTHORITIES, in order:
1. packages/core/legoesm/core/fv3_native_dynamics.py is the NumPy
   SPECIFICATION. Any behavioural difference from it is a port bug by
   definition. Its own moist arm landed one commit earlier; diff the two
   lanes' moist paths against each other line by line.
2. The pinned Fortran oracle at
   /burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/model/
   -- fv_dynamics.F90:281-294 (dp1 formation), :396-408 (the pt ->
   theta_v conversion), :307-309 (the NH moist pkz that stays refused),
   and fv_mapz.F90:975 (the closing pt/(1+r_vir*q)).

ATTACK IN THIS ORDER. Cite file:line for every claim.

1. THE dp1 SLICE. The port forms
   `dp1_theta = zvir * q[sphum_index][:, ia:ia+n, ia:ia+n, :]` on a
   TRACER-MAJOR list of FACE-STACKED arrays, where the spec loops faces
   and slices `q[t][sphum_index][ng:ng+n, ng:ng+n, :]`. Prove the two
   select the same numbers: check the axis order of the port's q
   contract (read fv_dynamics_step's docstring AND how q is consumed at
   the remap call), and check that pkz's shape really is (6,n,n,km) so
   the broadcast in pt_to_theta_v is element-wise and not silently
   broadcasting a wrong axis. A transposed-but-square fixture (n=n)
   would hide an axis swap -- say so if the gate cannot distinguish it.
2. WHEN dp1 IS FORMED. The oracle forms dp1 from the STEP-INITIAL q,
   before the k_split loop (:281-294 precedes :451). Confirm the port's
   placement relative to the `_n_map` loop and to any q mutation, and
   confirm that at k_split=2 the port does NOT re-form or stale-read it
   differently than the spec.
3. THE ASSOCIATION. The commit changes `win * ((1.0 + dp1) / pkz)` to
   `win * (1.0 + dp1) / pkz` claiming the Fortran associates left to
   right. Verify that claim against fv_dynamics.F90:402 as WRITTEN, and
   verify JAX/Python evaluate the new expression in that order (it is
   not obvious that a jnp expression preserves it under jit -- if XLA
   may reassociate or contract these into an FMA, say so, and say what
   that does to the bitwise claim the test makes).
4. THE REMAP HOP. The commit asserts the closing pt/(1+r_vir*q) needed
   no change because the call already passes r_vir=zvir. Verify in
   packages/core/legoesm/core/fv3_mapz.py that r_vir != 0 is actually
   honoured, that its sphum_index guard matches the spec's, and that
   `adiabatic=True` being hard-coded at that call site does not disable
   the moist path. This is the hop most likely to be silently dead.
5. THE REFUSALS AND VALIDATION. zvir with hydrostatic=False must raise;
   consv_te must raise; sphum_index must be validated. Check the guard
   ORDER against the shape checks -- a guard that raises only after a
   shape error is not the guard the test thinks it is. Check the bool
   case (True is an int in Python) and the negative-wrap case.
6. CAN THE NEW TESTS FAIL? For EACH new test in
   tests/grids/test_fv3_dynamics.py, name the code edit that makes it go
   red. Specifically:
   - test_moist_hydrostatic_parity_against_the_spec: would it still pass
     if the port dropped dp1 entirely? (both lanes are separately
     invoked, so answer from the code, not from intent)
   - test_the_moist_arm_actually_changed_the_answer: its second half
     compares two max-abs SCALARS across lanes. Is that a real
     cross-lane check or can two different fields give the same max?
   - test_dry_branch_is_not_a_multiply_by_one: does the anti-vacuity
     assert actually hold for the chosen dp1, and does the bitwise
     assert survive x64/jit differences?
   - the refusal parametrize: does each case reach the intended guard,
     or could an earlier error raise the same exception type first?
7. THE TRACER `scale` KNOB. Tests now run the moist arm at q ~ 0.01
   while the dry arm stays at O(1). Does that change what the dry gates
   measure? Is any existing tolerance in that file now calibrated
   against different data than when it was measured?
8. ANY COMMENT OR DOCSTRING ADDED HERE THAT IS NOT TRUE OF THE CODE.
   This repo treats those as testable claims.

Output BLOCKER / MAJOR / MINOR with file:line, the concrete failure
scenario, and the smallest fix. Say explicitly where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"
echo CODEX_MOISTJAX_REVIEW_DONE
