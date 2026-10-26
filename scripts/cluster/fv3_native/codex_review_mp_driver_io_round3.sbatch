#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_mpio_r3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_mpio_r3_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha (base): $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | wc -l)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW, ROUND 3, of the UNCOMMITTED multi-process fv3_duo
checkpoint restart path. Run `git diff -- packages/coupler/legoesm/driver/model_driver.py`
and focus ONLY on `_load_fv3_duo_checkpoint`, `_broadcast_fv3_duo_bundle`,
`_validate_fv3_duo_broadcast_source`, and `_fv3_duo_broadcast_leaf_spec`
(grep those names).

Round 2 found 4 issues in these functions; this is the fix for each. Verify
each is ACTUALLY closed:

1. (was: unguarded per-rank work ahead of the barrier could deadlock).
   `template = self._fv3_duo_flatten_bundle(self.model.dcmip16_initial_state(...))`
   is now INSIDE the SAME try block that also does root's file-check +
   decode + validate + header-fill, and EVERY rank (not just root) enters
   that try unconditionally and calls `self._spmd_barrier_on_root_error(err)`
   right after, whether or not it raised. Verify: is there ANY statement
   between the start of `_load_fv3_duo_checkpoint`'s multi-process branch
   and the `_spmd_barrier_on_root_error(err)` call that executes OUTSIDE
   this try/except, on any rank?

2. (was: broadcast order could silently diverge across ranks if dict
   iteration order were ever non-deterministic). `names` in
   `_broadcast_fv3_duo_bundle` is now built via
   `sorted(nm for nm in template if not nm.startswith("q_"))` plus
   `[f"q_{i}" for i in range(nq)]`. Verify this fully removes any
   dependence on dict iteration order for the broadcast sequence.

3. (was: checkpoint nq never cross-checked against this deck's actual
   tracer count, so nq=0 or a foreign count could silently pass).
   `_validate_fv3_duo_broadcast_source` now computes
   `template_nq = sum(1 for nm in template if nm.startswith("q_"))` and
   raises if `nq != template_nq`. Verify this actually runs before the
   `expect`/`got` set comparison and closes the nq=0 hole.

4. (was: the shape/dtype ternary for a leaf was duplicated between the
   validator and the broadcaster, risking drift). Both now call a shared
   `_fv3_duo_broadcast_leaf_spec(nm, template, q_shape, q_dtype)`. Verify
   both call sites actually use it (not a stray leftover inline ternary
   anywhere).

ALSO: `self.model.dcmip16_initial_state(do_pert=True)` -- confirm (grep
`packages/core/legoesm/core/fv3_native_dcmip16_ic.py::dcmip16_bc_six_face_state`,
the function it calls) that `do_pert` is a DETERMINISTIC analytic
perturbation (no `np.random`/RNG anywhere in that file), so calling this on
every rank for template shapes is safe and rank-invariant. State what you
find, don't assume.

Output BLOCKER/MAJOR/MINOR with file:line and the concrete failure
scenario, or say explicitly CLEAN if all 4 are closed and the do_pert
claim holds.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_MPIO_R3_DONE
