#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_fp32r
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_fp32r_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID; mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"
codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of the LAST COMMIT (git show HEAD; git log --oneline -3):
"fv3-duo: fp32 runtime (increment 2)". It makes the FV3 duo dycore run
coarse single-precision (whole step in one storage dtype) while keeping the
certified fp64 path BYTE-IDENTICAL. It builds on increment 1 (dtype-
uniformity gates + FV3DuoConfig.storage_dtype + a step() boundary guard).

Validation already run (for context, verify the CLAIMS not the numbers):
fp32 hydro+NH smoke finite/physical; DCMIP16 fp32-vs-fp64 divergence
~1e-5 rel over 10 steps; fp64 duo suite 57 passed BEFORE the 0-dim gate
change.

ATTACK, cite file:line, BLOCKER/MAJOR/MINOR or CLEAN:
1. fp64 BYTE-IDENTICAL is the core claim. For an all-fp64 run, is EVERY
   change a no-op? Check: (a) workspace allocs `dtype=jnp.float64` ->
   `dtype=<field>.dtype` (field is f64 -> same); (b) the _new/_nan
   `functools.partial(..., dtype=<field>.dtype)` per-function partials --
   any that capture a variable `del`'d later (the commit fixed one_grad_p's
   `del delp` -> pk.dtype; are there OTHERS across fv3_duo_sw_core /
   fv3_tp_core where a partial or a `<field>.dtype` sits after a `del` or
   references a field not in scope?); (c) metric casts guarded by
   `dtype is not None` (None=f64 uncast); (d) `dt = jnp.asarray(dt,
   dtype=state["delp"].dtype)` at the acoustic substep -- for fp64 the
   carry is f64 so dt stays f64, same as before?
2. THE 0-DIM SCALAR GATE SKIP (require_uniform_float_jax + 7 private copies
   now `if _arr.ndim == 0: continue`). Does this dangerously weaken the
   anti-mixed-precision guard? Argument in the diff: a scalar is weak-
   promoting and a real strong-f64 0-dim leak resurfaces as an f64 FIELD at
   the next phase. Is that argument SOUND, or is there a case where a 0-dim
   f64 strong scalar silently promotes an f32 field and is NEVER caught
   downstream (e.g. the last phase, or a reduction to a scalar output)?
3. Did any workspace alloc take a METRIC (area/cos/rd*) as its `.dtype`
   source instead of a field? Under coarse-uniform both are storage dtype
   so it runs, but it is fragile -- flag any.
4. The metric casts (gs6/hs6/area/amat/vlon4/es4/...) for fp32 downcast the
   pressure-column/geopotential metrics too -- the GLM/codex feasibility
   study said the pressure column + energy fixer should stay fp64 even in
   fp32 for accuracy. The commit does COARSE all-fp32 (not per-op mixed)
   and documents mixed as increment 3. Is coarse all-fp32 with the pressure
   column in fp32 a CORRECTNESS problem, or an acceptable accuracy tradeoff
   given the ~1e-5 DCMIP divergence and the deferred mixed mode? (The duo
   lane runs consv_te=0, so the energy fixer is not in the path.)
5. The scatter FutureWarning (an f64 value .set into an f32 base) is
   acknowledged as a residual. Can you LOCATE it (grep `.at[` `.set(` in
   the fp32 step path where the value expression is f64 while the base is
   f32)? Is it benign (safe downcast) or a latent bug?
6. Any comment/docstring now false; any NEW bug the diff introduces.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_FP32R_DONE
