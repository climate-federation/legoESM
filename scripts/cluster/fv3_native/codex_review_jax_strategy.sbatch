#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3jax_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/fv3jax_codex_strategy_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
# Codex needs a NODE-LOCAL CODEX_HOME (shared-FS home has failed before).
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; echo "CODEX_HOME=$CODEX_HOME"; codex --version
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW OF A STRATEGY DOCUMENT, not of code.

Read docs/atmosphere/fv3_duo_jax_lane_strategy.md (HEAD). It is the contract
for porting this repo's certified NumPy FV3 duo-grid dynamical core to a JAX
lane that is jit-compilable, differentiable, and wired into the model. Also
read, for context: docs/atmosphere/fv3_duo_gap_register_2026-08-06.md,
packages/core/legoesm/core/fv3_nh_core.py (the JAX pattern-setter this
strategy generalises), and packages/core/legoesm/core/fv3_native_nh_core.py
(its NumPy authority).

Your job is to find where this strategy will FAIL IN PRACTICE. Be maximally
skeptical and CONCRETE (cite file:line or section). In particular:

1. THE AUTHORITY CHAIN. Section 2 certifies every JAX kernel against the NumPy
   lane, never against Fortran directly ("one authority per hop"). What class
   of defect does that HIDE? Specifically: if the NumPy lane has an error that
   the JAX lane faithfully reproduces, which tier catches it, and is tier 5
   (5-9 day integrations vs the oracle's atmos_daily.nc) actually sensitive
   enough to see it? Give a defect that would survive to production.

2. THE TOLERANCE POLICY (section 4). It classifies kernels as pointwise
   (~1e-15) vs accumulating (~1e-12) and requires bounds = measured x 3..10.
   Where does that classification break down for THIS core? Name kernels in
   fv3_native_d_sw.py / fv3_native_mapz.py that are neither cleanly pointwise
   nor cleanly accumulating. Is "measured x 10" a defensible margin when the
   measurement comes from ONE fixture? What is the failure mode of pinning a
   bound to a single measured value?

3. THE LITERAL-TRANSLATION RULE (R1). It permits four mechanical
   transformations. Find a case in this specific core where one of those four
   is NOT semantics-preserving. Consider especially: (a) "a for-loop over grid
   indices becomes a vectorized slice over the identical window" -- when is
   that false? (b) "an if on array DATA becomes jnp.where" -- what does that
   change about NaN/inf propagation and about gradients? (c) in-place mutation
   to functional return -- where does the NumPy lane rely on read-after-write
   WITHIN one routine such that a naive functional rewrite changes the answer?

4. THE TWO FLAGGED NODES (section 9). `_rezone` (a remap interval search) and
   `tracer_2d_1l_sixface` (nsplt is a data-derived loop trip count). Is the
   proposed handling of each actually workable under jit? For nsplt: what
   breaks with "resolve it outside the traced region and treat it as static"
   -- specifically, does that make the step function retrace, and does it make
   the model non-differentiable with respect to anything that feeds cmax?

5. DIFFERENTIABILITY (section 7). PPM limiters and upwind sign selections are
   named as the non-smooth sites. Is order-2 check_grads at a state "away from
   switching points" a meaningful gate for an ADVECTION scheme whose entire
   behaviour is switching? What would a better gate be?

6. THE VALIDATION LADDER (section 5) and the LIVE TWIN (section 6). Is any
   tier vacuous or unreachable as specified? The live twin claims "zero cost
   when off (a Python if on a static flag, resolved at trace time)" -- verify
   that claim is achievable given the kernels are jitted, and say what it
   costs when ON inside a jitted function (can it even run the NumPy twin
   there?).

7. WHAT IS MISSING. Name anything a port of this specific dynamical core needs
   that the strategy does not mention at all. Conservation gates? Halo-width
   invariants? The float32/float64 boundary? Something about the six-face
   list-of-arrays state representation vs a stacked (6, ...) array?

Output a numbered list of findings, each tagged BLOCKER / MAJOR / MINOR / NIT,
with a concrete suggested change to the document. If a concern turns out to be
unfounded after you check the code, SAY SO EXPLICITLY rather than padding the
list. I would rather have four real findings than fifteen plausible ones.
PROMPT
)" 2>&1
echo "codex rc=$?"
echo ALLDONE
