#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3jaxk_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/fv3jax_codex_kernels_%j.log
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
ADVERSARIAL CODE REVIEW of the new FV3 duo-grid JAX lane on this branch.

CONTEXT. This repo holds a loop-faithful NumPy fp64 port of the GFDL FV3
duo-grid dynamical core, certified against Fortran dumps. The commits under
review add a JAX twin of it. The contract is
docs/atmosphere/fv3_duo_jax_lane_strategy.md -- READ IT FIRST, especially
rules R1/R1a/R1b/R2/R4/R7, the tolerance policy (section 4), and the
validation ladder (section 5). Every JAX kernel is certified against the
NumPy lane, never against Fortran directly.

FILES UNDER REVIEW (run `git diff --stat main...HEAD` and `git show` as needed):
  packages/core/legoesm/grids/fv3_duo_halos.py     (exchanges, k2e ring remap,
      corner fills, BOTH duo flux-averaging barriers, ext_scalar/ext_vector)
  packages/core/legoesm/core/fv3_tp_core.py        (xppm/yppm/pert_ppm/
      fv_tp_2d/copy_corners/deln_flux/xtp_u/ytp_v)
  packages/core/legoesm/core/fv3_pgrad.py          (geopk/p_grad_c/one_grad_p/
      nh_p_grad/pk3_halo/pln_halo/pe_halo + a private a2b_ord4)
  packages/core/legoesm/core/fv3_mapz.py           (Lagrangian_to_Eulerian and
      the PPM profile/limiter/rezone chain)
  packages/core/legoesm/core/fv3_nh_core.py        (existing 5 NH kernels +
      a newly added update_dz_d)
  their tests under tests/grids/

Their NumPy authorities are the same-named fv3_native_* modules. THE NUMPY
LANE IS THE SPECIFICATION: any behavioural difference is a port bug by
definition, so diff them routine by routine.

BE MAXIMALLY SKEPTICAL AND CONCRETE (cite file:line). Priorities, in order:

1. BATCHING LEGALITY. fv3_duo_halos.py converts the NumPy lane's sequential
   scalar-indexed writes into grouped `.at[idx].set()` scatters, licensed by
   build-time checks `_check_batchable` (dst n src == empty) and
   `_check_no_clobber` (phase-hoist legality), with `_dedup_last` resolving
   repeated destinations last-wins. Attack this hard:
   - Are the destination and source index sets it computes actually the ones
     the scatter uses at run time, or could a code path build indices the
     check never sees?
   - `.at[].set()` with duplicate indices is UNDEFINED in JAX. Is `_dedup_last`
     applied on EVERY path that can produce duplicates? Find one that skips it.
   - The module claims the corner-region Lagrange fill is ONE independent group
     because `diag` reads only the incoming field. Verify that against
     fv3_native_ext_vector.py `_CornerLagrange.fill`/`_apply_dir`/`diag`. If it
     is wrong the whole corner fill is silently mis-ordered.
   - Is the `n >= 4` guard sufficient for the claimed non-collision of the X+/X-
     abscissa windows at every stagger?

2. THE TWO DUO BARRIERS. dyn_core.F90:872 (CGRID_NE) and :984 (BGRID_NE) in the
   pinned tree. Barrier 1 must average slots iq==1, iq==4 and iq>4 and must NOT
   average slot 2 (w) or slot 3 (q_con) -- dyn_core.F90:856. Confirm the JAX
   barrier reproduces that selection, the 0.5*(mine+neighbour) blend, the edge
   partner topology, and the compute-ring index origin. A barrier that averages
   the wrong slot or picks the wrong neighbour is the single highest-impact
   defect available here and would still look plausible.

3. R1b -- jnp.where. There are ~100 `jnp.where` sites in fv3_tp_core.py and ~47
   in fv3_mapz.py. `where` evaluates BOTH branches. Find sites where the
   unselected branch can produce NaN/Inf (a division, log, sqrt, or an index
   clamped into a wrong range) and where that NaN can reach the primal or, via
   reverse-mode AD, the gradient of the SELECTED branch. Name the ones that
   need the double-where / mask-before-select idiom and do not have it.

4. LITERAL TRANSLATION. Diff each JAX routine against its NumPy authority for:
   dropped or altered index windows; a loop vectorized where an iteration reads
   a slot an earlier iteration writes; a k-recurrence turned into cumsum or
   associative_scan; a limiter branch reordered; a guard silently softened; a
   scheme dispatch (hord / nord / kord / stagger) that falls through to a
   default instead of raising.

5. _rezone (fv3_mapz.py) specifically. It is meant to be a static-length nested
   lax.scan carrying (found, k0, qsum, out), ascending traversal, explicit
   first-match tie rule, and an explicit no-bracket path. Verify the tie rule
   and the k0 carry match fv3_native_mapz.py:594-669 exactly, and that a
   clamped read is only ever consumed behind `found`.

6. TESTS. Are the gates NON-VACUOUS? For each: would it still pass if the
   feature under test were deleted? Do the guard tests actually exercise the
   guard? Do the gradient gates prove anything on a LINEAR operator (order-2
   check_grads on a linear map is trivially satisfied -- say where that makes a
   gate worthless and what should replace it)? Is any tolerance bound loose
   enough to hide a real defect? Note: bounds still marked TOL-PENDING are
   provisional by design and are being replaced from measurement -- do not
   report those as findings, but DO report a gate whose STRUCTURE is wrong.

7. AD / JIT / pytree hygiene: donate_argnums anywhere (forbidden); Python
   control flow on a traced value; a static argument that should be traced or
   the reverse; anything that forces a retrace per step; f64 gates that read a
   traced value instead of a static dtype.

Output a numbered list of findings, each tagged BLOCKER / MAJOR / MINOR / NIT,
with file:line and a concrete fix. If a concern is unfounded after checking,
say so explicitly rather than padding. Four real findings beat fifteen
plausible ones.
PROMPT
)" 2>&1
echo "codex rc=$?"
echo ALLDONE
