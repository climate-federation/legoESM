#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_fixes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_fixes_%j.log
# Adversarial review of the fixes made after the modules 4-6 review.
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
ADVERSARIAL REVIEW of a day's worth of defect fixes on the FV3 duo-grid
JAX port. Read docs/atmosphere/fv3_duo_jax_lane_STATE.md first -- its top
section lists what was fixed and why. The NumPy twin of each module
(fv3_native_*) is the SPECIFICATION; any behavioural difference is a port
bug by definition.

Review this range: `git log --oneline main..HEAD` and the diffs therein.
All six modules' gates are now green (27/16/19/18 plus the earlier
phases), so the question is NOT "does it pass" -- it is "does it pass for
the right reason, and did any fix introduce a new defect or paper over
one".

ATTACK IN THIS ORDER, cite file:line.

1. THE FIXES THAT CHANGED NUMERICS. Each of these altered what the model
   computes; verify each against the spec rather than against
   plausibility:
   - fv3_dynamics.p_var_hydrostatic: the k-recurrence scan axis changed
     from (0,3,1,2) to (3,0,1,2) and the three consumer transposes with
     it. Are pe/pk/peln/pkz now EXACTLY the spec's layouts, element for
     element, including the (i,k,j) ones? A transposed-but-symmetric
     fixture would hide an error here.
   - fv3_acoustic_3d's gz seed: delz is no longer re-windowed and the
     scan now takes ys rather than the carry. Confirm against
     dyn_core.F90:384-416 that ys[k] really is gz at level k under
     reverse=True, and that the zs seed lands at the right interface.
   - fv3_tracer2d's frac rescale now indexes [:, :, 0:n, k] and
     [:, 0:n, :, k]. Check both against the spec's per-face ranges
     (fv3_native_tracer2d.py:328-333), including which axis is the
     halo-padded one.
2. THE SEAM FIXES. Three places where two modules disagreed about a
   data shape were bridged rather than unified: flux capacitors (flat
   vs nested), tracers (face-major vs tracer-major), the NH carry
   (nested under "nh"). For each: is the bridge correct in BOTH
   directions, and is there any remaining caller still using the old
   shape? Grep, do not assume.
3. THE JIT FACTORY. make_fv_dynamics_step_jit now bakes the deck
   constants and re-attaches two non-array leaves outside the trace. Is
   the static set exactly right -- anything baked that a caller would
   legitimately vary per call, or anything still traced that a callee
   takes float() of? Does the re-attached metadata match what the eager
   path returns in EVERY configuration (km <= 4 vs km > 4)?
4. THE CARRY CONTRACTS. fv_dynamics_step's k_split carry gained
   nsplt/nsplt_exceeded, one early-return branch was missed and fixed,
   and the state dict now carries through keys the remap does not own.
   Enumerate every return path and confirm they agree in element count,
   dict keys and container types. Is nsplt_exceeded OR-accumulated
   correctly across k_split, and can check_nsplt_schedule still see it?
5. THE GATE HARNESS, which is where the worst defect of the day lived.
   _stack_dsw_np now REFUSES a name that exists in both generations and
   callers declare face_level_wins=("pt","delp","w","allflux_x",
   "allflux_y"). Verify each of those five is genuinely consumed
   face-level by the tail phases -- one wrong entry silently certifies
   the wrong generation again, which is exactly what happened before.
6. THE TOLERANCES that stopped being TOL-PENDING: nh carry zh at 3.5e-9
   (measured 3.498e-10) and update_dz_d at 1.4e-8 (measured 1.333e-09).
   Are those defensible as accumulation floors for a vertical
   recurrence in fp64, or are they hiding a real difference? Say which.
7. ANY COMMENT ADDED TODAY THAT IS NOT TRUE OF THE CODE. Several fixes
   carry long explanatory comments asserting why something is correct;
   this repo treats those as testable claims.

Output BLOCKER / MAJOR / MINOR with file:line, the concrete failure, and
the smallest fix. Say explicitly where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"
echo CODEX_FIXES_REVIEW_DONE
