#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_uph
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_uph_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/burg-archive/home/pg2328/.codex
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
RE-REVIEW (round 2) confirming two prior findings are addressed. ADVERSARIAL REVIEW of a GLM-authored port: commit a2c8db513 (fixes to fac9fd345) --
fv_update_phys_dry_duo (+ _jax twin) in
packages/core/legoesm/core/fv3_native_physics_coupling.py, and its 5 new tests
in tests/grids/test_fv3_physics_coupling.py (the ones named *orchestrator*).
Cite file:line, classify BLOCKER / MAJOR / MINOR, give the smallest fix, and
say explicitly where you found NOTHING.

CONTEXT. This is the DRY, HYDROSTATIC, moist_phys=.false. restriction of FV3's
fv_update_phys (pinned oracle
/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/model/fv_update_phys.F90).
It applies Held-Suarez tendencies (t_dt,u_dt,v_dt) back onto the post-dynamics
state. Per level k, the Fortran hydrostatic branch does:
    pt = pt + t_dt*dt*con_cp/cvm      (con_cp=cp_air; cvm from moist_cp)
    ua = ua + dt*u_dt ; va = va + dt*v_dt      (#ifndef GFS_PHYS block)
then one call to update_dwinds_phys (D-grid increment from the A-grid tendency).
No p_var (delp unchanged, no physics tracers). The port composes the already
certified update_dwinds_phys_duo once per level.

ATTACK SPECIFICALLY:
1. THE cp FACTOR = 1.0 CLAIM. The port drops con_cp/cvm entirely (pt += t_dt*dt).
   Its justification: run_hs_1step_gfs/input.nml pins nwat=0, so moist_cp
   (fv_mapz.F90:3656, case default) returns cpm=cp_air. VERIFY: grep nwat in
   that input.nml; confirm moist_cp's default branch is cpm=cp_air; confirm
   fv_update_phys's hydrostatic branch (around fv_update_phys.F90:366-372) calls
   moist_cp with nwat=flagstruct%nwat and NOT some other capacity. Is there ANY
   path where the inert sphum in the tracer restart (max 1.76e-2) reaches cvm
   despite nwat=0? If the factor is genuinely 1.0, say so; if not, it is a BLOCKER.
2. THE n = pt.shape[0] - 2*ng RECOVERY. update_dwinds_phys_duo takes the CELL
   count n and derives m=n+2*ng internally. Confirm pt is the full data domain
   (m,m,npz) so n=m-2*ng is right, and that an off-by-one here would be caught
   by the tests (it was: the pre-fix bug gave shape (11,12,3) vs (11,11,3)).
3. LEVEL-LOOP EQUIVALENCE. The Fortran updates pt/ua/va inside its k-loop then
   calls update_dwinds_phys ONCE (with its own internal k-loop). The port
   computes pt/ua/va (whole-array) then loops levels calling the 2-D helper.
   Is this exactly equivalent? In particular, does the D-grid update read the
   RAW u_dt/v_dt (correct) or accidentally the updated ua/va? Do the metrics
   (vlon/vlat/es1/ew2) legitimately not vary with k?
4. JAX TWIN. Is fv_update_phys_dry_duo_jax numerically identical to the numpy
   authority? The level loop is a Python unroll over static npz (not lax.scan) --
   is that jit-safe and free of tracer leaks? static_argnums=(13,) for ng in the
   jit test -- correct? Any float32 promotion?
5. THE TESTS. Are the 5 orchestrator tests non-vacuous -- would
   test_orchestrator_dgrid_is_the_per_level_helper actually FAIL on a mis-indexed
   loop, and does test_orchestrator_differentiable probe nonzero gradients?
   atol=1e-12 defensible for the jax-vs-numpy identity?
6. Anything dropped from fv_update_phys that IS live for dry+hydrostatic+nwat=0
   (a halo fill, a diagnostic, a delz touch) that the port silently omits?
PROMPT
)" 2>&1
echo "CODEX_UPH_REVIEW_DONE"
