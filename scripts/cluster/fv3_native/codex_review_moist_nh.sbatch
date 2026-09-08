#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_moistfix
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_moistfix_%j.log
# Adversarial review of the NON-hydrostatic moist arm + the adiabatic-flag
# finding. REVIEW_RANGE selects the commits (default: the last two).
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
FINAL ADVERSARIAL REVIEW of the moist arms on the FV3 duo-grid port.
FIND THE COMMITS: `git log --oneline -8`. The work under review is the
range from the commit whose subject starts "feat(fv3): the
non-hydrostatic moist arm" through HEAD. Do NOT assume HEAD is the
implementation commit -- review runners get committed on top, and
reviewing the wrong commit has already happened in this campaign.

CONTEXT YOU NEED. Moist (virtual-temperature) coupling now runs on both
the hydrostatic and non-hydrostatic arms of both lanes
(fv3_native_dynamics.py = the NumPy SPECIFICATION, fv3_dynamics.py =
the JAX PORT). The hydrostatic arm is now ORACLE-CERTIFIED against
run_hydro_1step_moist_gfs at worst one-step rel 1.1866e-09 on both
backends, with the IC instrument control at 3.685e-14. The pinned
Fortran is at
/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/.

TWO EARLIER ROUNDS FOUND NO BLOCKER AND CONFIRMED THE FORTRAN CLAIMS.
Do not re-litigate those unless you find them wrong. Attack what is NEW
and what the last round could not see:

1. THE LATEST COMMIT RETRACTS A CLAIM I MADE IN CODE COMMENTS. Four
   places said a `dp1=zeros` array is NOT bit-identical to the
   `dp1=None` branch ("rounds twice", "a reciprocal then a multiply").
   That was true of the OLD association `pt *= (1.0+dp1)/pkz` and false
   after the correction to `(pt*(1+dp1))/pkz`, because 1.0+0.0 is
   exactly 1.0 and win*1.0 is exact. VERIFY THE RETRACTION IS ITSELF
   CORRECT -- is the equality exact for EVERY finite input, including
   subnormals, negative zero, and any pkz the lane can produce? If
   there is an input class where it does NOT hold, the new assertion
   (`assert equal`) is a latent flake and the old comment was
   half-right.

2. THE NEW DISCRIMINATOR in test_nh_moist_pkz_matches_the_oracle_expression.
   It replaced a bitwise cross-lane assert (which failed by 1 ulp on an
   exp/log chain) with: measure the DRY arm's port-vs-spec residual
   too, and require the moist arm's to be within 10x of it. Is that a
   real discriminator or a rubber stamp? Specifically: could a
   genuinely misplaced (1+dp1) -- say, applied OUTSIDE the log, or to
   pt at the call site instead of inside -- still land within 10x of
   the dry residual? If so the gate is decorative; say what would not
   be.

3. THE SUBSTITUTION GATE'S NEW PERTURBATION. It now scales the caller's
   pkz by 1.001 instead of building an NH pressure bundle (which raised
   KeyError: delz on the hydrostatic control). Does a 1.001 scaling
   actually reach every consumer a genuinely different pkz would? Is
   there any path where a multiplicative perturbation cancels?

4. THE ORACLE-CERTIFICATION HARNESS, which no reviewer has seen yet:
   scripts/validate/fv3_native/full_step_oracle_parity.py's new
   --moist flag, check_physics_is_inert, check_moist_deck, the
   _nml_logical last-assignment-wins parser, and build_port_ic's new
   delegation to dcmip16_bc_six_face_state. The claim being certified
   is 1.1866e-09 against a real Fortran run, so a defect here
   manufactures a confident wrong result. Attack: can the namelist
   parser miss a switch (continuation lines, array syntax, repeated
   groups, tab separators)? Is the switch LIST complete for
   "fv_phys cannot touch the state"? Does build_port_ic still produce
   byte-identical output to the code it replaced on the DRY path (the
   dry parity still reads 1.1866e-09, which is evidence -- is it
   proof)?

5. THE MOIST IC ITSELF: dcmip16_bc_six_face_state's new zvir arm
   divides pt by (1 + zvir*q) at the END of the per-face body, after
   delz. Verify the ORDER against test_cases.F90:6721/6737/6760 and
   confirm the division uses the same q the caller receives.

6. ANY COMMENT OR DOCSTRING ADDED IN THIS RANGE THAT IS NOT TRUE OF THE
   CODE. One false comment already survived two review rounds in this
   very change; assume there is another.

Output BLOCKER / MAJOR / MINOR with file:line, the concrete failure
scenario, and the smallest fix. Say explicitly where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"
echo CODEX_MOISTNH_REVIEW_DONE
