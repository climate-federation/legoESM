#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_moistnh
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_moistnh_%j.log
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
ADVERSARIAL REVIEW of the NON-HYDROSTATIC moist arm on the FV3 duo-grid
port. FIND THE COMMITS FIRST: `git log --oneline -6`; the ones under
review have subjects starting "feat(fv3): the non-hydrostatic moist arm"
and "fix(fv3): fold codex and GLM". Do NOT assume HEAD is the right
commit -- review runners get committed on top, and reviewing the wrong
commit has already happened once in this campaign.

THE AUTHORITIES:
1. packages/core/legoesm/core/fv3_native_dynamics.py and
   fv3_native_mapz.py are the NumPy SPECIFICATION; fv3_dynamics.py and
   fv3_mapz.py are the JAX PORT. Divergence between the pair is a port
   bug BY DEFINITION -- but note that this change edits BOTH lanes, so
   an error can be symmetric and still wrong. Check both against the
   Fortran, not only against each other.
2. The pinned Fortran at
   /burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/model/
   -- fv_dynamics.F90:295-343 (the NH dp1/pkz block and its moist_phys
   .true./.false. arms), fv_mapz.F90:960-995 (the closing conversion and
   its `if (.not. adiabatic)` guard), fv_mapz.F90:479-481 (the pkz the
   remap itself writes).

THE FOUR CLAIMS THIS CHANGE MAKES. Attack each; cite file:line.

1. "On the NH arm the oracle RECOMPUTES pkz at the top of every
   fv_dynamics call, so the caller's pkz must not be trusted." Verify
   against :295-343. Then check the port actually recomputes it from the
   STEP-ENTRY delp/pt/delz with pt still TEMPERATURE, not after any
   mutation, and that the recompute is skipped on the dry arm so the
   certified dry NH parity cannot have moved.
2. "The moist NH pkz is exp(kappa*log(rdg*delp*pt*(1.+dp1)/delz)) with
   the factor INSIDE the log in the Fortran's left-to-right
   association." Check the expression AND the association in both
   lanes, and check MOIST_CAPPA / USE_COND are genuinely undefined in
   this build (if either is defined the ported arm is the wrong one).
3. "moist_phys = .false. is not a third lane -- the oracle forces
   dp1 = 0 there, so it IS the zvir = 0 lane and needs no flag."
   Verify at :333-337, and verify moist_phys reaches nothing else this
   lane models (:364, :546, fv_mapz.F90:129/:1168/:1205).
4. THE RISKIEST ONE: "fv_mapz.F90:985 puts the NH closing T_v -> T
   divide inside `if (.not. adiabatic)`, so adiabatic=.true. with
   r_vir != 0 makes the oracle SKIP the conversion while this lane
   always divides; therefore refuse that combination and pass
   adiabatic=False on the moist NH arm, where dtmp is identically 0
   because consv = 0." Check EVERY step of that: the guard's true
   extent, whether dtmp really is 0 when consv = 0 (fv_mapz.F90:628 and
   its dtmp initialisation), whether cv_air vs cp matters once dtmp is
   0, and whether flipping `adiabatic` to False reaches any OTHER
   branch in either the oracle or this port. If the conclusion is
   wrong, the port now runs a different branch than the oracle on every
   moist NH step.

THEN THE USUAL:
5. CAN THE NEW TESTS FAIL? For each new test in
   tests/grids/test_fv3_dynamics.py and tests/grids/test_fv3_mapz.py,
   name the single code edit that turns it red. Pay attention to
   test_nh_moist_pkz_is_recomputed_not_trusted, whose whole design is
   "substitute a different pkz and require the answer not to move" plus
   a hydrostatic control -- say whether the control genuinely proves the
   NH half is not vacuous.
6. THE GUARD ORDER. The zvir block now raises ValueError for a missing
   delz on the NH arm. Does it fire before anything indexes state, and
   does any earlier guard raise the same type first (which would make
   the parametrized refusal case pass for the wrong reason)?
7. The sphum_index guards were widened from isinstance(int) to
   __index__ with an explicit bool exclusion, in FOUR places. Is that
   equivalent-or-stricter everywhere, and does any caller pass
   something that now newly passes or newly fails?
8. ANY COMMENT OR DOCSTRING ADDED HERE THAT IS NOT TRUE OF THE CODE,
   including the claim that adiabatic and moist_phys "reach nothing
   else this lane models".

Output BLOCKER / MAJOR / MINOR with file:line, the concrete failure
scenario, and the smallest fix. Say explicitly where you found nothing.
PROMPT
)"
echo "CODEX RC=$?"
echo CODEX_MOISTNH_REVIEW_DONE
