#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_moistjax
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_moistjax_%j.log
# Second adversarial reviewer on the moist (zvir) JAX port. GLM does NOT
# recall FV3 internals, so the prompt SHIPS the source: the commit diff,
# both lanes' moist paths, and the oracle lines they cite.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
ORACLE=/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/model
P=$G/prompt_moistjax_${SLURM_JOB_ID}.txt
echo "host: $(hostname)"
cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"

{
cat <<'HDR'
You are the SECOND adversarial reviewer on a JAX port of the FV3 duo-grid
dynamical core. Your partner (codex) works the diff line by line; your
job is the ARCHITECTURE and the INVARIANTS -- what the diff assumes about
the code around it, what it makes newly reachable, and what claim it
makes that nothing checks.

Do not rely on any memory of FV3. Everything you need is pasted below.

THE CHANGE: the NumPy lane (fv3_native_dynamics.py, the SPECIFICATION)
gained hydrostatic moist coupling last commit. This commit ports the same
feature into the JAX lane (fv3_dynamics.py, the PORT). Moist coupling
means: dp1 = zvir*q(sphum) enters the temperature-to-virtual-potential-
temperature conversion at the top of a dynamics step, and the inverse
factor is removed again inside the vertical remap at the bottom.

TWO ARMS STAY DELIBERATELY REFUSED and must keep raising: non-hydrostatic
moist (the NH pressure diagnostic pkz would need the same (1+dp1) factor
INSIDE its log argument) and consv_te (a total-energy fixer that is not
ported at all).

WHAT I MOST WANT FROM YOU, in this order:

A. THE INVARIANT THE CHANGE ASSUMES BUT DOES NOT STATE. The port forms
   dp1 ONCE, before the outer k_split loop, from the step-initial
   humidity. Inside that loop the tracers are TRANSPORTED and the remap
   REWRITES them. So dp1 is stale by construction after iteration 1.
   The Fortran does the same thing, so it is faithful -- but is anything
   else in the port silently assuming q and dp1 are consistent? Trace
   every consumer.

B. WHAT THIS MAKES NEWLY REACHABLE. Before this commit, zvir != 0 raised
   immediately, so every code path downstream of it in the JAX lane was
   DEAD. This campaign's single most expensive lesson is that unexercised
   code is unknown code: nine defects were found the first time modules
   ran, seven of them in shipped, reviewed, never-executed code. Which
   paths does this commit switch on for the first time, and which of them
   has NO gate? Name them.

C. THE REMAP HOP. The commit claims the closing division needed no work
   because the remap call already passes r_vir=zvir. That is a claim
   about a function the diff does not touch. Check it (fv3_mapz.py is
   pasted below), including whether the hard-coded adiabatic=True at
   that call site disables anything.

D. CAN THE NEW TESTS FAIL? For each new test, name the single edit that
   turns it red. A test that greps source, a parametrized value never
   passed, an assert that is true by construction -- all three have
   shipped in this campaign and all three were caught in review.

E. THE ANTI-VACUITY TEST specifically compares two SCALARS (a max-abs
   difference) across the two lanes and asserts they agree. Is that a
   meaningful cross-lane check, or can two genuinely different fields
   produce the same maximum? If it is weak, say what the strong version
   is.

Output MAJOR / MINOR findings with file:line, the concrete failure
scenario, and the smallest fix. Where you find nothing, say so
explicitly -- an honest "clean" is more useful than a manufactured
finding. Flag anything you are inferring rather than reading.

================ THE COMMIT ================
HDR
git show HEAD~1
echo "================ THE JAX PORT: fv3_dynamics.py (moist region) ================"
sed -n '185,230p;300,470p' packages/core/legoesm/core/fv3_dynamics.py
echo "================ THE NUMPY SPEC: fv3_native_dynamics.py (moist region) ================"
sed -n '195,230p;285,425p' packages/core/legoesm/core/fv3_native_dynamics.py
echo "================ THE REMAP'S MOIST GUARD: fv3_mapz.py ================"
sed -n '1380,1430p;1560,1600p' packages/core/legoesm/core/fv3_mapz.py
echo "================ ORACLE: fv_dynamics.F90:275-320 ================"
sed -n '275,320p' $ORACLE/fv_dynamics.F90
echo "================ ORACLE: fv_dynamics.F90:390-415 ================"
sed -n '390,415p' $ORACLE/fv_dynamics.F90
echo "================ ORACLE: fv_mapz.F90:960-985 ================"
sed -n '960,985p' $ORACLE/fv_mapz.F90
} > $P
echo "prompt bytes: $(wc -c < $P)"

$PY $REPO/scripts/validate/glm_review.py \
    --prompt-file $P --model glm-5.2 --max-tokens 48000 \
    --out $G/review_moistjax_${SLURM_JOB_ID}.md
echo "rc=$?"
echo GLM_MOISTJAX_REVIEW_DONE
