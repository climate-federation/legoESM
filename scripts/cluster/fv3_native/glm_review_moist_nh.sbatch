#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_moistfix
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_moistfix_%j.log
# Second reviewer on the NH moist arm. GLM does not recall FV3, so the
# prompt SHIPS the source: the diff, both lanes' changed regions, the
# tests, and the oracle lines they cite.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
ORACLE=/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/model
P=$G/prompt_moistfix_${SLURM_JOB_ID}.txt
echo "host: $(hostname)"
cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"

{
cat <<'HDR'
You are the SECOND adversarial reviewer on a JAX + NumPy twin port of
the FV3 dynamical core. Your partner (codex) works the diff line by
line; your job is ARCHITECTURE and INVARIANTS.

Do not rely on any memory of FV3. Everything is pasted below. If
something you need is NOT pasted, say so and mark the finding inferred
-- you did that last round and it was the right call.

WHERE THIS STANDS. Moist (virtual-temperature) coupling now runs on
both the hydrostatic and non-hydrostatic arms of both lanes, and the
HYDROSTATIC arm is now scored against a real Fortran run at worst
one-step relative 1.1866e-09 on both backends -- the same figure the
dry arm carries. Two review rounds found no blocker.

YOUR LAST ROUND'S MAJOR WAS RIGHT AND IS FIXED: the commit shipped a
STATE.md paragraph asserting the moist oracle deck was unusable while
its own code and checkers encoded the opposite. The measurement settled
it in favour of the code; the paragraph is now a recorded retraction.

WHAT I WANT THIS ROUND -- the things no reviewer has seen:

A. THE CERTIFICATION HARNESS ITSELF. Nobody has reviewed it, and it is
   what produces the 1.1866e-09. It adds: a --moist flag; two deck
   checkers that refuse an oracle run whose physics could have touched
   the state, or that does not resolve adiabatic=.false.; a namelist
   parser; and a de-duplication of the IC builder. A defect in any of
   these manufactures a confident wrong certification rather than an
   error. Attack the parser especially -- what namelist syntax does it
   mis-read?

B. A RETRACTION INSIDE THE DIFF. I asserted in four places that a
   zeros-valued moisture array is NOT bit-identical to the dedicated
   dry branch. It is identical, because multiplying by exactly 1.0 is
   exact. Is the retraction itself airtight for every input the lane
   can produce? An assertion of exact equality is a latent flake if
   there is any input class where it fails.

C. A TOLERANCE THAT REPLACED A BITWISE ASSERT. A cross-lane bitwise
   check on an exp(kappa*log(...)) chain failed by one ulp, so it now
   measures the DRY arm's residual too and requires the moist arm to be
   within 10x. Is that a real discriminator, or does it merely look
   like one? What misplacement of the moisture factor would still
   pass?

D. WHAT IS STILL UNCERTIFIED, and whether the docs say so honestly:
   the NON-hydrostatic moist arm has no oracle deck and rests on
   source-reading plus port-vs-spec parity -- which, as you noted last
   round, a SYMMETRIC edit to both lanes would not catch.

E. ANYTHING ASSERTED IN A COMMENT THAT NOTHING CHECKS. One false
   comment survived two rounds in this change. Assume another.

Output MAJOR / MINOR with file:line, the concrete failure scenario, and
the smallest fix. Where you find nothing, say so explicitly.

================ THE COMMITS ================
HDR
git log --oneline -8
# Resolve by SUBJECT, never by HEAD~N: the review runner itself gets
# committed on top, so a positional default reviews the wrong commit
# the moment anything lands after it (GLM MINOR, job 9442483).
BASE=$(git log --format=%H --grep="non-hydrostatic moist arm" -1)
RANGE=${REVIEW_RANGE:-${BASE}~1..HEAD}
echo "REVIEWING: $RANGE"
git show $RANGE
echo "================ JAX PORT: fv3_dynamics.py (changed regions) ================"
sed -n '215,270p;300,375p;415,480p' packages/core/legoesm/core/fv3_dynamics.py
echo "================ NUMPY SPEC: fv3_native_dynamics.py (changed regions) ================"
sed -n '228,280p;300,360p;405,450p' packages/core/legoesm/core/fv3_native_dynamics.py
echo "================ JAX mapz guards ================"
sed -n '1395,1450p' packages/core/legoesm/core/fv3_mapz.py
echo "================ NUMPY mapz guards ================"
sed -n '845,900p' packages/core/legoesm/core/fv3_native_mapz.py
echo "================ ORACLE fv_dynamics.F90:280-345 ================"
sed -n '280,345p' $ORACLE/fv_dynamics.F90
echo "================ ORACLE fv_mapz.F90:955-1000 ================"
sed -n '955,1000p' $ORACLE/fv_mapz.F90
echo "================ THE NEW TESTS ================"
sed -n '/^ZVIR = /,/^# ---* *$/p' tests/grids/test_fv3_dynamics.py | head -260
echo "================ THE CERTIFICATION HARNESS ================"
sed -n '/^def build_port_ic/,/^# ---* *$/p' scripts/validate/fv3_native/full_step_oracle_parity.py | head -140
sed -n '/^def dcmip16_bc_six_face_state/,$p' packages/core/legoesm/core/fv3_native_dcmip16_ic.py
echo "---- mapz guard tests ----"
sed -n '/def test_moist_nonhydrostatic_refuses_the_adiabatic_flag/,/^# ---* gate 2/p' tests/grids/test_fv3_mapz.py
} > $P
echo "prompt bytes: $(wc -c < $P)"

$PY $REPO/scripts/validate/glm_review.py \
    --prompt-file $P --model glm-5.2 --max-tokens 48000 \
    --out $G/review_moistfix_${SLURM_JOB_ID}.md
echo "rc=$?"
echo GLM_MOISTNH_REVIEW_DONE
