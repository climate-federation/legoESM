#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_moistnh
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_moistnh_%j.log
# Second reviewer on the NH moist arm. GLM does not recall FV3, so the
# prompt SHIPS the source: the diff, both lanes' changed regions, the
# tests, and the oracle lines they cite.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
ORACLE=/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean/model
P=$G/prompt_moistnh_${SLURM_JOB_ID}.txt
echo "host: $(hostname)"
cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"

{
cat <<'HDR'
You are the SECOND adversarial reviewer on a JAX + NumPy twin port of
the FV3 dynamical core. Your partner (codex) works the diff line by
line; your job is ARCHITECTURE and INVARIANTS -- what the change assumes
about the code around it, what it makes newly reachable, and what claim
it makes that nothing checks.

Do not rely on any memory of FV3. Everything is pasted below.

THE CHANGE: moist (virtual-temperature) coupling was already enabled on
the HYDROSTATIC arm of both lanes. This change enables it on the
NON-HYDROSTATIC arm. Three things happen:

(1) On the NH arm the pressure diagnostic `pkz` is now RECOMPUTED at the
    top of the step with an extra (1+dp1) factor inside its log, instead
    of being taken from the caller. Justification: the Fortran does that
    recompute on every call.
(2) The remap's closing "divide the virtual factor back out" step turns
    out to be guarded, in the Fortran, by `if (.not. adiabatic)` on the
    NH arm only. This lane always divides. So the combination
    (non-hydrostatic AND adiabatic=true AND moist) is now REFUSED, and
    the dynamics passes adiabatic=false on the moist NH arm.
(3) Tracer-index guards were widened to accept numpy integers while
    still rejecting booleans.

WHAT I MOST WANT FROM YOU:

A. IS (2) SOUND? It is the load-bearing claim and it rests on reading
   one Fortran guard. The argument is: our expression equals the
   Fortran's `.not. adiabatic` expression because `dtmp` is identically
   zero when the energy fixer is off, so passing adiabatic=false is
   FAITHFUL rather than a workaround. Check the reasoning, and check
   whether flipping that flag can reach anything else.

B. WHAT (1) MAKES NEWLY REACHABLE, and what is now DOUBLE-COMPUTED.
   The caller still builds a pressure bundle; the step now overwrites
   part of it. Is there any consumer that reads the caller's pkz
   between construction and the overwrite, or any consumer that assumes
   the whole bundle is self-consistent (pe/peln/pk/pkz from one
   delp)? Trace it.

C. THE MULTI-STEP QUESTION. The step returns a pressure bundle that the
   driver threads into the next step. On the moist NH arm the returned
   pkz comes from the remap, which writes the DRY form. The next step
   overwrites it. Is that safe for every path -- including a path that
   stops between steps (a checkpoint, a diagnostic) and reads pkz?

D. CAN THE NEW TESTS FAIL? For each, name the single edit that turns it
   red. One of them substitutes a different pkz into the same step and
   requires the answer NOT to move, with a hydrostatic control that
   requires it TO move. Say whether that control genuinely rules out a
   lane that ignores pkz everywhere.

E. ANYTHING ASSERTED IN A COMMENT THAT NOTHING CHECKS.

Output MAJOR / MINOR with file:line, the concrete failure scenario, and
the smallest fix. Where you find nothing, say so explicitly. Flag
anything you are inferring rather than reading.

================ THE COMMITS ================
HDR
git log --oneline -8
git show ${REVIEW_RANGE:-HEAD~1..HEAD}
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
echo "---- mapz guard tests ----"
sed -n '/def test_moist_nonhydrostatic_refuses_the_adiabatic_flag/,/^# ---* gate 2/p' tests/grids/test_fv3_mapz.py
} > $P
echo "prompt bytes: $(wc -c < $P)"

$PY $REPO/scripts/validate/glm_review.py \
    --prompt-file $P --model glm-5.2 --max-tokens 48000 \
    --out $G/review_moistnh_${SLURM_JOB_ID}.md
echo "rc=$?"
echo GLM_MOISTNH_REVIEW_DONE
