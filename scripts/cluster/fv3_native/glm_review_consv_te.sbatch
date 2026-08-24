#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_consv
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_consv_%j.log
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
ORACLE=/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean
P=$G/prompt_consv_${SLURM_JOB_ID}.txt
echo "host: $(hostname)"; cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"
{
cat <<'HDR'
You are the SECOND adversarial reviewer on a JAX + NumPy twin port of
the FV3 dynamical core. codex works the diff; you work ARCHITECTURE and
INVARIANTS. Do not rely on memory of FV3 -- everything is pasted. Mark
anything you infer; you were right to do that last round, and one of
your findings then was wrong precisely because the paste was
incomplete.

WHAT IS NEW. The total-energy fixer (`consv_te`) was the last refused
arm. It is now ported to both lanes and scored against a generated
Fortran deck: residual 1.1866e-09, response gate 1.347e-08, identical
on both backends.

THE STRUCTURAL POINT, which is where I most want your attention. In the
Fortran, one routine computes a per-column energy, reduces it GLOBALLY
across every tile, and applies the resulting scalar correction to
temperature -- all inside a single call, because there a "domain" is
the whole globe. In this port that routine is called ONCE PER FACE, so
the scalar cannot be known inside any single call. I therefore SPLIT
the routine: a flag makes it skip its final temperature conversion and
return early, the caller collects all six faces, reduces, and then
applies the conversion through a new small routine.

A. IS SPLITTING A CERTIFIED ROUTINE SAFE HERE? The routine is certified
   at 1.1866e-09 and the split moves its last operation outside it.
   What would make that NOT a pure refactor -- what has to be true
   about everything between the split point and the end?
B. THE SCALAR IS A GLOBAL REDUCTION OF A DIFFERENCE of two large column
   integrals (~1e9) whose result drives a ~4e-6 K correction. What does
   that mean for how much of the answer is real? What would you want
   measured before trusting it?
C. TWO INTEGRALS THAT LOOK ALIKE AND ARE NOT. One takes temperature and
   forms a virtual temperature itself, integrating the geopotential
   DOWNWARD; the other takes an already-virtual temperature and
   integrates UPWARD. They telescope to the same quantity when the
   surface term is zero. Is sharing their kinetic-energy term (which I
   did factor out) safe, and is there a failure mode where the two get
   confused that no test would catch?
D. THE JAX LANE REUSES THE NUMPY LANE'S INTEGRALS by converting traced
   arrays to numpy, on the argument that these run once per step
   outside any compiled region. What breaks that argument?
E. THE RESPONSE GATE is credited with certifying this, because the
   feature's signal is only ~12x the residual floor. Given B, is a
   response agreement of 1.347e-08 strong evidence, and what defect
   would pass both gates?
F. ANYTHING ASSERTED IN A COMMENT THAT NOTHING CHECKS.

Output MAJOR / MINOR with file:line, the concrete failure scenario, and
the smallest fix. Where you find nothing, say so explicitly.

================ THE COMMITS ================
HDR
git log --oneline -6
git show $(git log --format=%H --grep="port the consv_te energy fixer" -1)
git show $(git log --format=%H --grep="consv_te fixer in the JAX lane" -1)
echo "================ ORACLE fv_mapz.F90:620-750 (the fixer) ================"
sed -n '620,750p' $ORACLE/model/fv_mapz.F90
echo "================ ORACLE fv_mapz.F90:1120-1160 (compute_total_energy, hydro) ================"
sed -n '1120,1160p' $ORACLE/model/fv_mapz.F90
echo "================ ORACLE fv_mapz.F90:960-1005 (the :975 conversion) ================"
sed -n '960,1005p' $ORACLE/model/fv_mapz.F90
echo "================ ORACLE g_sum (fv_grid_utils.F90:2946-2996) ================"
sed -n '2946,2996p' $ORACLE/model/fv_grid_utils.F90
echo "================ PORT: the energy routines (spec lane) ================"
sed -n '/^# The consv_te energy fixer/,/^def fv_dynamics_step/p' packages/core/legoesm/core/fv3_native_dynamics.py
echo "================ PORT: close_out_pt + defer_close (spec) ================"
sed -n '/^def close_out_pt/,/^def lagrangian_to_eulerian/p' packages/core/legoesm/core/fv3_native_mapz.py
} > $P
echo "prompt bytes: $(wc -c < $P)"
$PY $REPO/scripts/validate/glm_review.py --prompt-file $P --model glm-5.2 \
    --max-tokens 48000 --out $G/review_consv_${SLURM_JOB_ID}.md
echo "rc=$?"; echo GLM_CONSV_DONE
