#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_nhorc
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_nhorc_%j.log
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
PIN=/burg-archive/glab/users/pg2328/fv3_oracle_pinned
P=$G/prompt_nhorc_${SLURM_JOB_ID}.txt
echo "host: $(hostname)"; cd $REPO && echo "PINNED sha: $(git rev-parse HEAD)"
{
cat <<'HDR'
You are the SECOND adversarial reviewer on a JAX + NumPy twin port of
the FV3 dynamical core. Your partner (codex) works the diff; your job is
ARCHITECTURE and INVARIANTS. Do not rely on memory of FV3 -- everything
is pasted. Mark anything you infer.

WHAT IS NEW, AND WHY IT NEEDS YOU. The non-hydrostatic MOIST arm had no
Fortran reference, so it rested on source reading plus port-vs-spec
parity -- which, as you pointed out last round, a SYMMETRIC edit to both
lanes would not catch. So a reference was GENERATED: copy the existing
non-hydrostatic decks, flip ONE namelist flag (adiabatic .true. ->
.false.), run the Fortran, score the port against the result. It scored
6.6116e-04 on both backends, which is EXACTLY the non-hydrostatic DRY
arm's own standing figure.

THIS IS A SELF-GENERATED ORACLE. If the deck is wrong, the port is being
certified against a number its own author manufactured, and it looks
like a pass. That is the thing to attack.

A. IS THE ONE-FLAG ARGUMENT SOUND? The claim is that in the solo driver
   `adiabatic = .false.` is precisely what turns moisture on -- it sets
   zvir and moist_phys together -- so flipping it and nothing else
   yields the intended experiment. What ELSE could that flag change in a
   run, and is anything in the copied deck now internally inconsistent?
   Note `phys_hydrostatic` was left at its non-hydrostatic value.

B. IS 6.6116e-04 A REAL RESULT OR AN ARTEFACT? It equals the dry arm's
   number to five digits. The anti-vacuity argument offered is that the
   two decks' own one-step temperature tendencies differ by ~1%
   (0.05833/0.002619/0.04616 dry vs 0.058/0.002653/0.04603 moist) and
   that the port reproduces each deck's own tendency. Is that sufficient
   to rule out the port silently running DRY against the moist deck? If
   not, what measurement would?

C. THE HARNESS NOW HAS FOUR DECK PAIRS (hydro/NH x dry/moist) selected
   by two independent flags. Can any flag combination silently pair
   mismatched runs -- a moist initial condition with a dry step
   reference, or a non-hydrostatic IC with a hydrostatic step? Such a
   pairing would not raise; it would just report a number.

D. WHAT DOES A GENERATED ORACLE NEED THAT A SHIPPED ONE DOES NOT? The
   pinned tree was previously read-only ground truth; this change writes
   new directories into it. What provenance or immutability is missing,
   and what happens on a re-run or a partial failure?

E. ANYTHING ASSERTED IN A COMMENT OR IN THE STATE DOC THAT NOTHING
   CHECKS. Every prior round found one false statement; assume one.

Output MAJOR / MINOR with file:line, the concrete failure scenario, and
the smallest fix. Where you find nothing, say so explicitly.

================ THE COMMITS ================
HDR
git log --oneline -6
git show $(git log --format=%H --grep="build the missing NH moist oracle" -1)
git show $(git log --format=%H --grep="NH moist arm is ORACLE-CERTIFIED" -1)
echo "================ THE GENERATED DECK vs ITS SOURCE ================"
echo "--- diff run_nh_1step_gfs -> run_nh_1step_moist_gfs (input.nml) ---"
diff $PIN/run_nh_1step_gfs/input.nml $PIN/run_nh_1step_moist_gfs/input.nml
echo "--- diff field_table (empty means identical) ---"
diff $PIN/run_nh_1step_gfs/field_table $PIN/run_nh_1step_moist_gfs/field_table
echo "--- diff vs the CERTIFIED hydrostatic moist deck ---"
diff $PIN/run_hydro_1step_moist_gfs/input.nml $PIN/run_nh_1step_moist_gfs/input.nml
echo "--- what the generated run's OWN logfile resolved ---"
grep -iE "^ *(HYDROSTATIC|ADIABATIC|NPZ|K_SPLIT|N_SPLIT|DUOGRID|CONSV_TE|FV_SG_ADJ|PHYS_HYDROSTATIC) |FMSConstants" $PIN/run_nh_1step_moist_gfs/logfile.000000.out | head -20
echo "================ THE HARNESS REDIRECT + CHECKERS ================"
sed -n '/^def _nml_value/,/^def build_port_ic/p' scripts/validate/fv3_native/full_step_oracle_parity.py
sed -n '/if args.nh:/,/^    from legoesm/p' scripts/validate/fv3_native/full_step_oracle_parity.py
} > $P
echo "prompt bytes: $(wc -c < $P)"
$PY $REPO/scripts/validate/glm_review.py --prompt-file $P --model glm-5.2 \
    --max-tokens 48000 --out $G/review_nhorc_${SLURM_JOB_ID}.md
echo "rc=$?"; echo GLM_NHORC_DONE
