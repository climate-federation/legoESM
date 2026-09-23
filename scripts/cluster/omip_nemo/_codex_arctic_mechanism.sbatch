#!/bin/bash
#SBATCH --job-name=cdx_arctic
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_arctic_%j.out
# CLAIM review BEFORE code (CLAUDE.md ACCURACY FIRST): the Arctic/Siberian SSS
# mechanism, measured today. No fix written yet.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
LOG=$WT/omip_nemo/cdx_arctic_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarially review a CLAIM (no code written yet) about the Arctic surface-salinity bias in our NEMO-ORCA1-matched OMIP runs. ALL numbers below are day-30 MPAS, Ob/Yenisei box (66-74N, 66-84E), 58 wet cells of 273, every mean taken with NEMO's tmask (three earlier versions of these numbers were wrong from averaging flood-filled LAND — treat unmasked means as a known trap here).

MEASUREMENTS: (1) Same IC file for both models (woce_salt_monthly_init_4p2.nc, January). IC masked surface = 27.81. NEMO day 26-30 surface = 26.68 (its profile 26.68 at 1 m -> 33.0 at 14 m: halocline intact, and it is FRESHER than its own IC surface, consistent with the January Dai-Trenberth input of ~1.5e-5 kg/m2/s in this box ~ 1.2 psu/month into a 1 m cell). Ours day 30 = 30.46 (8 runoff spread passes) / 30.12 (2 passes). (2) BRINE: ice volume (conc x thickness) vs the SI3 ice IC (Ice_initialization.nc, masked: conc 0.985, h 0.855 m, volume 0.843 m) changes by -0.011 m (8 passes) / +0.036 m (2 passes) over 30 days; at S_ocean-S_ice = 24 that is <= +0.07 psu into 12 m. (3) IC depth-weighted means: surface 27.81, top-12 m 29.17, top-25 m 31.16, top-47 m 31.24, top-100 m 31.26. Our day-30 surface (30.46) sits between the 12 m and 25 m values. (4) Our own top-47 m column mean is 32.74 vs the IC's 31.24 (+1.5 psu). (5) A one-variable A/B (8 vs 2 runoff spread passes, matched protocol, same 49358 cells) moved the box bias only 3.78 -> 3.44 psu and made global SSS rmse worse (0.524 -> 0.546), so runoff placement is refuted as the driver.

CLAIM: our Arctic SSS bias is (a) halocline EROSION — the top ~15-25 m of the initial column is homogenized, which by itself reproduces the +2.65 psu surface change — plus (b) a salt IMPORT of ~1.5 psu over 47 m that is not brine (lateral/vertical advection or the restoring).

ATTACK: (1) Is the inference in (3) sound, or could a different process produce a surface value that coincidentally equals the IC's 15-25 m mean (e.g. surface freshwater loss by evaporation/ice growth combined with weaker mixing)? Name the measurement that separates them. (2) Does (4) actually follow — our column mean is computed on the MPAS ladder over its own cells while the IC mean is on NEMO's 75-level ladder over NEMO cells; is that comparison valid, and if not what is? (3) The prior campaign record ([[omip_arctic_halocline_erosion]] in the session memory) EXONERATED vertical diffusion at the halocline depth by measuring K_v vs NEMO avt and finding the implied 90-day salinification was ~0.04 psu. Reconcile: can the column be mixed as the state says while diffusivity at the halocline is too small? Candidates to rank: convective adjustment / EVD, the TKE surface boundary condition under ice (eice attenuation), the ice-ocean stress driving shear, the freshwater/virtual-salt placement in the top cell, or numerical mixing in the tracer advection (superbee) on a 1 m top cell. (4) What single day-5 or day-10 diagnostic from the EXISTING snapshots (we have T, S, ice, tke, land_mask, per-level) would discriminate the top candidate from the others? (5) Anything in our SSS restoring that could import salt here despite --sss-ice-gate-nemo (box is 91-99% ice) — read the gate in ocean/forcing/sss_restoring.py and say whether it is truly off under ice. Rank CRITICAL/MAJOR/MINOR, cite file:line where code matters, label CONFIRMED vs PLAUSIBLE, under 500 words." > $LOG 2>&1
echo "codex rc=$?"
grep -q "Failed to refresh token\|refresh token was already used" $LOG && echo "CODEX AUTH FAILURE"
grep -vE "^\s*$" $LOG | tail -45
echo DONE
