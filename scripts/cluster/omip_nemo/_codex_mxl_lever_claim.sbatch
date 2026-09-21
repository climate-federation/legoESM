#!/bin/bash
#SBATCH --job-name=cdx_mxl
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_mxl_%j.out
# CLAIM review BEFORE any new physics-path code (ACCURACY FIRST): is the
# mixed-layer-depth lever dead, and is the MPAS trio genuinely unresolvable?
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_mxl_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "ONE REVIEW. Adversarially review two CLAIMS from an OMIP ocean model scored against a NEMO ORCA1 oracle at day 5, equatorial Pacific nino3 (5S-5N, 150W-90W). All numbers below come from ONE scoring job on ONE 42254-cell intersection (absolute path /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/score_grids_d5_mxl4_d5_arms), the same tool and cell set that produced the controls.

MEASURED, TRIPOLE ARM (TKE mixing-length choice 3 -> 4, one variable, everything else identical):
- nino3 SST bias +0.5641 (control) -> +0.5661 (arm). Moved +0.002 C.
- nino3 MLD (dsigma=0.01) 14.70 m -> 15.12 m. Deepened +0.42 m. NEMO 17.95 m, so the standing gap is 3.25 m and the dose was 13% of it.

MEASURED, MPAS ARM (harmonised lateral mixing + surface TKE boundary condition moved to interior_pinned):
- hybrid +0.4565; harmonised-lateral control +0.5194; pre-harmonisation control +0.4676.
- |hybrid - OLD| = 0.011, |hybrid - NEW| = 0.063. The campaign noise floor (day-5 three-grid spread) is 0.060.

CLAIM 1: the mixed-layer-depth hypothesis cannot carry the +0.564 C equatorial warm bias, because a 0.42 m deepening bought +0.002 C; assuming local linearity, closing the whole 3.25 m gap buys order 0.015 C, i.e. 3% of the error.
CLAIM 2: the MPAS trichotomy is instrument-limited and NO verdict may be taken, because the distance separating the two competing readings (0.052 C) is below the 0.060 C noise floor that defines closeness.

ATTACK, and rank CRITICAL/MAJOR/MINOR, labelling each finding CONFIRMED vs PLAUSIBLE, under 400 words:
(1) Is differencing day-5 snapshot BIASES the right estimator for a lever whose mechanism is a tendency (entrainment flux), or does the claim need d(bias)/dt across two days and therefore prove less than stated?
(2) Entrainment heat flux scales with the temperature jump at the mixing base. If both 15 m and 18 m sit above the sharp upper thermocline, a 13% depth dose can be a near-zero flux dose -- does that rescue the hypothesis, and if so does the same convexity also undercut attributing NEMO's cooler surface to its deeper mixed layer?
(3) The arm deepened the mixed layer and WARMED the surface by 0.002 C. Is that sign physically possible for real entrainment, or is it the signature of a threshold diagnostic sliding in weak stratification?
(4) Name the single cheapest check on snapshots that ALREADY EXIST (each carries T, S, u, v, eta, tke, K_M_diag, K_H_diag, land_mask, z_center_ref) that separates 'weak lever' from 'wrong diagnostic'. Say what result would confirm each.
(5) Is CLAIM 2's refusal to pronounce correct, or is it over-cautious given three arms and a defined threshold?" > $LOG 2>&1
echo "codex rc=$?"
grep -vE "^\s*$" $LOG | tail -70
echo DONE
