#!/bin/bash
#SBATCH --job-name=cdx_supercool
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_supercool_%j.out
# Claim review BEFORE code (CLAUDE.md ACCURACY FIRST). Codex CLI, read-only.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
LOG=$WT/omip_nemo/cdx_supercool_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarial review of a CLAIM (no code written yet). Under sea ice our ocean top cell supercools: day-105 Arctic cells with ice concentration >0.95 are 93% below -1.8 C, mean deficit 1.4 K, min SST -7.7 C, on BOTH grids (MPAS and tripole) to 0.1 K; deficit grows ~0.011 K/day in winter (0.35 K d15, 0.64 d30, 1.05 d60, 1.39 d105), recovers in summer; flat across concentration bins 0.85-1.0 (~1.0 K), 0.37 K at 0.5-0.85, zero in open water. NEMO ORCA1 never goes below freezing under ice; compare_omip_nemo.py --freeze-clamp-C -1.8 hides it. CLAIM: cause is the lead-freeze latent-heat DOUBLE COUNT: packages/ice/legoesm/ice/sea_ice.py ocean_heat_extraction (lines ~571 and ~2288) adds +rho_i*L_f*dV_lead (ocean debited), while packages/coupler/legoesm/coupler/ocean_forcing.py blend_ice_ocean_forcing (~line 240) ALSO applies the open-water atmospheric non-solar flux (1-A)*q_nonsolar to the ocean, and q_ice=-ocean_heat_extraction; no cap at the freezing point; the SST<=Tf gate (sea_ice.py ~1937) then latches open. NEMO SI3 icesbc.F90:357-405 caps basal extraction at zqfr and forms lead ice only from the remainder zqld-zqfr (ocean cooled to freezing, never below). Unmerged commit 37b2aa454 (branch omip-arctic-vfs-tke-fix) implemented a 'credit' sign (ocean +L_f dV_lead) with driver selecting it under --prognostic-sea-ice. Proposed fix: cherry-pick that (and 8fa9f52c4, db3ca4e2e), select 'credit' in run_omip_core2 under --prognostic-sea-ice, optionally add NEMO's zqfr cap. TASKS: (1) read the CURRENT sea_ice.py ocean_heat_extraction compositions (both thermo paths) and ocean_forcing.py blend and confirm or refute the double count; (2) find ANY other term that can cool a fully ice-covered (conc>=0.99) top cell in winter at 5-10 W/m2 — check the ice model's basal growth, snow-ice flooding, sublimation, brine, the coupler's f_open/f_water weights, the SW transmittance, the ocean-side under-ice fluxes in scripts/run/run_omip_core2.py host loop (grep blend_ice_ocean_forcing); (3) check whether the cherry-pick applies cleanly onto HEAD (git show 37b2aa454; compare with current sea_ice.py) and what else changed since; (4) say whether dV_lead from the ICE-SKIN flux vs the ocean receiving the OPEN-WATER flux leaves a residual under the credit convention and whether NEMO's zqld-zqfr form should be the target. Rank findings CRITICAL/MAJOR/MINOR, cite file:line, label each CONFIRMED (read in code) vs PLAUSIBLE. Under 600 words." > $LOG 2>&1
rc=$?
echo "codex rc=$rc"
grep -q "Failed to refresh token\|refresh token was already used" $LOG && echo "CODEX AUTH FAILURE"
cat $LOG
echo DONE
