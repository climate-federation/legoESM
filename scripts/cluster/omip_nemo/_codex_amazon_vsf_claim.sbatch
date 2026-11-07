#!/bin/bash
#SBATCH --job-name=cdx_amazon
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_amazon_%j.out
# CLAIM review BEFORE any config change (CLAUDE.md ACCURACY FIRST).
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
LOG=$WT/omip_nemo/cdx_amazon_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarially review a CLAIM about our NEMO-ORCA1-matched OMIP runs. No code written yet.

MEASURED (MPAS ico7, day-60 snapshots, wet cells only via land_mask): 16 wet cells have surface salinity EXACTLY 0.0000 in the newer arm (results/omip_nemo/mpas_unified_180d) and 0 such cells in the older arm at the same day (mpas_fullharm_180d). All 16 sit within 1.5 deg of the equator at 310-314 E = the Amazon mouth. A weighted localization of the day-60 tropical SSS rmse growth (0.575 -> 0.803, band BIAS unchanged at -0.05) puts 100.9% of the increase in the 300-330 E sector and 94.9% in the top 25 cells, all the same plume. Time series of the count of exact zeros: newer arm 0,0,6,16 at days 15/30/45/60; OLDER arm 0,0,0,0,0,6,6,9,8,0,0,0 at days 15..180 — so the older arm reaches the same state by day 90, i.e. this is PRE-EXISTING and the newer config only accelerates it. Profiles show S=0 in the top 2-3 levels rising to ~20-35 by level 5.

CODE READ: ocean_model_mpas.py:1109 does S_final = jnp.maximum(S_corrected, 0.0) with a comment saying the virtual_salt_flux closure uses a constant S_ref so it can overshoot negative in shallow cells with large freshwater input. run_omip_core2.py --freshwater-closure defaults to virtual_salt_flux, whose own help text says it is 'a spurious salt source NEMO does not have under variable volume (measured at +10.109 psu.m of excess Arctic salt over 60 d)'; --freshwater-salinity defaults to 's_ref' (fixed 35) while its help says 'local' is NEMO's tra_sbc convention (sfx = emp*sss). Neither production card sets either flag. NEMO's own runoff-depth spreading IS on in both arms (--runoff-depth-nemo-ini, dep_max 150 m, rnf_max 0.05, matching namelist_cfg of the oracle run).

CLAIM: the zeros are a CLAMP firing, i.e. the model computed NEGATIVE salinity and silently created salt; the mechanism is that our virtual-salt closure removes salt at a CONSTANT S_ref=35 regardless of local salinity, so in a plume already at ~5 PSU it over-extracts, whereas NEMO under z-star dilutes with the LOCAL salinity and cannot go negative.

ATTACK: (1) Is 'the clamp fired' actually established by S==0.0 exactly, or could exact zeros arise another way (initialisation, fill, a masked write, a different clamp)? Name the cheapest decisive check. (2) Read the tendency path in packages/ocean/legoesm/ocean/freshwater.py (runoff_spread_virtual_salt_tendency_3d, resolve_runoff_spread_arg) and packages/ocean/legoesm/ocean/dynamics/ocean_pe_mpas.py around line 1021-1060: does the S_ref-vs-local choice actually reach the runoff channel, or only the P-E/ice channel? If runoff always uses S_ref regardless of --freshwater-salinity, say so with file:line — that changes the proposed fix. (3) Does the clamp violate salt conservation, and is there any compensating global normalization (normalize=True path) that would hide it in the global mean? (4) Rank which of these actually owns the Amazon collapse: constant S_ref, the VSF-on-top-of-z-star double count, runoff regridded onto too few cells, or the top layer being 1 m thick. Name the ONE offline measurement from existing snapshots that discriminates the top two. (5) The newer arm additionally turned on bbl-adv, ISF, IWM and a NEMO lead-freeze budget; could any of those plausibly accelerate an Amazon-mouth freshening, or is the acceleration more likely unrelated drift? Rank CRITICAL/MAJOR/MINOR, cite file:line, label CONFIRMED vs PLAUSIBLE, under 450 words." > $LOG 2>&1
echo "codex rc=$?"
grep -q "Failed to refresh token\|refresh token was already used" $LOG && echo "CODEX AUTH FAILURE"
grep -vE "^\s*$" $LOG | tail -50
echo DONE
