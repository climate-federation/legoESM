#!/bin/bash
#SBATCH --job-name=cdx_lens
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_lens_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/_omip_merge
codex exec --sandbox read-only -C $WT \
"Adversarial review of a CLAIM (no code yet). Repo: legoESM ocean vs NEMO 5.0.1 ORCA1 (source at /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE; our TKE: packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py; solar penetration: grep -rn rgb packages/ocean/legoesm/ocean/physics; driver scripts/run/run_omip_core2.py). MEASURED on 5-day WINDOW MEANS (new --state-accumulate; NEMO files are window means): nino3 SST bias +0.10 C at day 5 -> +0.64 C at day 30; daily-mean SST minus T(9.85 m): ours +0.93, NEMO +0.39 at day 30; enhanced vertical diffusion (NEMO-exact) moves it +0.003 with the lever engaged. Previously (on snapshots) column heat matched NEMO. CLAIM: the remaining bias is a near-surface (top 10 m) mixing/solar-deposition defect on the daily mean, not the afternoon diurnal layer, and the candidate list (mixing-length selector nn_mxl 3 vs ORCA1 2 giving l_eps 1.7x at 1 m; RGB solar fraction discretisation in the 1-m top levels; TKE surface BC; nn_etau) must be re-tested on window means because every prior refutation was read on 00 UTC snapshots. Numbered, severity-tagged: (1) read NEMO trasbc.F90 / traqsr.F90 and our solar-penetration code: is the solar heating of level 1 computed from the interface fractions (qsr*(1-frac(w1)))? Do we deposit the non-penetrative fraction (1-rn_abs? in RGB the full qsr penetrates with 3 bands) at the interface or in the cell, and is the RGB depth attenuation evaluated at gdepw (interfaces) as NEMO does? A mismatch of one half-level in 1-m cells is a 0.5 m error in deposition depth. (2) In zdftke.F90 lines 680-720 confirm what nn_mxl=2 vs 3 does to l_k (the mixing length for avm/avt) not only l_eps, and state the SIGN of the SST effect of running nn_mxl=3 where ORCA1 runs 2. (3) Name the ONE offline measurement on the saved 5-day-mean T profiles (levels 1-15) that separates too-shallow solar deposition from too-weak mixing (shape of the excess vs depth). Be concrete: file:line." 2>&1 | tail -120
echo "=== codex exit ${PIPESTATUS[0]} ==="







