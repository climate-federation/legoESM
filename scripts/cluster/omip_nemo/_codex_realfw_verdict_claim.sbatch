#!/bin/bash
#SBATCH --job-name=cdx_fwverd
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_fwverd_%j.out
# CLAIM review: what the real-freshwater d90 result says about NEMO faithfulness.
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_fwverd_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarially review a CONCLUSION, not code. Context: z-star OMIP ocean cores (tripole eORCA1 C-grid, MPAS Voronoi) vs NEMO ORCA1 (non-linear free surface, qco/vvl: freshwater enters as volume through the top cell, no virtual salt flux). Two closures in our driver: virtual_salt_flux (default, salt flux -S*F into the top cell, volume also enters eta) and real_freshwater (volume into eta only; codex+GLM already CONFIRMED 2026-09-04 that our tracer w is diagnosed from HORIZONTAL flux divergence only, so the surface freshwater never dilutes salinity except via the ice brine channel — docs/ocean/fidelity/harmonization_plan_2026-09-05.md and results/omip_nemo/realfw_vs_virtual_d90.md).

MEASURED (day 90 vs NEMO GATEWAY 5-day means, both grids, one-variable arms): SST rmse and Nino-3 bias identical between closures (0.45/0.44, +0.59/+0.58 tripole; 0.47/0.45, +0.35/+0.35 MPAS); global SSS rmse WORSE under real_freshwater (0.58 -> 0.63 tripole, 0.68 -> 0.78 MPAS); Arctic SSS bias flips +0.26 -> -0.10 (tripole), +0.18 -> -0.24 (MPAS); Antarctic SSS +0.16..+0.21 salty. Same signs at day 30 (freezing regions fresh, melting regions salty). At day 30 the virtual arm's Amazon plume ran away toward S=0 (a clamp fired) while the real arm did not.

CONCLUSION under review: (a) this result says NOTHING about which closure is NEMO-faithful — the real arm is an INCOMPLETE port (no surface dilution), so its worse SSS measures the missing dilution, not the closure; (b) the virtual salt flux -S*F is exactly the LINEARISED top-cell dilution NEMO's volume closure produces, so it reproduces NEMO's SSS to first order except (i) the small uniform z-star stretch term (~0.004 PSU) and (ii) the non-linear regime S -> 0 at river mouths where the linearisation runs away (the Amazon clamp) — that is the only place NEMO's closure and virtual salt differ materially; (c) therefore keep virtual salt as the reference now, and the faithful fix is to put the surface volume flux F/rho into the continuity that diagnoses w (w_surface = F/rho before the z-star redistribution, zero surface tracer flux) on both cores, verified by a one-column exact test dS = S*F*dt/(rho*h_top); expected gain after the fix: Amazon/river-mouth SSS and the ice freeze/melt regions, not global SSS; (d) the sea-ice brine channel behaviour (freezing regions FRESH under real_freshwater) means the ice freshwater/brine bookkeeping is also entangled with the closure and must be re-checked after the fix, not before.

ATTACK: is (b) right for NEMO 5 qco (does trasbc really carry no emp*sss term and does sshwzv put emp in w(1))? Is 0.004 PSU the right size for the stretch term, and is the Arctic flip explained by the brine channel alone? Is there any reading of the data in which virtual salt is MORE faithful than a completed real closure? Rank CRITICAL/MAJOR/MINOR, label CONFIRMED/PLAUSIBLE, under 350 words." > $LOG 2>&1
echo "codex rc=$?"
grep -vE "^\s*$" $LOG | tail -40
echo DONE
