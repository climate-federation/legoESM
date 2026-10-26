#!/bin/bash
#SBATCH --job-name=cdx_north
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_north_%j.out
# CLAIM review (ACCURACY FIRST): the "northern cold bias regression" the user saw
# in the 2026-09-04 three-grid table is an arm-selection error, not a model change.
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
LOG=$WT/omip_nemo/cdx_north_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarially review a CLAIM about our NEMO-ORCA1-matched OMIP runs (three grids: tripole eORCA1, MPAS ico7, FESOM2 CORE2). The user reported: 'FESOM is much better at northern latitudes than tripole and MPAS, both of which suffer from a northern (cold SST) bias; this was not present yesterday so this is a major regression.'

MEASURED (from snapshot npz files, wet cells only via land_mask, surface level):
- The three-grid table the user saw (docs/ocean/fidelity/three_grid_status_2026-09-04.md) used arms trp_visc180 and mpas_fullharm_180d. At day 90 those have SST minimum -10.43 C (tripole, 5555 cells < -1.9 C, 2998 < -3 C) and -8.14 C (MPAS, 3890 / 2145). Arctic band (N of 45N) SST bias vs NEMO on the common mask: tripole -0.43, MPAS -0.59, FESOM -0.04.
- The arms the user saw YESTERDAY (trp_unified180, mpas_unified_180d) have SST min -1.81 / -1.80 at days 30 and 90, ZERO cells below -1.9 C. Native comparator Arctic SST bias at d30: tripole -0.07, MPAS -0.17, FESOM -0.27 (FESOM b5). NH-midlat d30: trp -0.34, MPAS -0.39, FESOM -0.45.
- Both old arms ran commit 2ada0c850, both unified arms f5addc5ed. Command-line diff: the unified arms add --ice-lead-freeze-source nemo_qlead (plus, on MPAS only, --isf --iwm --bbl-adv). Commit 116071f74 'fix(ice): NEMO SI3 lead heat budget — the ocean cools at most to freezing, the remainder forms ice' introduced that option; run_omip_core2.py:5497 now DEFAULTS it to nemo_qlead and every live card selects it.
- scripts/validate/compare_omip_nemo.py:363-375 floors legoESM SST at a freeze clamp before scoring and prints a WARNING with the count; the three-way tool scripts/validate/ocean_fidelity/compare_three_way_nemo.py does not clamp. That is why the native score of the old tripole arm read +0.18 (Arctic) while the three-way read -0.43 for the SAME snapshot (lego means 1.65 vs 1.63; NEMO means 1.47 vs 2.05 on the two masks).

CLAIM: (a) there is NO model regression — the table simply scored pre-fix arms whose Arctic ocean is supercooled to -10 C because open-water cooling under ice was not converted to ice formation before commit 116071f74; (b) the unified arms already carry the fix and show no supercooling; (c) the native comparator's clamp hid the defect in earlier scores; (d) the correct action is to re-score the three-grid table on the unified arms (job submitted), NOT to change model code.

ATTACK: (1) Is the -10 C supercooling fully explained by the missing lead-heat budget, or could 116071f74 have masked a second heat sink (e.g. an ice-ocean heat flux sign, or longwave under ice) that still runs in the unified arms and shows up as the residual Arctic -0.07/-0.17 and NH-midlat -0.34/-0.39 biases? Read packages/ocean/legoesm/ocean/physics/surface_forcing/ and ice/sea_ice.py lead-freeze path (grep lead_freeze_source) and say with file:line whether the nemo_qlead branch bounds the top cell at T_freeze by construction or by a clamp that hides heat. (2) The native comparator clamp: is reporting-with-warning enough, or should a supercooled count > 0 fail the score (exit non-zero / verdict FAIL)? Cite line. (3) Could the NEMO-mean difference between masks (1.47 vs 2.05) indicate the three-way common mask drops NEMO's coldest cells asymmetrically, i.e. an instrument artifact that would ALSO bias the unified-arm re-score? Name the cheapest check. (4) FESOM NH-midlat at d30 (-0.45) is WORSE than tripole/MPAS (-0.34/-0.39) in the native comparator, contradicting the user's reading — is the user more likely reacting to the three-way maps (which had the supercooled arms) than to any real FESOM advantage? Rank CRITICAL/MAJOR/MINOR, cite file:line, label CONFIRMED vs PLAUSIBLE, under 450 words." > $LOG 2>&1
echo "codex rc=$?"
grep -vE "^\s*$" $LOG | tail -60
echo DONE
