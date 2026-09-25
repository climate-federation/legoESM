#!/bin/bash
#SBATCH --job-name=cdx_rivergate
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_rivergate_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
LOG=$WT/omip_nemo/cdx_rivergate_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarial review of the LAST commit (git show HEAD) in scripts/run/run_omip_core2.py + tests/unit/test_run_omip_core2_fesom_cli.py + the two fesom cards: the FESOM lane now honours --river-mouth-restoring-gate (compute_sss_restoring_flux(river_runoff=_R if args.river_mouth_restoring_gate else None) in run_fesom_forced_loop, mirroring the host loop's _R_gate), the flag is allowlisted for fesom, and both fesom cards pass it. Motivation (measured): FESOM day-90 SSS rmse 0.85 vs tripole 0.56 / MPAS 0.67, the excess sitting at river mouths (Po, Kara/Laptev coast, Amazon) where FESOM is 10-20 psu SALTIER than NEMO — FESOM spreads runoff over 8 node-adjacency passes (MPAS also 8, tripole 2) and, until now, kept SSS restoring active at the mouths. Attack: (1) is _R (the month's runoff map, nod2D) defined and on the same node layout/units the host passes (kg/m2/s) at the fesom call site, and does compute_sss_restoring_flux gate on a threshold that assumes host units? (2) does the FESOM SSS-restore heat term (_sss_out['heat_flux']) also get gated consistently? (3) would the 8-pass spread (vs tripole 2) still leave the mouth cells saltier than NEMO, i.e. is the gate sufficient or is the spread the dominant term — cite the spread implementation (load_runoff_monthly / build_runoff_map spread_passes semantics) and NEMO's own runoff placement (sbcrnf: mouth cells, rn_rfact, ln_rnf_depth) to say what a faithful fesom placement would be; (4) the test: does test_river_mouth_gate_accepted_on_fesom prove wiring (it only checks the parser) — what minimal test would prove the gate reaches compute_sss_restoring_flux? Rank CRITICAL/MAJOR/MINOR, file:line, CONFIRMED vs PLAUSIBLE, under 400 words." > $LOG 2>&1
echo "codex rc=$?"
grep -q "Failed to refresh token\|refresh token was already used" $LOG && echo "CODEX AUTH FAILURE"
grep -vE "^\s*$" $LOG | tail -40
echo DONE
