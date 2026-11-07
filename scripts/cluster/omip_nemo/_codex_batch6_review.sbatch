#!/bin/bash
#SBATCH --job-name=cdx_b6
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_b6_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_b6_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarial DIFF review of the four commits 4eb1ac429..a2e584217 ('git log --oneline -4', 'git show <sha> -- packages tests scripts/validate' for each; the card commit is large: review only scripts/cluster/omip_nemo/_trp_base2_d30.sbatch, _trp_fwfix_d30.sbatch, _mpas_base2_d30.sbatch, _fesom_base2_d30.sbatch and one converted PYTHONPATH line).

(A) 4eb1ac429 virtual_closure_temperature_twin (freshwater.py; lat-lon block 8 virtual branch; ocean_pe_mpas tendency): dT/dt|top = -(P - E + restoring)/rho0 * T_top/h_top. Claim: our q_net (compute_omip2_surface_forcing, omip2_applicator.py ~1090-1120) = NEMO blk_oce_2 with -evap*cp*T_s + rain*cp*theta_air, and sss_restoring adds F*cp*sst, so NEMO's linear-free-surface trasbc counterpart (sbc_tsc(jp_tem) += emp*sst/rho0) is the correct twin; runoff/ice excluded. ATTACK: sign for each channel; is the restoring channel's heat REALLY added to q_net on the tripole card path (trace with_sss_restoring -> heat_flux -> q_net consumer) or only returned and dropped; does the twin double-apply on the MPAS path if physics also carries a surface heat term; interaction with normalize_freshwater (NEMO sbcfwb subtracts the correction's heat content) and with freshwater_salinity='local'; the real_freshwater branch must NOT get the twin (check the elif ordering); dtype/mask; the test's tolerance.
(B) e74d8b7cd MPAS vorticity-filter gate + recipe K_zeta_bih 0.0: bound lambda_max ~ 8/dv_min^2 evaluated at min dvEdge; raise >= 2, warn >= 1 inside check_barotropic_cfl (first checked step). ATTACK: is min dvEdge the right scale (Lloyd meshes have tiny edges near pentagons: false positives?); does step() (non-checked) bypass the gate; any test or recipe consumer still assuming 1e14; is the recipe change a hidden physics change for cards that did not pass --K-zeta-bih (every MPAS card: yes, and that is the user's decision, but list them).
(C) 8d4f5a851 fesom_level_rule --min-depth-m: does thers only clamp depth, or must the isolated-cell elimination start index and the floor move together (they do via 'thers'); off-by-one in 'argmin + 1'; does the 47-level self-check still run with the default.
(D) a2e584217 cards: PYTHONPATH '\$ROOT/src:' first — src/legoesm is a namespace package (no __init__) merging with packages/*/legoesm: any module shadowing risk (a file present in BOTH src/legoesm and packages/*/legoesm)? Run: comm on sorted module lists. The four new cards: is each truly ONE variable vs its stated control (base2 vs unified: viscosity shape + twin + kprofile + src-first + merged code = FOUR changes, it is a new BASELINE not an arm — state whether the header says so honestly); fwfix vs base2 = closure flags only (diff them); mpas_base2 single 30-d leg at 0.53 steps/s inside 11:55 h; fesom_base2 mesh path.
Rank CRITICAL/MAJOR/MINOR, CONFIRMED/PLAUSIBLE, cite file:line, under 600 words." > $LOG 2>&1
echo "codex rc=$?"
grep -vE "^\s*$" $LOG | tail -50
echo DONE
