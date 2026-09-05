#!/bin/bash
#SBATCH --job-name=cdx_dil2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_dil2_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_dil2_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarial DIFF review, ROUND 2, of commit HEAD (cad1c1e92) on top of defcbe43f: 'git show HEAD -- packages tests'. Round 1 findings applied: (a) interface donor is now the upwind cell by the sign of W; (b) runoff enters over NEMO's h_rnf through a per-level entry profile, W(k+1/2) = sum_{j<=k} entry_j - F H_above(k+1/2)/H; (c) entering water's tracer temperature per channel via real_freshwater_entry: rain/evap/restoring at 0 degC (heat content already in q_net: compute_omip2_surface_forcing = NEMO blk_oce_2 terms -evap*cp*T_s + rain*cp*theta_air + snow terms; sss_restoring adds F*cp*sst like sbcssr), runoff at SST over h_rnf (NEMO rnf_tsc = rnf*sst), ice melt and the normalisation residual at the local temperature.

ATTACK, cite file:line, CRITICAL/MAJOR/MINOR, CONFIRMED/PLAUSIBLE, under 450 words: (1) Verify (a): read the jnp.where donor selection and check the surface interface (S=0 both signs, T carried via entry_heat only) and the bottom (W=0). (2) Verify (b) algebra: for uniform S with entry profile e_k, does dS_k + stretch(-S F/H) equal -S e_k/h_k for every k, and is column salt exact? Is H_above_if the interface UNDER cell k (cumsum inclusive) -- an off-by-one here would put the stretch one level off. (3) Verify (c) against NEMO 5: with vvl, does trasbc apply NO emp*sst/emp*sss terms, and does sbcblk's qns really contain -evap*rcp*sst and +rain*rcp*theta_air? If yes, is 'rain at 0 degC in the tracer' the consistent counterpart? For runoff: NEMO rnf_tsc(jp_tem) = rnf*sst spread over h_rnf under vvl -- is 'entry at SST' right, and is the heat spread over h_rnf in NEMO or applied at the surface? For ice melt water: what does SI3 do with the heat content of the ice-ocean mass flux (fmmflx*rcp*sst?) and does our ice coupling add such a term (grep coupler/ocean_forcing, ice/sea_ice heat flux to ocean)? (4) The VIRTUAL closure (production default): q_net carries the rain/evap heat-content terms but the virtual closure adds no tracer dilution and NO emp*sst compensation (NEMO ln_linssh adds sbc_tsc(jp_tem) += r1_rho0*emp*sst): is that a spurious surface heat flux of order -(E-P)*cp*T (~2 W/m2 cooling where E>P)? CONFIRM or refute from trasbc.F90 / sbcblk.F90 semantics and our ocean_model_latlon_cgrid.py block 8. (5) MPAS: real_freshwater_entry receives F_slow_eta (normalised) and T_corrected; any shape/dtype/pytree hazard on the (nCells, nlev) layout, and is mask the cell mask? (6) tests/ocean/unit/test_real_freshwater_dilution.py: any remaining assertion that would pass with a wrong sign, a wrong donor, or the entry heat at the wrong level?" > $LOG 2>&1
echo "codex rc=$?"
grep -vE "^\s*$" $LOG | tail -50
echo DONE
