#!/bin/bash
#SBATCH --job-name=cdx_dil
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_dil_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_dil_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarial DIFF review of commit HEAD (defcbe43f, 'real_freshwater closure now dilutes the top cell'): run 'git show HEAD -- packages tests' and read packages/ocean/legoesm/ocean/freshwater.py::real_freshwater_dilution_tendencies, its two call sites (ocean_model_latlon_cgrid.py block '8-real', ocean_model_mpas.py before the S>=0 clamp) and tests/ocean/unit/test_real_freshwater_dilution.py.

Claimed physics: NEMO vvl/qco puts the surface water flux F into the free surface; the tracer step keeps zero surface tracer flux; the z-star stretch spreads the volume uniformly, so the resident water must be transported DOWN through each interface by W(k+1/2) = (F/rho) H_below/H (F/rho at the surface, 0 at the bottom); surface interface carries S=0, T=T_1; flux form => column salt exactly conserved, top cell -S_1 F/(rho h_1), layers below unchanged; T unchanged everywhere for uniform T. Verified numerically in a one-column step (top dS -1.76009e-6 vs exact -1.76009e-6; pre-fix -5.6e-8).

ATTACK, cite file:line, rank CRITICAL/MAJOR/MINOR, CONFIRMED/PLAUSIBLE, under 500 words: (1) Is the derivation right when F varies horizontally and the eta channel is NORMALISED (global mean removed) -- the tracer channel uses the same normalised NOW rate on lat-lon (_F_fw_rate_now) and F_slow_eta on MPAS: consistent, or does the MPAS F_slow_eta differ from what eta actually received (barotropic solver, cosine filter, fix_eta_drift projection)? (2) The lat-lon site uses h_k_new and the post-advection S/T; MPAS applies BEFORE the S>=0 clamp: order/time-level issues, AB2 outer integrator path (_apply_implicit_vmix False), the NEMO leapfrog 'before' state, restart? (3) Ice: the ice SALT flux stays on surface_forcing.salt_flux; F includes ice_fw (melt water). NEMO's net freezing tendency is (S - S_ice) F_ice: with this fix + the existing brine channel, is the real closure now (S - S_ice) or (S + S_ice) or S-only at ice points? Trace where fw.ice_fw and the brine/salt flux are built for the tripole prognostic ice (run_omip_core2 / ice coupling) and say which. (4) Temperature: NEMO's bulk adds the heat content of emp (qns -= emp*sst*rcp) so net T is unchanged; our surface T = T_1 convention reproduces that ONLY if our heat flux does NOT already carry an emp*sst term -- check surface_energy / bulk_flux for such a term (double count?). (5) Runoff depth spreading (runoff_depth_spread_map, NEMO ln_rnf_depth_ini) exists on the virtual path: under real_freshwater the runoff still enters at the surface interface only -- is that a regression vs NEMO's rnf depth (trasbc rnf spread) and how should the helper take a per-channel depth? (6) Any JIT/pytree/dtype hazard (f32 S with f64 F), MPI/band SPMD slicing of F, partial cells (h=0 below seafloor) -- the helper zeroes dry layers; is H_below right with partial bottom cells? (7) Tests: what would still pass if the helper's sign were flipped or the surface T convention wrong?" > $LOG 2>&1
echo "codex rc=$?"
grep -vE "^\s*$" $LOG | tail -50
echo DONE
