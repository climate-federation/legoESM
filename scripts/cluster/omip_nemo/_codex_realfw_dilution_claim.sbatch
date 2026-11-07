#!/bin/bash
#SBATCH --job-name=cdx_fwdil
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_fwdil_%j.out
# CLAIM review BEFORE any fix: the real_freshwater closure does not dilute.
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_fwdil_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarially review a CLAIM about the surface freshwater closure of our z* ocean cores (MPAS Voronoi and lat-lon/tripole C-grid). No fix written yet.

MEASURED (two 180-day OMIP arms per grid, identical except --freshwater-closure real_freshwater + --sss-restore-channel water_flux + --sss-restore-normalization live_s vs the virtual_salt_flux default; snapshots at day 15 and 30, mean salinity anomaly real minus virtual per level, MPAS and tripole agree):
- Amazon mouth (0-5N, 305-315E), day 15: surface +2.17 PSU (real saltier), +1.62 at 5 m, +0.39 at 14 m, +0.06 at 31 m, 0.0000 below 60 m.
- Weddell (70-62S, summer melt): +0.43 at the surface, 0 below 30 m.
- Kara/Laptev (72-78N, winter freezing): -0.53 at the surface, 0 below 14 m.
- Day-30 vs NEMO GATEWAY: Antarctic SSS bias +0.06 -> +0.25, Arctic +0.04 -> -0.14, global SSS rmse 0.43 -> 0.53 (tripole) and 0.54 -> 0.61 (MPAS); tropics improve because the virtual arm's Amazon plume runs away negative (a clamp fires at S=0).

CODE READ: docs/dev-notes/ocean_real_freshwater_design.md says real_freshwater 'keeps the eta/volume channel (already correct, already NEMO-like), skips only the virtual-salt block, retains surface_forcing.salt_flux'. The freshwater volume enters eta through F_slow_eta in the barotropic step (ocean_model_mpas.py step). The vertical velocity used by the tracer step is diagnosed from the HORIZONTAL flux divergence only (ocean_pe_mpas.py ~302-316 -> legoesm.ocean.vertical.diagnose_w_from_flux_div, 'z-star sigma correction so that w = 0 at both surface and bottom'); the freshwater flux is NOT in that continuity, and there is no surface tracer flux term for salinity under real_freshwater. Same pattern on the lat-lon core (ocean_pe_latlon_cgrid.py ~302-305).

CLAIM: (a) under real_freshwater the surface freshwater changes eta and stretches the column uniformly but never reaches the salinity equation: the tracer update conserves each level's h*S under the uniform stretch (a ~0.3 m freshwater input spread over a ~3 km column = 0.004 PSU, invisible), so there is NO surface dilution -- evaporation/precip/runoff/ice melt/ice freeze all leave SSS unchanged except through the ice brine channel (-S_ice*m), which is why freezing regions go FRESHER and melting regions SALTIER than NEMO; (b) NEMO's non-linear free surface gets the top-cell-only dilution because its vertical velocity includes the surface volume flux (sshwzv: w(1) = emp/rho0 from the continuity with the e3t tendency) while the surface advective tracer flux is zero (traadv: zwz(:,:,1)=0 for vvl) and trasbc carries no emp*sss term for vvl -- the salt is advected downward relative to the moving coordinate while the top cell keeps its salt and loses water; (c) the fix is to include the net freshwater volume flux F/rho in the continuity that diagnoses w (w_surface = F/rho before the z* sigma redistribution, zero surface tracer flux), on BOTH cores, which reproduces NEMO's dilution exactly and removes the need for any virtual salt term; (d) the virtual_salt_flux default was double-counting only by the small uniform-stretch part, which is why it scored better vs NEMO at day 30 than the 'faithful' real closure.

ATTACK: (1) Read diagnose_w_from_flux_div (packages/ocean/legoesm/ocean/vertical.py ~1374-1460) and the MPAS/lat-lon tracer steps: is the salinity really untouched by the eta freshwater, or is there a dilution path I missed (e.g. h_old*S/h_new with h_new including the freshwater, applied per level)? Quote lines. (2) Check NEMO 5 sshwzv.F90 and traadv/trasbc for the vvl case: is (b) an accurate statement of where NEMO's surface dilution comes from? If the vertical flux at the surface is w(1)*T(1) in vvl, redo the top-cell budget and say what NEMO actually does. (3) Is (c) the minimal correct fix, and does it interact with: the z* sigma redistribution (partial cells), the SSS-restoring water_flux channel, ice_fw, the normalize_freshwater global-mean correction, tracer conservation tests (which tests would need to change and which must NOT change)? (4) What single offline test proves the fix: propose a one-column/one-cell test (uniform S column, prescribed P-E for one step) with the exact expected SSS change S*F*dt/(rho*h_top) and zero change below. Rank CRITICAL/MAJOR/MINOR, cite file:line, label CONFIRMED vs PLAUSIBLE, under 500 words." > $LOG 2>&1
echo "codex rc=$?"
grep -vE "^\s*$" $LOG | tail -70
echo DONE
