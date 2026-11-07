#!/bin/bash
#SBATCH --job-name=cdx_kzeta
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_kzeta_%j.out
# CLAIM review BEFORE the fix (ACCURACY FIRST): the level-8 MPAS blowup.
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
LOG=$WT/../legoESM/omip_nemo/cdx_kzeta_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarially review a CLAIM about a numerical blowup in our MPAS (icosahedral Voronoi, TRiSK) ocean model. No fix written yet.

SETUP: the NEMO-match OMIP card runs stably on the level-7 mesh (163,842 cells, ~58 km, dt 150 s) but blows up from rest on the level-8 mesh (655,362 cells, ~29 km, dt 75 s): max|u| 0.09, 0.17, 0.24, 0.30, 0.74, 3.73, 19.6, 106, 600, 3750 m/s at steps 1-10 (per-step growth factor 5.1, 5.3, 5.4, 5.7, 6.2 -- super-exponential); first at the Black Sea ~800 m then the Alboran Sea at ~1700 m (edge 1553445, level 52). Halving dt (37.5 s) HALVES the per-step factor (2.5) => explicit operator with |lambda| dt ~ 5 at dt 75.

BISECT (30-step arms, one variable each, per-step max|u|): bottom-boundary advection, ice-shelf/tidal mixing, mixed-layer eddies, TKE->KPP, GM/Redi off, partial cells off, FLAT BOTTOM, WOA-only IC, analytic rest-state IC (no ice), split-explicit barotropic (+cosine filter), A_h=0 + Smagorinsky=0, TVD tracer advection, legacy linear drag, drag off: ALL blow up with the same growth (visc0 slightly slower). Mesh quality fine (dcEdge min 24.9 km), barotropic CFL 0.67 (same as level 7 at its dt), TRiSK tangential weights reconstruct a uniform flow to 1e-3.

PER-TERM TENDENCY on the step-6 state at the max edge (u=+3.73 m/s, level 52), from packages/ocean/legoesm/ocean/dynamics/ocean_pe_mpas.py with a new term_diagnostics return (du_dt_full = -grad_B + pv_flux + visc + vert_adv_u, line ~664): grad_B -2e-6, pv_flux -8e-7, vert_adv_u -7e-6, visc -3.115e-01 m/s^2. visc x 75 s = -23 m/s on +3.73 => -19.6 = the measured step-7 value. The visc term (line ~579-650) = A_h Laplacian + Smagorinsky + K_zeta_bih * biharmonic_vorticity_del4_3d(u). The NEMO-match recipe (packages/ocean/legoesm/ocean/fidelity/nemo_match_recipe.py:119) sets K_zeta_bih = 1.0e14 m^4/s, fixed, not scaled with resolution; it targets the zeta-checkerboard null mode of the energy-conserving PV flux and is applied EXPLICITLY.

CLAIM: (a) the level-8 blowup is the explicit biharmonic vorticity filter K_zeta_bih=1e14 crossing its stability limit: with the dual-mesh spacing dv ~ dx/sqrt(3) (14.4 km at level 8, 28.9 km at level 7) and the hexagonal Laplacian max eigenvalue ~6/dv^2, K dt (6/dv^2)^2 ~ 0.8 at level 7 (stable) and ~6 at level 8 (sign flip, |1-6| ~ 5 per step = measured); (b) A_h/Smag-off still blowing is consistent because K_zeta_bih was untouched in that arm; (c) the fix is to scale K_zeta_bih with dx^3 (1e14 x (29/58)^3 = 1.25e13 at level 8), or make the operator implicit/limited; confirmation arms (K_zeta_bih 1.25e13 and 0) are running.

ATTACK: (1) Read biharmonic_vorticity_del4_3d and its call: is it truly explicit forward-Euler in the momentum update, and what is the correct discrete stability bound on this dual mesh (vertex Laplacian eigenvalue on triangles/kites)? Does my 6/dv^2 estimate hold, and does the bound predict level 7 marginal (0.8) -- i.e. is level 7 ALSO close to the limit and quietly damping physical vorticity? (2) Is the fix 'scale K with dx^3' the faithful one (NEMO scales bilaplacian viscosity ~ dx^3; MPAS-Ocean scales del4 ~ dx^3), and does the recipe carry a dx_ref to tie 1e14 to? Name file:line of every place that would need the scaling (config default, recipe, tripole lane if it has the term, driver override). (3) Is there a CFL-style guard that should have caught this at build time (like smag_cfl_safety on the tripole)? Propose the minimal mechanical gate: assert K_zeta_bih * dt * lambda_max^2 < 2 at model construction using the mesh's actual min dvEdge, with a test that FAILS for the level-8 numbers. (4) Anything in the evidence that contradicts the claim? Rank CRITICAL/MAJOR/MINOR, cite file:line, label CONFIRMED vs PLAUSIBLE, under 450 words." > $LOG 2>&1
echo "codex rc=$?"
grep -vE "^\s*$" $LOG | tail -60
echo DONE
