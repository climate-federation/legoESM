#!/bin/bash
#SBATCH --job-name=cdx_uvw
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_uvw_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/_omip_merge
codex exec --sandbox read-only -C $WT \
"Review commit HEAD (git show HEAD): _SurfaceFluxAccumulator now also sums state.u, state.v and state.mass_flux_w (plain means on their own staggering, float64 canonical dtype) and drains u_mean/v_mean/mass_flux_w_mean; test added. Numbered, severity-tagged, file:line: (1) is mass_flux_w on this driver's state a velocity [m/s] or a thickness-weighted flux [m2/s] (read _save_snapshot around line 5934 and the state definition) and does a plain time mean of it correspond to NEMO's wo (a velocity, 5-day mean)? State what a reader must divide by. (2) memory: three more 3-D float64 sums on the GPU for the eORCA1 L75 run -- bound it in MB and say whether it risks the 64 GB host / GPU memory of the production card. (3) any silent shape or dtype hazard when u is (ny, nx+1, nlev). (4) test non-vacuous? One line verdict CLEAN or HOLD." 2>&1 | tail -120
echo "=== codex exit ${PIPESTATUS[0]} ==="







