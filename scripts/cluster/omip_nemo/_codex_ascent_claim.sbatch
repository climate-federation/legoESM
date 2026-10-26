#!/bin/bash
#SBATCH --job-name=cdx_ascent
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_ascent_%j.out
# CLAIM review before any port/arm (ACCURACY FIRST): the shared cold-tongue transient.
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_ascent_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT "Adversarially review a CLAIM about a shared day-30 equatorial Pacific warm transient in three OMIP ocean cores (tripole eORCA1 C-grid, MPAS Voronoi, FESOM2 node mesh) vs NEMO ORCA1 from the same rest+WOA cold start and forcing.

MEASURED (day 30, NEMO GATEWAY 5-day records 5:6; results/omip_nemo/threegrid_unified_table_2026-09-04.md):
- Nino-3 bias: tripole +1.30, MPAS +1.08, FESOM +1.39.
- Undercurrent core / NEMO at 200/220/240E: tripole 0.58/0.34/0.77, MPAS 0.23/0.22/0.49, FESOM 1.14/1.00/1.18. Z20 matched on all three (gap -3..-5 m, tilt 102-105%).
- Slab tendency days 15->30 at 220-240E |lat|<=2 (recs 2->5): NEMO SST -1.13, T(0-50m) -1.56, T(50-150m) -0.51, Z20 -9.3 m; tripole +0.68/-0.62/+0.94/+1.0 m; MPAS +0.35/-1.09/+0.61/-3.9 m; FESOM +0.59/-0.55/+0.98/+4.6 m. At 240-280E all of ours cool 0-50 m as much as or more than NEMO.
- Earlier (08-18/08-23, older card): 240E vertical velocity ours -12e-6 m/s vs NEMO +19e-6 at d30; arm with ORCA1's eddy_viscosity_3D.nc latitudinal shape (1000 eq -> 20000 midlat) restored 240E ascent and Ekman v(lat) and halved the 220-240E tendency (+0.38 vs +0.74; NEMO -1.67); mixing-length nn_mxl 3 vs 2 A/B: nil; our TKE closure fed NEMO's own column reproduced NEMO's K.

CLAIM: (a) the shared transient is a missing equatorial ASCENT confined to 220-240E (heat pumped into 50-150 m instead of lifted), not an undercurrent, closure-ratio, or surface-flux effect, because FESOM has NEMO's undercurrent and the same transient, and the adjacent bins cool like NEMO; (b) the operator is the lateral momentum friction structure off the equator (Ekman divergence carrier), so the ORCA1 latitudinal A_h shape is the lever on all three grids; (c) a running one-variable arm (tripole, shape only, eq 500 / midlat 1e4, Smagorinsky ramp unchanged) is pre-registered to CONFIRM if dT(50-150m) at 220-240E <= 0 and dSST <= +0.2.

ATTACK: (1) Does the slab table actually discriminate ascent from horizontal advection/export at 220-240E? Name a cheaper offline check on the existing snapshots (T, S, u_east, v_north, eta, mass_flux_w on tripole/MPAS) that separates the two, e.g. the meridional heat convergence at 2N/2S or w at the 50 m interface in the bin. (2) MPAS cools 0-50 m by -1.09 (near NEMO's -1.56) with the weakest undercurrent -- is that consistent with (a), or does it point to resolution-dependent something else? (3) Could the 50-150 m warming be the surface-forced downward mixing of a too-warm 0-50 m (closure) rather than downwelling -- what in the numbers rules that out? (4) Is (b) the right operator class: what else sets equatorial ascent in the first month from rest (wind stress curl at 2-5N, the SEC/NECC structure, the tropical cell) and which of these is NOT lateral friction? Rank CRITICAL/MAJOR/MINOR, label CONFIRMED vs PLAUSIBLE, under 450 words." > $LOG 2>&1
echo "codex rc=$?"
grep -vE "^\s*$" $LOG | tail -60
echo DONE
