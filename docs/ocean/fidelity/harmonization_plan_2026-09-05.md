# Three-grid harmonization plan (user directive 2026-09-05)

Directive: same resolution for FESOM; one lateral-viscosity strategy across the three grids
(pick the best of: Laplacian 1e4 + Smagorinsky [tripole], Laplacian 1e4-1e5 + Smagorinsky
[MPAS], flow-aware biharmonic [FESOM]); one vertical closure (our TKE vs CVMix-TKE); FESOM on
the 75 NEMO levels; no regression vs the 2026-09-04 virtual-salt unified arms; delete old
test runs; codex + GLM review.

## Reference for "no regression"
The 2026-09-04 unified virtual-salt arms (trp_unified180, mpas_unified_180d) and fesom_b5_d90,
scored at GATEWAY days 30/60/90 (results/omip_nemo/threegrid_unified_table_2026-09-04.md and
baseline_trajectory_2026-09-04.md). Every change below runs as ONE-VARIABLE arms against them at
matched days. The real-freshwater closure stays OFF (its dilution defect is reviewed and open).

## P1 Lateral viscosity — decide by measurement, then unify
NEMO ORCA1 itself runs a LAPLACIAN with a prescribed file (eddy_viscosity_3D.nc: 20000 m2/s
mid-latitude -> 1000 at the equator), i.e. none of the three current strategies is the oracle's.
Ladder on the tripole unified card, 30 days, one variable each, scored vs GATEWAY rec 5 (SST,
SSS, nino3, EUC from the new u_east snapshots):
  (a) ctl: Laplacian 1e4 + Smagorinsky schedule (= yesterday);
  (b) NEMO file profile (--A-h-profile-file, A_h 2e4 mid-lat, Smagorinsky off);
  (c) biharmonic only (B_h scaled ~dx^3 to FESOM's value).
"Best" = lowest SST/SSS rmse with the undercurrent closest to NEMO; (b) is the faithful one by
construction. The winner is then ported: MPAS needs the latitudinal profile plumbing (only an
equatorial boost exists), FESOM needs a Laplacian option in fesom_jax momentum (only opt_visc=7
biharmonic is ported). Both ports get the same one-variable arm before adoption.

## P2 Vertical closure — decide by the single-column twin
Our TKE is the NEMO zdftke port (tendency-matched to ORCA1); CVMix-TKE is FESOM's own Gaspar
implementation. Reuse scripts/validate/ocean_fidelity/frozen_column_tke_twin.py: same column
state + forcing into both closures, compare K_v and MLD against NEMO's avt at the same column.
If ours is closer, FESOM gets our TKE through the existing closure bridge (the iwm splice
pattern); otherwise the tripole/MPAS closure question is reopened.

## P3 FESOM on 75 NEMO levels
fesom_jax loads per-node/per-element level counts (nlvls/elvls) that FESOM's Fortran mesh setup
derived from node depth + the 47-level zbar. No FESOM build here, so: infer the level rule from
the shipped nlvls.out/elvls.out (self-check must reproduce all 126,858 node counts on the 47-level
ladder before it is trusted), regenerate aux3d/nlvls/elvls with NEMO's 76 interfaces, run
prepare_mesh.py, then a FESOM b5-style 30-day arm on the new ladder (dt may need to drop from
1800 s for the 1 m top cell — measured, not assumed). One variable vs fesom_b5_d90 at d30.

## P4 FESOM resolution
Only the CORE2 mesh exists locally (nominal 1 deg, 25 km equatorial refinement, ~126 k nodes)
vs eORCA1 (1 deg, 1/3 deg equatorial). Comparable nominal resolution; an identical-resolution
FESOM mesh needs an external mesh generator (not available here) — flagged, not attempted.

## P5 Space
Deleted (this session's scratch, 22 GB): l8bisect_*, mpas_fwnemo_l8_diag1d, mpas_fwnemo_l8_180d.
Candidates for user approval: 48 GB of >14-day arms (nemolev_ico7_*, nemolev_mpas*, legoesm_e025_*)
and 43 GB of late-August daily-snapshot probes (nemolev_trp_{kmean,bilin_dailysnap,
std1deg_dailysnap,windsonly,modfrc,iceab20}_d30). Kept: everything referenced by a table/memory.

## GPU budget
Three 48 GB glab1 GPUs hold the real-freshwater pair + twin (verdict in hand: closure defective
for SSS, twin divergence 1e-3 C). The level-8 chain (K_zeta_bih=0) is queued behind them.
P1 needs 3 short arms; P3 one; P2 is CPU. Proposal: cancel the real-FW trio to free the GPUs.

## Review dispositions (2026-09-05 02:00)
GLM: day-30 cannot rank viscosity (EUC/WBC need 100+ d) -> ladder runs to 90 d; (b) as written
changes 3 things (profile, base 1e4->2e4, Smag off) -> hold the ramp/Smag/dt, vary the background
only; P2 metric = MLD evolution through a forced sequence, run P2 AFTER P3 on the 76-interface
column; P3: a 1 m top cell at dt 1800 is a vertical-advective-CFL risk (w~1e-3 -> 1.8 m/step) ->
CFL census from baseline w before any arm; reproducing node counts validates the counter, not the
depth rule -> also check reconstructed bottom depths.
codex: --A-h-profile-file is a surface zonal-median, depth-invariant, latitude-only proxy applied
as a multiplier after the operator (run_omip_core2.py:942, ocean_pe_latlon_cgrid.py:3247) -> NOT
NEMO's 3-D coefficient in flux form; faithful option needs a prescribed 2-D/3-D field inside the
Laplacian flux. Ladder: profile-only atop the unchanged schedule -> Smag off -> 2e4 endpoint;
drop the biharmonic arm. P2 harness frozen_column_tke_twin.py has only control|nemo modes and no
MLD -> needs a CVMix adapter + matched prognostic column trajectories. P3: verify elvls/nlvls/
aux3d linkage and the minimum-level policy, not just counts.
