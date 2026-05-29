# Ocean faithfulness vs NEMO — all-grid correction tracker

**Goal:** correct ALL legoESM ocean grids (tripole/eORCA1, latlon, cubed_sphere,
mpas, spectral) so each gives a faithful comparison to NEMO ORCA1 (CORE-II NYF).
Review every code change with `/codex:adversarial-review`. Shrink this file every
10 iters. (Detailed history: `OMIP_faithful.md`.)

**Branch:** `omip-faithful-nemo-comparison`. Completion promise DONE only when grids
genuinely match NEMO — far off; no false DONE.

## State (carried over)
- NEMO ORCA1 5-yr reference COMPLETE (T/S/SSH/MLD/U/V/ice). Identical CORE-II
  6-hourly forcing (`nyf.zarr`), eORCA1 mesh, WOA18 all staged.
- Pipeline built + proven: `scripts/run_omip_core2.py` (runner, tripole + latlon_bathy),
  `build_core2_nyf_zarr.py`, `compare_omip_nemo.py` (regrid + SST/SSS score).
- Applicator fixes (39 tests pass): tripole branch, wind-stress sign (−tau), 6-hourly
  flux, true periodic NN, in-step forcing (`compute_omip2_surface_forcing`).

## THE BLOCKER (root cause, rigorously diagnosed)
legoESM ocean blows up from realistic (WOA) stratification — NOT topo/IC/convection/
dissipation/dt (ruled out via flat-bottom, static-stability, ~18 experiments).
Mechanism: cold-start geostrophic adjustment from rest (u=0) overshoots to ~5-10 m/s
→ nonlinear advective feedback (u·∇u) → blowup, amplified at the **equator (f→0)**.
- **FIX #1 DONE (kept):** WOA-IC flood-fill — NEMO mask had ~13% ocean cells WOA lacks
  data for (S=0 → 40-PSU spurious gradients → 12 m/s step-1 PGF). Now filled from nearest
  valid column (step-1 |u| 12→0.8, mean SSS 29.7→34.1).
- **In progress:** spin-up Rayleigh velocity drag (damp the cold-start overshoot during a
  nudged spin-up, then release) — test 8100129 pending. If insufficient → thermal-wind-
  balanced init.

## Per-grid status
| grid | runs stable (realistic IC)? | comparison |
|---|---|---|
| tripole/eORCA1 | NO (cold-start blowup; flood-fill + drag in progress) | pending |
| latlon_bathy | NO (same; regridded NEMO bathy) | pending |
| cubed_sphere | untested w/ CORE-II | — (applicator supports) |
| mpas | untested w/ CORE-II | — (applicator supports) |
| spectral | applicator unsupported | TODO |

## Next
1. Stabilize realistic-stratification run (drag spin-up, else balanced init).
2. Once a grid runs stable + free → compare to NEMO (SST/SSS, then ACC/AMOC/MOC/MLD).
3. Extend to all grids; spectral applicator support.
4. codex-adversarial-review each change.

## Iteration log
- **iter 1:** new loop. Drag-spinup test 8100129 queued (GPU busy). Created this tracker.
  codex-adversarial-review of dycore changes = needs-attention, 2 valid findings, both FIXED:
  (1) [high] flood-fill copied donor columns without reapplying the RECEIVER bathymetry deep-fill
  → below-seafloor T/S bias; now reapplies the deep-fill (z_cen > H_bathy → deep-ocean fill).
  (2) [med] NN forcing-sampler cache key omitted dst_lon.sum() → same-shape grids could collide;
  key now signs full src+dst lat/lon checksums. Drag test 8100129 (PD) will run the fixed code.
