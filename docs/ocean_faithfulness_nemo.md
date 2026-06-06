# Ocean faithfulness vs NEMO — all-grid correction tracker

**Goal:** correct ALL legoESM ocean grids (tripole/eORCA, latlon, cubed_sphere, mpas,
spectral) so each gives a faithful comparison to NEMO ORCA1 (CORE-II NYF). Review every
change with `/codex:adversarial-review`. **Shrunk at iter 18** (was 567 lines). Detailed
history: `OMIP_faithful.md`. Memories: [[omip-rk3-coldstart-solve]],
[[omip-pipeline-coordinate-bugs]], [[omip-smag-cap-stabilizer]], [[omip-faithful-project]].

**Branch:** `omip-faithful-nemo-comparison` (PRs #349, #352 merged to main). Completion
promise DONE only when grids genuinely match NEMO — tripole SST done; NOT all grids.

---

## GRID 3-5 SCOPING (iter-25, investigated — each a real multi-iter effort)

**Faithful CORE-II path = `run_omip_core2.py`** (applies the OMIP-2 bulk forcing NEMO uses,
via `omip2_applicator.apply_omip2_surface_forcing`). `run_omip.py` is a SEPARATE runner
using SST/SSS RESTORING (Haney) — NOT faithful, do not use for the NEMO match. The
applicator supports grid_type latlon/tripole/**cubed_sphere/mpas**, **NOT spectral**
(no grid-space u/v/T path — line ~317). `run_omip._create_setup(grid_type,...)` is the
shared builder for all grids; run_omip_core2 already reuses it for latlon/tripole.

### GRID-3 (cubed_sphere) EXECUTABLE ROADMAP (iter-26 — fully mapped)
Build is plumbing-ready BUT fidelity hinges on the dycore PGF fix (below).
- **Builder** `build_cubed_sphere` in run_omip_core2 (mirror build_latlon_bathy):
  `run_omip._create_setup("cubed_sphere", f"C{n}", nlev, H_max, ...)` → grid,z_coord,config
  (FC backend, A_h=5e5/K_h=5e6, fv3sw barotropic), model=OceanModel(grid,z,config,fc_config).
  Cube state: OceanState u,v,T,S (6,n,n,nlev) A-grid; eta,H_bathy,land_mask (6,n,n).
  NEMO bathy→cube: inline cKDTree-IDW on flattened cube `grid.lat/lon` (rad,(6,n,n))
  (regrid_curv_to_latlon is meshgrid/1D-only → need point-target variant). `rest_state_ocean`
  does NOT take bathy override → set via `state._replace(land_mask=state.land_mask.replace(
  data=...), H_bathy=...)`. WOA: `compute_woa_3d` (grid-agnostic) → T/S.
- **main()**: add `cubed_sphere` choice + branch, `app_grid_type="cubed_sphere"` (applicator
  already supports it, reads grid.lat/lon rad, forcing (6,n,n) at cell centres, step matches).
- **diag/snapshot cube-safe**: `_diag` unravels max|u| into 3 dims → cube u is 4D (6,n,n,nlev),
  guard by ndim (skip umax-loc for cube, keep max_speed+finite). `_grid_lat2d_deg` cube branch
  → grid.lat/lon (6,n,n) deg. Scorer is ALREADY grid-agnostic (flattens source) — no change.
- **BLOCKER = dycore**: cube `OceanModel` PGF-over-bathy instability. run_omip cube blows up
  ~4-5 d even WITHOUT bathy under gentle restoring; CORE-II bulk + realistic bathy is worse.
  FC-Gram backend + 5-50× A_h/K_h only DELAY it and over-smooth (kills fidelity). The real
  fix (per docs/ocean_experiments/cubed_sphere_pgf_stability.md): SMC03 density-Jacobian PGF
  + duogrid halo on T,S so the face-edge halo error stops feeding the PGF. Multi-iteration
  dycore effort (cf. tripole cold-start ~20 iters). NO freeze_floor/polar_filter on OceanModel
  (those are LatLonCGridOceanModel-only).

## Per-grid detail (older)
- **cubed_sphere**: PGF face-edge instability is MITIGATED (not cleanly fixed) by the
  **FC-Gram spectral baroclinic backend** (`ocean_pe_fc.py`, `build_fc_config`, default in
  run_omip.py) — BUT only together with **5-50× elevated A_h/K_h floors** (5e5/5e6). Uses
  `OceanModel` (cd-grid, NOT LatLonCGridOceanModel → no freeze_floor/polar_filter), and
  `_create_setup` builds it with NO realistic bathymetry. To get an EXCELLENT NEMO match
  needs: realistic bathy + reduce the 5-50× diffusion (reintroduces instability → needs the
  real SMC03-density-Jacobian-PGF + duogrid-halo-on-T,S fix per
  `docs/ocean_experiments/cubed_sphere_pgf_stability.md`) + WOA IC + a core2 builder. NOT a
  quick win.
- **mpas (#160 split-Coriolis)**: ocean uses relative-only PV (`zero_f`,
  `ocean_pe_mpas.py:466,473,476`) + separate Matsuno Coriolis; atmosphere uses full
  `(f+ζ)/h`. The split avoids #103 depth-mean-Coriolis double-counting; the documented clean
  fix needs a MOM6-style slow-forcing refactor (full PV in the TRiSK flux + remove the
  separate Coriolis without re-double-counting the barotropic mode). Plus a core2 mpas
  builder + bathy. Big.
- **spectral**: `SpectralOceanState` has no grid-space u/v/T/masks → the applicator cannot
  force it (line ~317). Needs a whole spectral forcing path (grid↔spectral transform of the
  CORE-II flux). Fundamental infra gap; largest effort.

## CURRENT STATE (iter 23 — latlon polar filter: 2nd grid started)

**Mask-aware Fourier polar filter for the lat-lon C-grid ocean (code, tested 7/7).**
latlon_bathy blows up ~day 0.25 (N-pole singularity: dx=R·dlon·cos(lat)→0, CFL). Reused
the shared `grids.polar_filter` (already used by the atmosphere C-grid) in
`LatLonCGridOceanModel._apply_polar_filter` (static `config.use_polar_filter` gate, end of
`step()`): truncates zonal modes above the per-lat CFL cap poleward of `cutoff_lat` (60°).
- **MASK-AWARE** (key): a naive zonal FFT smears continental zeros into ocean + couples
  basins across land — exactly why NEMO uses ORCA tripole. Land is filled with the per-lat
  ocean zonal mean before the FFT, restored after.
- **Conservation**: per-lat WET-CELL zonal mean restored post-filter (`filt += zmean-
  filt_mean`) → eta row-volume / zonal-mean u exact; tracer content exact only for zonally-
  uniform thickness, approximate under partial cells/z* (stability filter, not flux op).
  (Codex HIGH-1: original "exactly conserved" claim corrected.)
- Staggering handled: T,S cell (mask_c); v v-face (is_v_face mask, n_lat+1); u u-face
  periodic (filter `u[:,:-1]`, re-append col0); eta 2D. Shim grid carries lat_v/cos_lat_v
  built via the now-PUBLIC `compute_v_face_coords` (promoted from `_`-private per rules).
- Config: `use_polar_filter`, `polar_filter_{cutoff_lat_deg,max_wave_speed,safety_factor}`
  (off by default, bit-exact legacy). Runner: `--polar-filter[-cutoff-lat/-max-wave-speed/
  -safety]`. Tests: `tests/ocean/unit/test_polar_filter_ocean.py` 7/7 (conservation, land
  restore, high-k poleward damp, equatorward passband, u-wrap, shapes, step gating);
  freeze_floor regression 4/4. Codex review: 2 HIGH fixed, round-2 CLEAN. Commit acde1c6d.
- **LIMITATION (honest)**: even mask-aware, lat-lon can't be fully faithful in the
  land-locked Arctic (residual cross-pole basin coupling) — ORCA tripole stays THE faithful
  path. This is for lat-lon stability + a tropics/mid-lat/SH compare.
- **VALIDATED (8417577, iter-24): polar filter SOLVES the latlon cold-start.** 1° latlon
  WOA cold-start + `--polar-filter` runs STABLE past day 0.25 (prior blowup point): day-1..6
  max|u| bounded **1.6-2.0 m/s** (physical WBC), peak at **38.5°N (Gulf Stream), NOT the
  pole**; SST ~12.6 °C / SSS ~33.97 stable, finite. The N-pole singularity is gone. Config:
  `--grid latlon_bathy --latlon-res 180x360 --dt 300 --woa-init --partial-cell --pgf smc03
  --adaptive-implicit-vertadv --momentum-rk3 --min-levels 2 --C-smag-lap 3.0
  --smag-cfl-safety 0.125 --polar-filter --polar-filter-cutoff-lat 60`. Smoke = 18 days
  (snapshots day 5/10/15).
- **latlon vs NEMO (8417586, seasonal Jan `--nemo-month 1`): EXCELLENT at day-15.** raw
  RMSE **1.31 °C, corr 0.99**, all bands |bias|<1.1 (Arc −0.19, NHmid −0.15, trop −0.50,
  SHmid −1.02, Ant +0.01); +freeze-clamp 1.26. day-10 RMSE 1.36. Second grid matches NEMO.
  CAVEAT: day-15 is near the WOA IC (early); equilibration test (day-90, like tripole)
  needs the longer run — launched **8417589 (3mo latlon + freeze-floor)**. The Arctic
  cross-pole-coupling caveat may grow over equilibration; watch it at day-90.
- **EQUILIBRATION CONFIRMED (8417829, iter-25): latlon HOLDS EXCELLENT.** day 30/45/60/90 vs
  seasonal NEMO: RMSE 1.23/1.21/1.21/**1.12**, corr 0.99 throughout; day-90 all bands
  |bias|<0.65 (Arc −0.23 — caveat did NOT blow up; freeze-floor+filter hold it). RMSE
  CONVERGES (1.23→1.12), not degrades. raw==clamp (model-side freeze-floor active). **latlon
  is now a genuinely faithful 2nd grid — day-90 1.12 °C, marginally better than tripole 1.15.
  GRID 2 DONE.**

## CURRENT STATE (iter 22 — SEASONAL CONFOUND found)

**MAJOR METHOD FINDING — the "equilibration degradation" is largely a SEASONAL-PHASE
artifact, not model drift.** Trend scoring (8417518, no-freeze snapshots vs NEMO *annual*
mean) shows a hemispheric dipole that GROWS MONOTONICALLY FROM ~0 at day-0:
| day | RMSE | Arctic | NHmid | trop | SHmid | Ant |
|---|---|---|---|---|---|---|
| 15 | 1.87 | −1.79 | −2.56 | −0.73 | +0.67 | +0.82 |
| 45 | 2.64 | −3.74 | −3.72 | −0.43 | +1.89 | +1.70 |
| 90 | 2.92 | −4.90 | −3.68 | +0.26 | +2.30 | +1.46 |
legoESM runs from a WOA **annual** IC under perpetual NYF; a day-D snapshot is ~calendar
day-D (start Jan 1) → day-90 ≈ **end-March** (NH late winter = coldest, SH late summer =
warmest). It was scored against the NEMO **5-yr annual mean**. The growing NH-cold/SH-warm
dipole IS the seasonal cycle (NHmid −3.7 ≈ realistic Gulf-Stream-lat winter depression),
manufactured by instantaneous-vs-annual-mean scoring — NOT a faithfulness failure.

**FIX (iter-22, code): seasonally-matched scoring — CONFIRMED, dipole collapsed.** NEMO
RUN_REF has a monthly grid_T (`ORCA1_1m_20000101_20041231_grid_T.nc`, 60 rec). Added
`--nemo-month M` to `compare_omip_nemo.py`: averages records `(M-1)::12` = climatological
calendar-month mean (off by default, annual behavior bit-unchanged). Re-score 8417544
(day15,30→Jan; 45→Feb; 60,75,90→Mar) vs same-month NEMO:
| day-90 SST | RMSE | bias | corr | Arctic | NHmid | SHmid | Ant |
|---|---|---|---|---|---|---|---|
| annual (old, WRONG metric) | 2.92 | −0.30 | 0.97 | −4.90 | −3.68 | +2.30 | +1.46 |
| seasonal raw | 1.83 | −0.36 | 0.99 | −2.70 | **−0.33** | **−0.07** | +0.29 |
| seasonal + freeze-clamp | **1.15** | −0.06 | 0.99 | **−0.13** | −0.33 | −0.07 | +0.31 |
**The NH-cold/SH-warm dipole was 100% a seasonal-phase artifact** (instantaneous ~end-March
snapshot vs NEMO ANNUAL mean). NHmid −3.68→−0.33, SHmid +2.30→−0.07 under same-month scoring.
The only genuine residual = Arctic super-cooling (sea-ice), closed by freeze-floor
(−2.70→−0.13). **Tripole/eORCA025 day-90 = RMSE 1.15 °C, corr 0.99, all bands |bias|<0.5 =
EXCELLENT / genuinely faithful at equilibration.** Model-side confirmation: in-flight run
8417505 (every-step freeze-floor) — score vs `--nemo-month 3` when done; expect ~1.1-1.2.
ALL prior "POOR at equilibration" verdicts were the annual-vs-instant metric bug, NOT model.

**Codex adversarial review of `--nemo-month` (done, 2 HIGH fixed):** (1) no range
validation → added `1<=month<=12` guard; (2) positional `(M-1)::12` assumed Jan-first
without metadata check → now decodes TRUE calendar month per record from CF `units`+
`calendar` via cftime (`_nemo_record_months`), selects by real month, falls back to
positional only for whole-monthly-year files with a warning, else refuses. Re-score
8417551 confirms the hardened path reproduces RMSE 1.15 + prints decoded indices
(validates record-0 = January from metadata). MEDIUM (skipna mean) benign for NEMO's
fixed land mask.

## CURRENT STATE (iter 21 — ralph resume)

**In flight:** `8417505` freeze-floor 3mo ¼° (PENDING, est start 06-06 09:19, 48h walltime).
sbatch verified correct (PYTHONPATH=packages/*, winning config, `--freeze-floor`). Scores
day-90 vs NEMO when done; expect global SST RMSE < 2.45. Freeze-floor commit `7a81bc55`
already codex-reviewed (2 HIGH fixed) — not re-touched.

**Launched `8417518`** (CPU short): equilibration TREND scoring of the existing no-freeze
climatology snapshots (`legoesm_e025_rk3_3mo` day 15/30/45/60/75/90), RAW + `-1.8` freeze-
clamp, with band breakdown. Answers open item #1: is SST degradation monotonic? when/where
does SH warm bias emerge? predicts freeze-floor gain at each lead. No GPU/new sim. Score
→ results/omip_nemo/compare_trend_day*_{raw,clamp}/. Read on next iter.

## CURRENT STATE (iter 20 — ralph resume, post-merge)

**Merge:** `origin/main` merged into branch 2026-06-06. Main restructured `src/legoesm`
→ `packages/<pkg>/legoesm` (PEP420 namespace) + brought the **#356 tripole-MPI deadlock
fix** (fold active on all ranks + `fold_j=-1` sentinel, gated `_fold_is_local`) which
SUPERSEDED my branch's `_inactive_fold` fold approach. **sbatch now needs
`PYTHONPATH=$(ls -d packages/*/)`** (else jn2808's old `src/legoesm` shadows my code);
scorer → `scripts/validate/compare_omip_nemo.py`; this doc → `docs/`. See memory
[[omip-postmerge-packages-layout]].

**TRIPOLE SST: GOOD at short lead, DEGRADES at equilibration.** ¼° eORCA025 vs NEMO yr-5:
- day-4 (8132087): RMSE 1.49 °C, corr 0.991 → EXCELLENT
- day-12 (8134991): RMSE 1.76 °C, corr 0.987 → GOOD
- **day-90 (8417489, equilibrated climatology run 8135049): RMSE 2.92 °C, bias −0.30,
  corr 0.966 → POOR.** Global-mean bias IMPROVES (−0.6→−0.3) but pattern RMSE WORSENS and
  corr drops → regional errors grow as the model settles into its OWN climate. NOT yet
  "excellent" at equilibration — this is the open faithfulness gap.
- SSS corr ~0.89, RMSE ~1.18, bias +0.09 (runoff-gated, informational).

**Band breakdown (8417490) localises the 2.9 °C — hemispheric DIPOLE:**
| band | RMSE | bias | | band | RMSE | bias |
|---|---|---|---|---|---|---|
| Arctic >45N | 5.49 | **−4.90** | | tropics | 1.37 | +0.26 |
| NH-mid 23-45N | 4.56 | **−3.68** | | SH-mid 23-45S | 2.40 | +2.30 |
| | | | | Antarctic <45S | 1.72 | +1.46 |
NH too COLD (Arctic/midlat), SH too WARM. NH dominates global RMSE.

**FIX #1 — freezing-point floor (iter-20, DONE+tested, run launched).** Arctic −4.9
diagnosed as **missing sea-ice/freezing process**: legoESM ocean has no freezing, so
exposed Arctic water super-cools below seawater freezing (−1.8 °C) — unphysical; NEMO's
LIM ice caps SST. Scorer freeze-clamp `−1.8` (8417491) PROVED it: global RMSE **2.92→2.45
(poor→good)**, bias −0.30→0.00, Arctic 5.49→**2.86**. Implemented model-side as gated
`config.freeze_floor` (sea-ice thermodynamic surrogate, same `jnp.maximum(T,T_freeze)` clamp
as `simple_ocean.py`): `LatLonCGridOceanConfig.{freeze_floor,freeze_floor_temp_c}`,
`LatLonCGridOceanModel._apply_freeze_floor` in `step()` (static-bool gated), runner
`--freeze-floor`. 4 unit tests pass (8417498). Flooring EVERY step (vs only at scoring)
should beat the diagnostic — **run 8417505** (¼° 3mo + `--freeze-floor`, winning config;
codex-reviewed surface-only clamp, both step + scan paths). Committed 3a632fad. Score
day-90 vs NEMO when done; expect global SST RMSE < 2.45 (floor every step beats end-only).

**STILL OPEN after freeze-floor:** NH-midlat −3.68 (WBC cold bias, resolution/heat-transport
— NOT a freezing cell, unfixed by floor); SH warm +1.5..+2.3 (Southern Ocean too warm —
mixing / convection / AABW / WOA IC). Next diagnoses.

### THE SOLVE (winning config)
```
--mesh eORCA025_mesh_mask.nc --nlev 20 --dt 75 --woa-init --balanced-init --momentum-rk3
--partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --min-levels 2
--C-smag-lap 3.0 --smag-cfl-safety 0.125
```
**RK3 momentum integrator is the key** — it solved the corrected-WOA-IC cold-start that
forward-Euler+Matsuno could not, AND cured the velocity over-intensity (1 m/s vs 30 m/s).
Vindicates iter-9 "full stack together"; iter-10 "RK3 marginal" was pre-smag-cfl-cap.
`explicit_substep` baro DESTABILIZES (use implicit_cn). IC-smoothing = dead end
(`smooth_woa_ts` smooths T/S separately → static instability). EVD already active.

### Enabling fixes (all committed + merged to main)
- **smag-cfl viscosity ceiling** (`smag_cfl_safety`): per-cell `area·cos²(lat)·safety/dt`
  cap on Laplacian-Smagorinsky — the tuned ceiling that bounds the cold-start (NOT the
  rectangular CFL bound, which starves the WBC → blows up). `laplacian_smag_cfl_cap`.
- **Adaptive-implicit vertadv** (NEMO ln_zad_Aimp), **partial-cell smc03 PGF**, RK3.
- **Pipeline coordinate/unit bugs** (corrupted ALL prior comparison numbers): scorer
  double-`rad2deg` (corr 0.1→0.97); **WOA-init paired 2-D lat with 1-D lon on tripole**
  (IC mis-placed ~9°/155° Arctic); NEMO mask `|tos|>1e-6`; `_diag` umax_lon u-face index;
  finite flag +S,v; synthetic-forcing fallback; `_idx_t` +3h. (audit workflow w2gc8r1bh.)
- **omip2_applicator merge-corruption** (unclosed paren that was breaking main).

---

## Per-grid status
| grid | cold-start | comparison vs NEMO |
|---|---|---|
| **tripole/eORCA025 (¼°)** | **STABLE** (corrected IC + RK3 + stack) | **SST RMSE 1.15 °C corr 0.99 EXCELLENT at day-90 (seasonal scoring + freeze-floor, iter-22); SSS corr 0.90 (gated)** |
| tripole/eORCA1 (1°) | free run impossible (config exhausted) | superseded by ¼° |
| **latlon_bathy (1°)** | **STABLE** (mask-aware polar filter, iter-23/24) | **day-90 SST RMSE 1.12 °C, corr 0.99, all bands |bias|<0.65 = EXCELLENT (8417829); RMSE converges 1.23→1.12 over 3mo** |
| cubed_sphere | untested w/ CORE-II | **[high] face-edge PGF instab** masked by 5-50× diffusion → fix SMC03 + duogrid halo |
| mpas | untested w/ CORE-II | **[high] split-Coriolis (#160):** f zeroed in PV flux → MOM6 full-PV q=(f+ζ)/h |
| spectral | applicator gap TOTAL | SpectralOceanState has no grid-space u/v/T/masks → spectral forcing path |

---

## Open work (toward "all grids" + complete faithfulness)
1. **Score 8135049 climatology** (day-15/30/.../90) → equilibrated SST/SSS trend.
2. **SSS runoff ungate** — wire Dai-Trenberth: `apply_runoff_step` + the staged NEMO
   `runoff-icb_DaiTrenberth_Depoorter.nc` (eORCA1) NN-regridded to eORCA025; ungates
   SSS/MLD/AMOC. Needs curvilinear regrid + a re-run.
3. **Transports** (ACC@Drake ~130 Sv, AMOC@26N 15-20 Sv) — now meaningful (physical
   velocities); needs grid metrics in the scorer + NEMO grid_U/V ref + `_streamfunction.py`.
4. **latlon pole filter / Arctic mask** (geometric) for a 2nd-grid run.
5. **Coarse grids** (cubed_sphere PGF, mpas split-Coriolis, spectral forcing path) — each a
   real dycore fix (audit findings above).
6. **HW-KE fold halo** (deferred): `ocean_pe_latlon_cgrid.py` ~1015 fills j±1 by edge-
   replication — WRONG at the active tripole fold (should be `vector_sign_u·u[-1:,perm_T]`);
   prerequisite to `ke_gradient_scheme="hollingsworth"` on tripole. Centered KE is fold-safe.

## Pipeline (built + proven)
NEMO ORCA1 5-yr ref + identical CORE-II 6-hourly `nyf.zarr` + WOA18 staged.
`run_omip_core2.py` (runner, tripole + latlon, observable per-diag CSV + mid-run snapshots),
`build_core2_nyf_zarr.py`, `compare_omip_nemo.py` (cKDTree-IDW regrid + SST/SSS score, fixed).

## Session iter-19 (infra: MPI + convection + scan; all tested, codex-reviewed)
- **#353 tripolar MPI halo** (`parallel/latlon_mpi.py`): latitude-band MPI now folds the
  ORCA north by perm+sign (not the atmospheric 180° roll) → multi-CPU tripole ocean runs.
  `_fold_tripolar_north` bit-exact vs serial `_fold_row`; ocean scatter/gather/geometry/zcoord
  band-slicing. 31 serial tests pass; 7 codex findings fixed. (np=2/4 tests need mpi4jax — absent here.)
- **Grid-agnostic convection** (`ocean/physics/column.py`): `convective_adjustment_K` (Oceananigans
  EVD, identical per-column on any grid shape) + `extract_cell_center_velocity`. Convection on the
  faithful tripole path is now opt-in: `run_omip_core2.py --convection enhanced_diffusion`
  (default `none` = validated config untouched). Flows through the existing grid-agnostic
  `compute_vertical_K_profiles` → implicit solve. **Relevant to faithfulness: deep-water formation /
  MLD / SSS** (convective mixing was OFF on tripole, `physics=None`). 7 grid-agnostic + 3 regression pass.
- **#354 lax.scan forcing** (`coupler/omip2_applicator.py`): `compute_omip2_surface_forcing_jax`
  (on-device NN gather, no per-step host pull; bit-eq to host) + `build_omip2_scan_block_fn`;
  `run_omip_core2.py --scan-block N` (opt-in; default 0 = bit-identical Python loop). 5 tests pass.
