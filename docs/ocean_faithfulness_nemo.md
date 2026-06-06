# Ocean faithfulness vs NEMO — all-grid correction tracker

**Goal:** correct ALL legoESM ocean grids (tripole/eORCA, latlon, cubed_sphere, mpas,
spectral) so each gives a faithful comparison to NEMO ORCA1 (CORE-II NYF). Review every
change with `/codex:adversarial-review`. **Shrunk at iter 18** (was 567 lines). Detailed
history: `OMIP_faithful.md`. Memories: [[omip-rk3-coldstart-solve]],
[[omip-pipeline-coordinate-bugs]], [[omip-smag-cap-stabilizer]], [[omip-faithful-project]].

**Branch:** `omip-faithful-nemo-comparison` (PRs #349, #352 merged to main). Completion
promise DONE only when grids genuinely match NEMO — tripole SST done; NOT all grids.

---

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
| **tripole/eORCA025 (¼°)** | **STABLE** (corrected IC + RK3 + stack) | **SST corr ~0.99 (excellent/good); SSS corr 0.90 (gated)** |
| tripole/eORCA1 (1°) | free run impossible (config exhausted) | superseded by ¼° |
| latlon_bathy (1°) | **blows up day 0.25** — N-pole singularity (GEOMETRIC; RK3 doesn't rescue, 8131279) | needs pole filter / Arctic mask |
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
