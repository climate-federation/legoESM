# Ocean faithfulness vs NEMO — all-grid correction tracker

**Goal:** correct ALL legoESM ocean grids (tripole/eORCA, latlon, cubed_sphere, mpas,
spectral) so each gives a faithful comparison to NEMO ORCA1 (CORE-II NYF). Review every
change with `/codex:adversarial-review`. **Shrunk at iter 18** (was 567 lines). Detailed
history: `OMIP_faithful.md`. Memories: [[omip-rk3-coldstart-solve]],
[[omip-pipeline-coordinate-bugs]], [[omip-smag-cap-stabilizer]], [[omip-faithful-project]].

**Branch:** `omip-faithful-nemo-comparison` (PRs #349, #352 merged to main). Completion
promise DONE only when grids genuinely match NEMO — tripole SST done; NOT all grids.

---

## CURRENT STATE (iter 18)

**FAITHFUL TRIPOLE SST MATCH ACHIEVED.** legoESM ¼° eORCA025 reproduces NEMO ORCA1
(CORE-II) SST with **corr ~0.99**. Trustworthy run 8131128 (the solve config below)
completed 12 d stable, physical max|u| 0.67→0.47 m/s. vs NEMO yr-2:
- day-4 (8132087): SST bias −0.61, **RMSE 1.49 °C, corr 0.991 → EXCELLENT**
- day-12 (8134991): SST bias −0.60, **RMSE 1.76 °C, corr 0.987 → GOOD**
- SSS corr ~0.90, RMSE ~1.1, bias +0.06 (excellent pattern, runoff-gated).

Pattern corr stays ~0.99; RMSE drifts up as the model settles into its OWN equilibrium
(two cores under one forcing diverge in detail). **Definitive corrected+RK3 3-month
climatology run = 8135049** (glab1, snap/15 d) → equilibrated SST/SSS + spun-up transports.

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
