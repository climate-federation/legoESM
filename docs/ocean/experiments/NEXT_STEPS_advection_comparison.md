# Advection comparison — tomorrow's pickup notes (2026-04-23 night)

Read together with `docs/ocean/experiments/advection_scheme_comparison.md`.

## What completed today

### 1. Issues closed on GitHub
- **#211** MPAS Eady blowup (TRiSK weightsOnEdge + oversize-edge filter)
- **#210** Advection scheme upgrade (DST-3/PPM-FCT/SOM delivered, verdicts documented)

### 2. Follow-up issues opened
- **#212** PPM-FCT: sign-split Zalesak limiter (replace face-min heuristic)
- **#213** Hi-res (10 km) SOM on Eady — no stable parameter set at strong forcing

### 3. MPAS weak-forcing (U=0.2) 600-day cross-grid comparison — DONE
Locations:
- `results/ocean/eady_uniform/mpas_channel/20km/upwind_mpas_20km_U02_Bh2.3e11_Cs0.0_600d/`
- `results/ocean/eady_uniform/mpas_channel/20km/tvd_mpas_20km_U02_Bh2.3e11_Cs0.0_600d/`
- `results/ocean/eady_uniform/cross_grid_compare_U02_600d/` — 5-run overlay
- `results/ocean/eady_uniform/mpas_channel/20km/advection_compare_U02_600d/` — MPAS-only overlay
- `results/ocean/eady_uniform/bci_development/*__bci_snapshots.png` — 5 per-run BCI-development figures (via new `scripts/plot/replot_eady_bci.py`)

Both runs PASS; `T_drift` at machine precision; `max_speed` 0.057–0.058 m/s (BCI does not amplify at this forcing).

### 4. Key scientific findings (documented in `advection_scheme_comparison.md`)
- **Scheme ranking** (Var(T) loss, lat-lon): som < tvd < upwind, monotonic at both forcings.
- **Cross-grid comparison is confounded by KPP** — on lat-lon KPP is active; on MPAS it is silently skipped (`ignoring unsupported schemes: vertical_mixing='kpp'`). Apples-to-apples would need either a KPP-off lat-lon rerun or KPP ported to MPAS.
- **Weak-forcing wall-eddy divergence**: at U=0.2, lat-lon (all 3 schemes) develops wall-trapped eddies along the northern boundary by day 240. MPAS does NOT — flow stays zonally uniform over 600 d. Separate from pure advection-scheme question; involves mesh/wall geometry and PV-scheme defaults.
- **MPAS IC velocity has 17% zonal noise off-jet** — see IC investigation below. Not a bug, a discretization artefact. Noise decays, no instability.

### 5. Minor bug fixes applied during this session
- `scripts/plot/plot_T_volumetric_census.py` — hardcoded `(0, 360)` for MPAS mesh reconstruction → now reads `(cfg.lon_west, cfg.lon_east)` from `EadyUniformConfig`.
- `scripts/plot/replot_eady_bci.py` (new script) — produces physical-aspect, domain-cropped snapshot figures with a `SST − zonal-mean` anomaly row. Fixes the stock `snapshots_SST.png` showing 0–360° canvas for a 10°-wide domain. Color range excludes t=0 IC perturbation so evolving structure is visible.

## What is IN PROGRESS at bedtime

**Two MPAS strong-forcing 200-day runs — both alive, healthy, at ~day 95/200 when I stopped monitoring.**

Commands used:
```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
  --only =eady_uniform --grid mpas_channel --resolution 20km --levels 20 \
  --days 200 --dt 300 --tracer-advection upwind \
  --B-h 2.3e11 --C-smag 0 --no-sponge \
  --tag upwind_mpas_20km_U08_Bh2.3e11_Cs0.0_200d

CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
  --only =eady_uniform --grid mpas_channel --resolution 20km --levels 20 \
  --days 200 --dt 300 --tracer-advection tvd \
  --B-h 2.3e11 --C-smag 0 --no-sponge \
  --tag tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_200d
```

Logs: `logs/advection_matrix/mpas_{upwind,tvd}_U08_200d.log`

**Expected output dirs (check tomorrow):**
- `results/ocean/eady_uniform/mpas_channel/20km/upwind_mpas_20km_U08_Bh2.3e11_Cs0.0_200d/`
- `results/ocean/eady_uniform/mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_200d/`

**Early indication**: MPAS develops BCI at strong forcing (max_spd 0.4 m/s at day 20 → 1.1 m/s at day 95 for upwind, 0.6 m/s for tvd). This is a clean BCI saturation trajectory — matches the expected eddy-active regime. So the "MPAS doesn't develop BCI" concern from the weak-forcing runs is narrowed to "MPAS doesn't develop *wall-trapped* eddies at weak forcing".

**Update at ~22:00:**

- **upwind BLEW UP at step 54100 (≈ day 188)**: max_spd 2.57 m/s in the preceding diag window, then NaN. This is the same "too-energetic-eddy-blowup-without-Smag" pattern the lat-lon `upwind_nosponge_200d` and `som_nosponge_200d` runs hit. At strong forcing with `C_smag=0` and no sponge, the eddies saturate above what `B_h=2.3e11` alone can damp. Expected behaviour; not a grid-specific bug.
- **tvd is at day 195/200, max_spd 1.92 m/s, still alive** — should finish any moment. Earlier checkpoints show max_spd growing through day 170 (1.79 m/s) consistent with saturation.

**Implications for tomorrow:**

1. **tvd strong-forcing comparison should work** — MPAS tvd clearly makes it to ~day 200 with saturated eddies; pair with lat-lon `tvd_nosponge_200d` for a clean cross-grid plot.
2. **upwind strong-forcing blew up pre-saturation** — matches lat-lon `upwind_nosponge_200d` (also FAIL per earlier table in the main doc). So in both grids, strong forcing + pure upwind + no Smag is unstable. The MPAS failure has the same character; not a new finding.
3. To get all-scheme strong-forcing MPAS, would need to either add `C_smag=0.2` (matches what was required for lat-lon SOM) or a short-ish sponge. Both are follow-ups, not critical for the comparison story we already have.

## Tomorrow's pickup checklist

### A) Finalise the strong-forcing comparison (once runs complete)
1. `cat results/ocean/eady_uniform/mpas_channel/20km/*_U08_*/results.txt` — confirm PASS and final numbers.
2. Regenerate BCI-development figures:
   ```bash
   .venv/bin/python scripts/plot/replot_eady_bci.py \
     results/ocean/eady_uniform/mpas_channel/20km/upwind_mpas_20km_U08_Bh2.3e11_Cs0.0_200d \
     results/ocean/eady_uniform/mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_200d \
     results/ocean/eady_uniform/latlon_channel/100x50/upwind_nosponge_200d \
     results/ocean/eady_uniform/latlon_channel/100x50/tvd_nosponge_200d \
     results/ocean/eady_uniform/latlon_channel/100x50/som_csmag02_nosponge_200d \
     --out results/ocean/eady_uniform/bci_development_U08
   ```
   Target days may need tweaking — strong forcing saturates earlier. Consider editing `TARGET_DAYS = [0, 30, 60, 100, 140, 200]` at top of `replot_eady_bci.py`.
3. Generate T-census overlay:
   ```bash
   .venv/bin/python scripts/plot/plot_T_volumetric_census.py \
     results/ocean/eady_uniform/mpas_channel/20km/upwind_mpas_20km_U08_Bh2.3e11_Cs0.0_200d \
     results/ocean/eady_uniform/mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_200d \
     --out results/ocean/eady_uniform/mpas_channel/20km/advection_compare_U08_200d
   ```
4. Append results to `docs/ocean/experiments/advection_scheme_comparison.md` under a new "Strong-forcing cross-grid (Run 6)" subsection.

### B) IC balance noise — fix or punt
Ocean-expert recommendation is to rewrite `_set_linear_shear_mpas` in `src/legoesm/ocean/experiments/eady_uniform.py` to use the discrete edge-normal gradient `(T[c2] − T[c1]) / dcEdge` instead of `cos(angleEdge) · envelope(lat_edge)`. Expected to drop IC zonal noise from 17% → <1% off-jet. ~15 LOC edit.

**Decide:** is cleaner IC worth a code change, or leave as documented and move on? If yes, new issue + small PR.

### C) Files changed, awaiting decision on commit
```
M docs/ocean/experiments/advection_scheme_comparison.md
M scripts/plot/plot_T_volumetric_census.py
?? scripts/plot/replot_eady_bci.py
?? docs/ocean/experiments/NEXT_STEPS_advection_comparison.md (this file)
?? results/ocean/eady_uniform/bci_development/ (5 PNGs)
?? results/ocean/eady_uniform/cross_grid_compare_U02_600d/ (3 PNGs)
?? results/ocean/eady_uniform/mpas_channel/20km/advection_compare_U02_600d/ (3 PNGs)
?? results/ocean/eady_uniform/mpas_channel/20km/{upwind,tvd}_mpas_20km_U02_Bh2.3e11_Cs0.0_600d/ (full MPAS 600d run dirs)
```
The user did not request a commit in this session. Two commit-worthy groupings:
- **"Cross-grid MPAS weak-forcing advection comparison + census/replot fixes"** — the doc update, census-script lon-range fix, replot script, weak-forcing result dirs.
- **"MPAS strong-forcing Eady runs"** (once they finish) — the U=0.8 result dirs.

### D) Open follow-up questions (beyond this session)

1. **Wall-eddy suppression on MPAS at weak forcing** — is it the staircase boundary, the enstrophy-conserving PV flux, or both? Candidate tests: switch MPAS to `pv_scheme="energy"` (but that re-opens the ζ-checkerboard risk — may need APVM); roughen the lat-lon wall by randomising the north/south wall cells to mimic staircase; check whether the lat-lon TVD/SOM wall eddies survive with a discrete-staircase wall.
2. **KPP on MPAS** — port or keep grid-exclusive? This is the main apples-to-apples blocker for cross-grid scheme comparisons.
3. **Lat-lon KPP-off control runs** — cheap path (3 × ~15 min) to validate the "KPP-is-the-Var(T)-loss-difference" hypothesis.
4. **PPM-FCT sign-split Zalesak** (#212) and **Hi-res SOM recipe** (#213) are documented but untouched.

## Quick-context facts for a cold pickup

- Branch: `dhruv/mpas-trisk-weights`.
- Both V100S GPUs used; each 600d MPAS run is ~14 min wall; each 200d U=0.8 MPAS run is ~projected 20–25 min.
- `EadyUniformConfig` domain: 10°×18°, 25°N centre, H=5500 m; default U_surface=0.8 (changed via `--U-surface 0.2` for weak forcing).
- Standard physics for this comparison: `B_h=2.3e11, C_smag=0, no sponge, 20 levels`. For SOM the lat-lon runs use `C_smag=0.2` (per Hill et al. / the doc recommendation); SOM not available on MPAS.
- KPP confound noted throughout — active on lat-lon, silently skipped on MPAS.
