# Phase 2b — CLM-faithful multi-layer snow thermal (cold-bias + snow-tower fix)

**Goal.** Fix the boreal/high-latitude **cold bias** (`T_soil` → ~221 K) and the
**snow tower** (unbounded SWE) with real, conserving snow physics — not a
resistance approximation. The two-leaf canopy is currently *snow-blind*: the
surface energy flux is applied directly to the thin (0.025 m) top soil layer with
vegetation albedo and no snow insulation. CLM (Oleson 2013 §6) solves snow+soil as
one column with prognostic snow layers; we wire the existing, validated
`land/snow_column.py` (multi-layer, equal-mass remap, enthalpy phase change,
conservative drainage — 8 conservation tests pass) between the surface and the
top soil layer.

## Conservation contract (must hold after every stage)

- **Water**: `Δ(pack SWE) = snowfall·dt − drainage`; `drainage` → soil
  infiltration / runoff (nothing created or lost).
- **Energy**: pack `ΔH = (Q_top − G_bottom)·dt + fresh_enth − drainage_heat`; the
  soil receives `G_bottom` at its top BC and `drainage_heat` with the drained
  water, so the coupled snow+soil energy budget closes. `L_f` for melt/refreeze is
  handled *inside* the column (enthalpy method), so it must NOT be double-booked at
  the old `multilayer_land.py:674-679` melt-energy seam.
- Snow leaving via the cap is **ice at pack temperature** → no fusion term.

## Coupling design (operator split, first-order in the interface temperature)

`step_snow_column(state, precip_snow, T_air, Q_top, G_bottom, dt)` takes `G_bottom`
as a **prescribed input** (the pack base loses it to the soil) and returns
`(new_snow, drainage, drainage_heat)`. So the coupler computes `G_bottom` from
LAGGED (start-of-step) temperatures and hands the SAME value to both the column base
and the soil top BC — that is what makes the split conservative (identical flux at
both ends of the snow↔soil interface).

Per step, per column, with `has_pack = total_column_SWE > eps`:
1. **Q_top** = the canopy/atmosphere net flux into the surface
   (`surface_out.G_soil = sw_net + lw_net − shflx − lhflx`). Faithful (CLM): the
   canopy's ground/skin boundary `T_surface` is the **pack-top T** for `has_pack`
   columns (lagged from `state.snow_column`), else the soil top — and the two-leaf
   Picard callback `_soil_thermal_cb(G)` routes through the column for `has_pack`
   columns (step the pack with `Q_top=G`, return the tentative pack-top T) so the
   canopy converges against the surface it actually exchanges with.
2. **G_bottom** = snow-base↔soil-top conductive flux
   `= g_iface·(T_snow_base − T_soil_top)`, `g_iface` = harmonic mean of the
   snow-base half-conductance (`k_snow/(½·dz_base)`) and the soil-top half-conductance
   (`k_soil/(½·dz_top)`), lagged T. Sign convention: **positive downward** (into the
   soil), matching `solve_soil_thermal`'s `+G_surface` top BC and `step_snow_column`'s
   `rhs[...,-1] += −G_bottom` (base loses it). Confirm both ends carry this sign.
3. `step_snow_column(...)` → `(new_snow, drainage, drainage_heat)`.
4. Soil top BC flux = `jnp.where(has_pack, G_bottom, Q_top)`; `drainage_heat` folded
   into the top-layer soil energy; `drainage/rho_w` added to the Richards `flux_top`
   (infiltration) with any excess → runoff. Melt/refreeze fusion is handled INSIDE
   the column (enthalpy method) — do NOT also subtract `melt_energy` here (would
   double-book `L_f`); the single-scheme `update_snow`/`melt_energy` block is
   bypassed for `has_pack` columns.
5. **Empty-pack limit** (`~has_pack`): `G_bottom = Q_top`, column left inert (its
   thermal solve is skipped/masked — a zero-mass pack is ill-conditioned), so a
   snow-free cell is bit-identical to the "single" scheme. Fresh snowfall onto an
   empty pack accumulates into the column (handled by `step_snow_column`'s
   accumulation step) and flips `has_pack` next step.
6. `snow_depth` (cell-mean SWE, for albedo/diagnostics/restart) = column total SWE.

### Conservation tests to write (Stage 5, but author alongside Stage 3)
- **Water**: over a multi-step run, `Δ(Σ column SWE + Σ soil water + ponding)` equals
  `∫(snowfall + rain − ET − runoff)` to machine tol (uses `step_snow_column`'s own
  water closure + the Richards closure).
- **Energy at the seam**: the flux the column reports at its base (`G_bottom`) equals
  the flux the soil receives at its top, and `drainage_heat` is added exactly once —
  assert the coupled snow+soil enthalpy budget residual ≈ 0 on an analytic cold
  column (no double-booked `L_f`).

## Stages (each independently tested + committed)

- **Stage 1 — promote the module + config gate. ✅ DONE (2026-07-13, commit
  a45f1e70d).** Moved `_future/snow_column.py` → `land/snow_column.py`; added
  `snow_scheme: {"single","multilayer"}` (default `"single"` = current bulk
  `update_snow`) + `snow_column: SnowColumnConfig` to `MultiLayerLandConfig`, with a
  fail-fast dispatch guard at `_step_multilayer_land_impl` entry (unknown → ValueError;
  "multilayer" → NotImplementedError until Stage 3). Registered in
  `param_collector.SPEC_MODULES`; dispatch-hardening baseline + behavioural test.
  No physics change.
- **Stage 2 — state integration (no-op). ✅ DONE (2026-07-13, commit 57ae8fd8a).**
  Added `snow_column: SnowColumnState | None = None` to `MultiLayerLandState`; init
  seeds an empty pack for the multilayer scheme (`initial_snow_state`, new `dtype`
  kwarg); `step_multilayer_land` passes it through (preserves scan-carry pytree);
  restart save/load/merge round-trips it as an additive optional `snow_col_*` payload
  (no version bump — in-flight "single" v1 restarts still load) in BOTH
  `land/restart.py` and `run_lmip.py`. Restart round-trip + init + merge-skew tests.
  Physics unchanged (multilayer path still raises).
- **Stage 3 — couple it in (the physics). ✅ DONE (2026-07-13).** Wired into
  `step_multilayer_land` (two-leaf canopy), sequential operator split. Added the
  `thermal_active_swe` (~10 kg/m²) **zero-layer-snow threshold** (CLM5 §8.1): a thin
  dusting accumulates + brightens albedo but the surface flux passes through to the
  soil — without it, a full `Q_top` dumped into a ~0-heat-capacity fresh layer melts
  it in one step (found + fixed via the per-step probe). Verified: a cold column
  (258 K air) builds a real pack and holds the **soil surface 8.8 K warmer** than the
  snow-blind `"single"` budget (the cold-bias mechanism), **water conserves** to
  <1e-6 kg/m² over a mixed snow+rain run, all finite. Guards: multilayer + non-
  two-leaf → raise; multilayer + elev_bands → raise. Tests:
  `tests/land/integration/test_multilayer_snow_coupling.py` (+ 3 helper conservation
  tests in `test_snow_column.py`).
  - **Known first-cut limitation (Stage 4):** sub-threshold (thin) snow gets `Q_top=0`
    / `G_bottom=0`, so it accumulates + sublimates but does not melt-from-below off a
    warm soil until it either crosses the activation threshold or the single-scheme
    would have melted it; and rain-on-snow routes straight to soil infiltration
    (no in-pack refreeze). Both are documented, mass-conserving, and deferred.
- **Stage 3 (superseded design note).** Behind `snow_scheme=="multilayer"`,
  **sequential operator split** (the seam is already explicit via `G_bottom`, so a
  lagged canopy↔pack-top boundary is consistent, conservative, and avoids surgery on
  the converged Newton closure): canopy(lagged pack-top T) → column(Q_top, G_bottom)
  → soil(G_bottom). Scoped to the `two_leaf_canopy` surface scheme first (the LMIP +
  flux-site config), with an explicit guard raising on other schemes + multilayer.
  The module ingests **snowfall only**, so the coupler handles two mass terms outside
  it, conservatively:
  - **Sublimation** — the latent partition already removes `sublim_actual` from the
    cell; for a packed column, remove that ICE mass (+ its sensible enthalpy) from the
    pack top via `apply_sublimation` (the L_s energy is already in `Q_top` as
    `−lhflx`, so mass-removal + Q_top together close the seam).
  - **Rain-on-snow (v1)** — `precip_rain` on a packed cell routes straight to soil
    infiltration (`flux_top`), mass-conserving but skipping in-pack refreeze; a Stage-4
    refinement (main's #902 rain-on-snow refreeze is banded-path only anyway). Logged
    as a known first-cut limitation.

  `G_bottom = g_iface·(T_pack_base − T_soil_top)`, `g_iface` = harmonic mean of the
  pack-base half-conductance (`snow_base_interface_conductance`) and the soil-top
  half-conductance (`k_soil_top/(½·dz_top)` from `compute_thermal_conductivity`),
  positive **downward**. Same `G_bottom` to `step_snow_column` base AND the soil top
  BC; `drainage/rho_w` → `flux_top`; `drainage_heat/dt` → soil-top energy; `L_f`
  handled inside the column (single-scheme `melt_energy`/`update_snow` bypassed for
  packed columns). `snow_depth` (cell-mean) := column total SWE for
  albedo/diagnostics/restart. Empty-pack columns: `G_bottom = Q_top`, column inert.
  Conservation tests (water + energy seam residual ≈ 0) + a controlled cold-cell
  comparison (single vs multilayer): the boreal soil must warm toward physical winter
  values and stop overcooling to 221 K.
- **Stage 4 — snow albedo + cap.** Blend snow into the canopy shortwave soil
  albedo (so the surface reflects like snow); the pack's own drainage handles the
  tower, but add an explicit `h2osno_max` shed-to-runoff as a belt-and-braces cap.
  (Albedo lands WITH/after insulation — alone it would worsen the cold bias.)
- **Stage 5 — enable + validate.** Flip the 10-yr template to
  `snow_scheme: multilayer`; re-run the boreal probe; confirm `T_soil` realistic,
  snow bounded, `reverts.nc` ≈ 0, and global fluxes still physical.

## Notes / references

- CLM5 Tech Note (Oleson et al. 2013) §6.1–6.3 (snow-on-ground thermal column),
  §7 (`h2osno_max` capping), §3.2 (snow albedo blend).
- Sturm (1997) `k(ρ)` (in the module); fresh/max snow density Anderson-1976/CLM5.
- The single-bulk `snow_budget.update_snow` stays the default (`snow_scheme=
  "single"`) — backward-compatible; multilayer is opt-in until validated.
