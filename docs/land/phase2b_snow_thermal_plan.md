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

Per step, snow-covered path:
1. Canopy computes the surface energy balance → `Q_top` = net flux into the
   surface (`sw_net + lw_net − shflx − lhflx`). The canopy's ground/skin
   temperature is the **pack-top T** (lagged from the carried snow state) when snow
   is present, else the soil top (current behaviour).
2. `G_bottom` = snow-base↔soil-top conductive flux
   `= g_iface·(T_snow_base − T_soil_top)` with `g_iface` the harmonic mean of the
   snow-base and soil-top-half conductances (lagged T → explicit, stable operator
   split).
3. `step_snow_column(state, precip_snow, T_air, Q_top, G_bottom, dt)` →
   `(new_snow, drainage, drainage_heat)`.
4. Soil top BC: `solve_soil_thermal(G_surface = G_bottom, ...)` (snow-attenuated),
   plus `drainage_heat` folded into the top-layer energy; `drainage` → the Richards
   `flux_top` (infiltration) / runoff.
5. **Empty-pack limit** (no snow): bypass the column, `G_bottom = Q_top` (identical
   to today) so a snow-free cell is bit-unchanged.

## Stages (each independently tested + committed)

- **Stage 1 — promote the module + config gate (DONE-marker below).** Move
  `_future/snow_column.py` → `land/snow_column.py`; add `snow_scheme:
  {"single","multilayer"}` (default `"single"` = current bulk `update_snow`) +
  `snow_column: SnowColumnConfig` to `MultiLayerLandConfig`, with a fail-fast
  dispatch guard. No physics change.
- **Stage 2 — state integration (no-op).** Add `snow_column: SnowColumnState |
  None = None` to `MultiLayerLandState`; init (empty pack) + restart round-trip +
  every `MultiLayerLandState(...)` call site. Physics unchanged (not yet coupled).
- **Stage 3 — couple it in (the physics).** Behind `snow_scheme=="multilayer"`,
  replace the cell-mean `update_snow` with the coupling above; pack-top T as skin
  T; drainage → soil/runoff; `G_bottom` → soil top BC. Conservation tests
  (water + energy closure at the coupled seam) + a controlled cold-cell comparison
  (single vs multilayer): the boreal soil must warm toward physical winter values
  and stop overcooling to 221 K.
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
