# Phase 2b — CLM-faithful multi-layer snow thermal

> **STATUS: SUPERSEDED — global validation FAILED. Do NOT enable
> `snow_scheme="multilayer"` in production.** The scheme was built (commits
> `a45f1e70d`…`f86d12206`) and passed its unit tests, but the 10-yr global run
> (`lmip_canopy_10yr_mlsnow`, 2026-07) is **unstable** in snow/cold regions. The
> production LMIP config runs the stable `single` scheme. This doc is retained
> because the multi-layer code still lives in the repo (gated off) and this records
> what was built, why it failed, and the fix path. See the memory note
> `mlsnow-unstable-global.md`.

## What this was trying to fix
The `single` snow budget (`snow_budget.update_snow`) is *snow-blind*: the surface
energy flux hits the thin (0.025 m) top soil layer directly, so boreal/Arctic winter
`T_soil` overcools toward air (~221 K) and SWE accumulates unbounded (snow tower).
These are **finite realism** problems, not crashes. The intended fix: wire the
validated multi-layer snow column (`land/snow_column.py` — equal-mass remap, enthalpy
phase change, conservative drainage) between the surface and the top soil so the pack
**insulates** the soil (CLM, Oleson 2013 §6).

## What was built (all gated behind `snow_scheme="multilayer"`, two-leaf only)
- `land/snow_column.py` — the multi-layer snow core + coupling helpers
  (`snow_base_interface_conductance`, `apply_sublimation`, `pack_top_temperature`).
- `MultiLayerLandState.snow_column`; init + restart round-trip (additive `snow_col_*`
  payload).
- Coupling in `multilayer_land.py`: a **sequential operator split** — canopy(lagged
  pack-top T) → `step_snow_column(Q_top, G_bottom)` → `solve_soil_thermal(G_bottom)`,
  with the same `G_bottom` at both ends of the snow↔soil seam, a `thermal_active_swe`
  (~10 kg/m²) zero-layer-snow threshold, and sublimation/drainage routed
  conservatively.
- Unit tests (insulation +8.8 K, water + energy conservation) **passed**.

## Why it failed globally (root cause)
The unit tests used **constant** forcing and short runs; the global run exposed an
instability the tests could not:
- **46% of land cells** (3676/8006) hit the NaN-revert guard; final `T_soil_top`
  ran to ~1.4×10⁹ K, `snow_depth` to ~3.7×10¹⁴ kg/m² (large-but-finite, so the
  NaN-guard's "PASS" was misleading). Blowup grows from month 0.
- **100% confined to snow/cold latitudes** (Antarctic + boreal/Arctic); tropics +
  mid-latitudes were clean and physical.
- A single-column repro with **constant** cold forcing is stable even at 223 K — so
  steady cold is not the trigger.

Mechanism: the seam is an **explicit operator split**. The soil side is semi-implicit
(Robin term), but the **snow-column base takes `G_bottom` as a fully explicit
prescribed flux**. With a thin base layer (small heat capacity), a large interface
conductance (`g_iface ≈ 120 W/m²/K`), and `dt = 3600 s`, that explicit flux can swing
the base temperature ~30 K in one step. Under **time-varying** forcing a cell whose
SWE oscillates across the hard `thermal_active_swe` threshold also switches the soil
BC discontinuously — together these drive a growing oscillation.

## Fix path (if revisited)
Solve **snow layers + soil layers as ONE implicit tridiagonal column** each step,
with the snow-base↔soil-top interface as an *internal* implicit interface (CLM's
combined snow+soil solve) — no lagged explicit seam. Challenges: fixed-shape
zero-mass snow layers must be conditioned to be thermally transparent for an empty
pack (JAX wants static shapes, vs CLM's dynamic `snl`); preserve differentiability +
freeze/thaw + the canopy Newton coupling. A cheaper interim (both-sides Robin-implicit
+ a smooth activation ramp instead of the hard threshold) may reduce but likely will
not eliminate the instability. **Prerequisite for any fix:** a deterministic in-repo
**time-varying-forcing multi-step stability test** that reproduces the blowup — the
current unit tests pass while the global run fails.
