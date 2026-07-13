# Boreal / Arctic NaN blow-up — diagnosis & fix plan

**Symptom.** The 10-yr 2° biophysics run (`two_leaf_canopy` + MOST + multilayer
Richards + freeze/thaw ON, carbon off) produces physically reasonable output over
most of the globe but a considerable number of **boreal / Arctic** land cells
diverge to NaN. A 10-day January probe (2°, GPU) showed `T_soil_top` down to
~220 K and `snow_depth` up to ~178 kg/m² on the affected cells, and the run
reports `FAIL` (~5% of land cells NaN).

## Root-cause diagnosis

The `two_leaf_canopy` path is missing the surface-thermal stabilization that
`simple_seb` has, and is snow-blind in its own energy balance:

1. **PRIMARY — explicit (not semi-implicit) surface thermal BC.** SimpleSEB
   returns a linearised `surface_conductance` so the soil-thermal solve damps the
   surface-temperature feedback (Robin BC); its docstring says this "removes the
   explicit-coupling large-dt/thin-layer/stiff-surface instability that otherwise
   diverges to NaN" (`surface_scheme/simple_seb.py:221-259`). The canopy returns
   `surface_conductance=None` (`surface_scheme/two_leaf_canopy.py:549-581`), so the
   final soil solve runs **explicitly** (`multilayer_land.py:796`). Over a cold,
   stiff, thin-top-layer boreal surface this is the blow-up.
2. **COMPOUNDING — cell-mean canopy energy balance never sees the snow.**
   `G_surface = surface_out.G_soil` uses the canopy's *vegetation* albedo
   (`ALB_VIS≈0.1/ALB_NIR≈0.2`) and no snow insulation on the non-banded path; snow
   albedo/insulation only enter the *banded* branch (`multilayer_land.py:604-619`,
   off in this run). A cell under 178 kg/m² of snow gets its soil overcooled toward
   air temperature (~220 K) — the stiff state that then diverges via (1). (The
   albedo *reported* to the coupler is snow-aware after the 2026-07 fix, but the
   canopy's *internal* G is not — this is the gap.)
3. **No safety net.** `multilayer_land.py` has no `isfinite` guard, so one cell's
   NaN poisons its own future steps and the run-level PASS/FAIL.

Ruled out as the origin: freeze/thaw soil thermal is well-conditioned (backward-
Euler Thomas system is diagonally dominant even at 220 K); the Newton solver is
finite-guarded; the "snow tower" is a symptom, not a NaN generator; the Jan-1
cold-start is benign as a transient (the blow-up is an accumulated phenomenon).

### Is there a flux-tower (EC-site) config that already fixes this? No.

The EC-site harness (`scripts/run/run_ec_site.py`, `boundary_data/ec_site.py`)
*runs* cold/boreal FLUXNET sites (FI-Hyy, CA-Qfo, US-NR1, DE-Obe) to completion,
but with the **same** scheme — it survives them via **numerical/init
infrastructure, not physics** (its own docs flag frozen/cold-season soil
evaporation as an *open, unsolved* item). The two directly-borrowable pieces:
- an **atomic per-step NaN-revert carry** (`run_ec_site.py:297-334`), and
- a **realistic soil-T warm-start** (observed near-0 °C sub-snow profile) instead
  of broadcasting Jan-1 air temperature onto the whole column
  (`run_lmip_biophys.py` cold-start vs `run_ec_site.py:268-282`).
It also runs freeze/thaw *off* (sidesteps the problem) — not something to copy.

### Do main's newer commits fix this? No.

Checked all commits on `origin/main` ahead of this branch, per file:
`soil_thermal.py`, `simple_seb.py`, `core/bulk_flux.py`, `canopy/solver.py` have
**zero** changes. `#917`'s canopy work is *reverse-mode differentiability*
(gradient NaNs) + latent-mass conservation over snow — not the forward thermal
blow-up. `#902`'s rain-on-snow refreeze fix is in the *banded* snow path (off
here). **Merging main would not fix the boreal NaN.** (Main does have unrelated
good land fixes worth syncing eventually — a latent-mass-over-snow conservation
fix, snow-age albedo — but keep that separate from this.)

## Re-diagnosis from the 1-yr Phase-2a run (2026-07-13)

The revert map (`reverts.nc`) overturned the original diagnosis. The NaN cells are
**not** the cold thermal blow-up:
- **357 cells revert EVERY one of 8760 steps** (median revert fraction 1.0) →
  chronic init-NaN (the guard reverts to the previous state; if that was never
  finite it stays NaN). They fail instantly, not gradually (count is identical at
  10 days and 1 yr: 358→357).
- They are **mid-latitude** (275), not Antarctica (0) — a coastline-following, not
  polar, pattern.

Root cause: **the CRU-JRA forcing regrid.** CRU-JRA is land-only (ocean = NaN);
`build_forcing_weights` built the KD-tree over the FULL grid, so a coastal model
column whose 4 nearest source cells are all ocean regridded to NaN → NaN
cold-start → chronic NaN. This is a forcing/masking gap, **not** a physics
divergence. Fixed by building the KD-tree from **land-only source cells**
(`compute_latlon_to_voronoi_weights(src_valid=...)`), so every column draws its
nearest *actual* land forcing (2026-07-13). Also removed the arbitrary
`land_frac_min=0.5` output cutoff (default → 0.0 = any surfdata land) and now emit
`land_fraction` per cell for area-weighting in analysis.

Consequences for the plan: **Phase 2a was correct but not the cause** of these
NaN (a real latent instability, kept). The revert map shows **no polar/thermal
divergence at all**. The genuinely-cold cells (`T_soil 221 K`, `snow 1580 kg/m²`)
are FINITE — a cold-bias / snow-tower **realism** issue, which is what Phase 2b now
targets (no longer a NaN emergency). Re-run after the forcing fix should show
`reverts.nc` ≈ 0 → confirming only the realism work remains.

## The plan

**Phase 1 — Robustness guard (DONE, 2026-07-13).** Atomic per-column NaN-revert
carry in `run_lmip_biophys`'s scan (`_nonfinite_per_col` + `jnp.where` revert):
a column whose state goes non-finite reverts to its previous state (columns are
independent), its step's fluxes are masked to NaN, `revert_count` tallies per
cell, and a `reverted` (0/1) tape var records the per-cell failure rate. Writes
`lmip_biophys.reverts.nc` (per-cell revert-step count) and a run-summary line.
Effect: the run **completes and PASSes** (finite state) instead of NaN-poisoning,
and you get a precise **map + count of where/how often cells fail** — the
diagnostic that tells us how much Phase 2 must do. This unblocks the baseline
today; the reverted cells are honestly flagged (masked output + the map).

**Phase 2 — Root-cause physics (the real fix).**
- **2a. Semi-implicit surface BC for the canopy — DONE (2026-07-13).** The canopy
  now returns `surface_conductance = λ = −∂G_soil/∂Ts ≥ 0` (analytic ground-LW
  slope `4·ε_s·σ·Ts³` + ground sensible `ρCp/rah_below` + ground latent via
  Clausius-Clapeyron), computed in `canopy/solver.py::canopy_forward` and set on
  `SurfaceFluxOutput.surface_conductance` in `two_leaf_canopy.py`.
  `multilayer_land.py:796` already forwards it, so the final soil-thermal solve now
  uses the Robin BC (mirrors `simple_seb.py:221-259`). Because every term is ≥0 it
  only ADDS to the tridiagonal diagonal — strictly stabilising, never destabilising.
  Validated: `surface_conductance` finite/positive (~80 W/m²/K, sensible-dominated);
  canopy suite (14) + albedo (2) + driver (13) + 2 new conductance tests pass; global
  4° smoke physical (no NaN). Real cold-cell confirmation is the Derecho re-run (Phase 4).
- **2b. Snow-aware cell-mean canopy G.** Apply snow albedo + insulation (+ `L_s`)
  to the non-banded `G_surface` so a snow-covered cell's soil isn't overcooled to
  220 K. Aligns the canopy's internal energy balance with the already-snow-aware
  reported albedo.

**Phase 3 — Cold-start.** Initialise the sub-surface soil T toward a physical
winter profile (or a short spin-up) rather than broadcasting Jan-1 air T down the
column.

**Phase 4 — Validate.** Re-run the 10-day Jan boreal probe with guard+fixes;
controlled comparisons (revert count with/without each change, and freeze/thaw
on/off); validate against the cold EC-sites (FI-Hyy, CA-Qfo). Target: revert /
NaN count → ~0.

**Sequence:** 1 (done) → 2a → 2b → 3 → 4. After Phase 1, read
`lmip_biophys.reverts.nc` from the next baseline to quantify the failure map
before sizing 2a/2b.
