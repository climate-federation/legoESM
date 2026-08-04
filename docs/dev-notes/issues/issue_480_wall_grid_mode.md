# #480 — WENO vector-invariant eddy-permitting blow-up: a masked-wall tracer artifact

## Symptom
The lat-lon C-grid WENO **vector-invariant** ocean momentum (`momentum_advection="weno9"`,
W9V) NaNs at ~**day 11** on the eddy-permitting Silvestri §5 baroclinic jet
(160×128×50, no closure, `stabilize=False`). Coarser/closure runs are fine.

## Root cause (DIAGNOSED + FIXED at the source)
The blow-up is a **2Δx-in-lon, v-dominant, ROTATIONAL grid mode that grows only in the
first/last ~8 N/S free-slip-wall rows** (the interior stays clean). Its source is a
**spurious near-wall buoyancy front created by the masked-land row**:

- legoESM represents the free-slip wall as a **masked LAND row** (row 0 / row −1,
  `land_mask=0`) whose tracer is set to the masked fill value **T = 0 °C** (a cold cell),
  while the first wet row is ~12.5 °C — a spurious Δb ≈ 0.025 m/s² step right at the wall.
- The **wide WENO7 tracer stencil** (half-width 4) at the first wet v-faces reaches into
  that cold land cell, so the reconstructed face value carries a spurious meridional
  gradient. A 2Δx-in-lon `v` perturbation advecting this spurious front
  (`−v·∂T/∂y`) manufactures grid-scale buoyancy at the wall, which feeds the
  density→pressure→PGF→v feedback. The mode is **un-dissipatable by advection** (≈zero
  zonal velocity), so it grows until it NaNs.

### The Oceananigans oracle nailed it (wall tendency-bridge)
Feeding the **identical** 2Δx-in-lon wall-`v` perturbation into both codes and measuring
the **v→buoyancy gain at the wall** (one step, linearised response):

| code | wall v→buoyancy gain | vs oracle |
|------|----------------------|-----------|
| legoESM (before fix) | `5.20e-8` (`g·α·dT`) | **68×** |
| **legoESM (after fix)** | `9.25e-10` | **1.2× (matched)** |
| Oceananigans (z, no closure) | `7.64e-10` | 1.0 |

The `w`-link (continuity) is innocent (grid-scale `w` at the wall ≈ 0, wall/interior 0.1),
so the excess is purely the **horizontal tracer advection of the spurious cold-cell front** —
exactly the masked-land-row difference vs Oceananigans' clean grid-edge wall.
(Probes: `scripts/tmp/_silvestri_vrho_gain.py`, `_silvestri_w_gain.py`,
`_silvestri_dTdy_wall.py`; oracle counterpart `/tmp/ocn_silvestri/vrho_gain.jl`.)

### Ruled out as the cause (with the method)
WENO momentum scheme (tendency-bridge `bridge_momentum.jl`: 0.23 vs 0.248 grid-scale
dissipation — MATCH); vertical resolution; EOS (linear, matches); tracer-advection scheme
itself; grad/curl operators (`curl(grad)=0`); spherical metric; **z\*** (Oceananigans-in-z\*
stays bounded → innocent); Coriolis; continuity/`w` at the wall.

## Fix (FAITHFUL, shipped, default-on)
**Zero-gradient (Neumann) fill the tracer over the DEAD cells BEFORE the flux-form
advection reconstruction** — `LatLonCGridOceanConfig.tracer_wall_neumann_fill`
(default **True**). The reconstruction now sees a **flat extension** across solid walls
instead of the masked cold cell — i.e. the physical **no-flux insulating wall**, which is
exactly what Oceananigans' clean grid-edge wall does. Implemented as `recon_fill_mask`
threaded into `_compute_advection_flux_div` / `compute_advection_flux_div_pair` /
`_ssp_rk3_tracer_pair_step` (the fill happens inside the reconstruction; the flux-form
UPDATE and land-gating keep the ORIGINAL tracer, so stored dead-cell values are unchanged).
The mask is the **per-level `active_3d` (`is_active`)**, not the 2D surface land_mask, so
the fill also cleans **topographic-step dead cells** in partial-cell runs (a column wet at
the surface but dead below the partial seafloor — the same wide-stencil contamination at
depth). `neumann_fill_cgrid` was made 3D-mask-aware (a lower-rank mask broadcasts over the
level axis = the historical 2D behaviour; a rank-matching `(…,nlev)`/`(…,1)` mask is
applied per level). Flat-bottom runs pass a `(n_lat,n_lon,1)` singleton mask ⇒ bit-identical
to the 2D path (§5 unchanged).

- **NOT a viscosity/closure backstop** (distinct from the rejected A_h/Smag/biharmonic):
  it adds zero dissipation; it removes a spurious *source*.
- **Conservative**: the wall-face flux stays zero (carried by `mass_flux_u/v`), so the fill
  only redistributes tracer *within* the wet domain — it never sources/sinks it.
- **Strict no-op where there is no land** (periodic/global aquaplanet): interior values are
  bit-identical; `recon_fill_mask=None` reproduces the legacy path exactly.
- **`jax.grad` + JIT safe** (verified; reuses the AD-safe `neumann_fill_cgrid`).
- **SPMD/MPI-correct** (the fill's pole/fold edge handling is gated by
  `lat_ends_are_poles()` + `spmd_pole_end_masks()`).

### Validation
- §5 W9V no-closure **SURVIVES 20 days** at max|u|≈0.067 = the Oceananigans-oracle
  amplitude, with **`wall_grid_filter_rate_s=0.0` (the symptom filter OFF)** and only the
  faithful tracer-wall fill (`scripts/tmp/_silvestri_faithful_validate.py`, GPU ~80 s).
  Set `tracer_wall_neumann_fill=False` to reproduce the day-11 blow-up.
- `tests/ocean/unit/test_tracer_wall_fill.py` — 5 tests (None=legacy identity, no-land
  no-op, fill active-at-wall / inert-in-interior, wet-domain conservation, config-flag
  through the step).
- No regression: `test_advection_weno.py`, `test_tracer_pair_advection.py`,
  `test_advection_som.py` (62), `test_weno_momentum.py` (51), `test_realistic_coastlines.py`
  (10), `test_advection_grad_underflow.py` (11), `test_no_scheme_duplication.py`,
  `test_wall_grid_filter.py` — all green.

## Superseded symptom fix (retained, default-off)
The earlier boundary-localised 2Δx-in-lon Shapiro filter on the wall rows
(`_bc_wall_grid_filter`, `LatLonCGridOceanConfig.wall_grid_filter_rate_s`, default 0.0) is
now **superseded** by the faithful root fix and defaults OFF everywhere (including
`SilvestriJetConfig`). It is kept as an optional belt-and-braces boundary smoother.

## Open items
1. **Codex adversarial review** still owed (per repo policy) — could not run in this
   environment (no `codex` binary / plugin); run before merge.
2. The same masked-cell contamination affects **any** masked-land/partial-cell WENO ocean
   run; now fixed by the default-on `active_3d` fill (lateral walls AND topographic steps).
   Adversarial-review Findings 1–3 addressed: (1) clarified that the filled tracer also
   feeds vertical advection and is safe because only dead cells change (gated out);
   (2) switched to the per-level `active_3d` mask so partial-cell topographic steps are
   covered; (3) explicit ack that the fill is a real (conservation-neutral, default-on)
   change to every coastline run, not literally a no-op there.
3. **Perf follow-up** (adversarial-review note): in the RK3 tracer path the fill runs per
   tracer per stage (3 stages × 2 tracers = 6 `neumann_fill_cgrid` calls/step ≈ 18 halo
   passes/step) — correct but a real MPI cost. Could be halved by filling the stacked
   `[T‖S]` once per stage (filling before the vertical step is harmless: it only changes
   land cells, and wet columns — the only ones the vertical advection keeps — are
   unchanged). Not done now to avoid destabilising the validated fix; single-node §5 is
   unaffected (~80 s / 20 days).
