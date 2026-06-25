# Scope — flux-form horizontal momentum advection (lat-lon C-grid)

> Status: **scoping only** (not implemented). The highest-priority "gap → block" item from the
> Veros ACC missing-blocks ledger (`oracle_recipe_strategy.md` §8, doctrine rule H). This closes
> the `du_adv` corr-0.58 delta and is broadly useful (MOM6/MITgcm/NEMO/Veros all use flux-form).

## Why

legoESM's lat-lon C-grid momentum advection is **vector-invariant only**
(`momentum_advection ∈ {vector_invariant, weno5, weno7}` — all compute ζ at vertices then form
the PV flux `q × mass_flux`). Veros and most z-coordinate models use **flux-form** `∇·(uu)`.
This is a genuine *missing method*, not a wiring bug — so per rule H it becomes a new
config-selectable option in the canonical dycore, gated on the truth tiers (not on the oracle
match), reusing existing shared blocks rather than a `veros_*` clone.

## What to reuse (do NOT reinvent)

The tracer side already has the full flux-form machinery; momentum reuses it:
- `divergence_cgrid`, `gradient_x_cgrid`, `gradient_y_cgrid` (`latlon_cgrid_operators.py`)
- face interpolation / limiter schemes at u/v points: `_upwind_to_u/v_points`,
  `_tvd_to_u/v_points` (in `ocean_pe_latlon_cgrid.py`); `dst3_to_u/v_points`,
  `ppm_to_u/v_points`, `weno5/weno7_to_u/v_points` (`ocean/advection.py`)
- `min_cell_to_uface` / `min_cell_to_vface` (min-rule face thicknesses — **must** match the PV
  path's convention so the two schemes are comparable under conservation tests)
- `compute_face_masks_3d`, `_neumann_fill_cgrid`
- `_compute_advection_flux_div` in `ocean_model_latlon_cgrid.py` — the reference tracer
  flux-div dispatch pattern to mirror for momentum
- the just-extracted `_bc_*` substage seam: the new code is one more substage,
  `_bc_horizontal_momentum_advection_flux_form`, swapped in where `_bc_pv_flux` is called.

## New code

1. **`_bc_horizontal_momentum_advection_flux_form(du_dt, dv_dt, u, v, u_prime, v_prime, h_u,
   h_v, h_k, u_mask_3d, v_mask_3d, mask, grid, config, z_coord, rho_0)`** (~200–250 LOC):
   compute horizontal mass fluxes `h_u·u`, `h_v·v`; interpolate `u→v-points` and `v→u-points`
   with the chosen scheme; form momentum fluxes; take the divergence; mask + accumulate; return
   `(du_dt, dv_dt, diag_horiz_mom_flux_u, diag_horiz_mom_flux_v)`.
2. **Config surface** (`state.py LatLonCGridOceanConfig`): add `"flux_form"` to the
   `momentum_advection` literal set + a new `momentum_flux_scheme: str = "upwind"`
   (`"upwind"|"tvd"|"dst3"|"weno5"|"weno7"`, only used when `momentum_advection="flux_form"`).
3. **Dispatch + fail-fast validation** in `latlon_cgrid_ocean_baroclinic_tendencies` /
   `_validate_config`: `ValueError` on unknown `momentum_advection` (dispatch discipline — the
   current code silently falls through; fix this regardless), branch to the new substage when
   `"flux_form"`, else `_bc_pv_flux`.
4. **Diagnostics**: add `horiz_mom_flux_u/v` (or reuse the `vortcor` slot, mapped) to
   `MomentumTendencyDiagnostics` so the closure test (`Σ components == du_dt`) still holds.

## Truth-tier tests required (gate — BEYOND oracle-matching)

- **Tier 0 conservation**: zero forcing in a closed/periodic box → total KE conserved to
  machine ε; **zero-velocity → zero advective tendency** exactly.
- **Tier 1 equivariance**: rigid-rotation equivariance (`u→−v, v→u`); uniform-flow analytic
  (constant zonal flow on flat periodic bottom → zero advection); differentiability
  (`jax.grad` finite/nonzero, comparable to vector-invariant).
- **Tier 2 idealized**: Stommel + Munk gyres — stable, physically-sound KE growth→plateau;
  compare strength/boundary-layer width vs vector-invariant (expect schemes to differ by their
  implicit dissipation; bound, don't force-match). High-CFL stability (upwind stable; WENO may
  oscillate, per `test_advection_weno.py`).
- **Tier 3 oracle** (last, lowest trust): re-run the ACC tier-2 comparison with
  `momentum_advection="flux_form"` and check `du_adv` corr improves toward Veros.

## Risks

- C-grid stagger off-by-one / periodic-wrap on the cross-face interpolation (`u@v` vs `v@u`).
- Partial-cell `h_u/h_v` weight consistency with the PV path's min-rule (or conservation tests
  drift).
- Sign convention on cross-face interpolation (easy to flip).
- Energy-conservation tightness: if O(1e-3) KE drift appears in the inviscid test, suspect the
  `h_u/h_v` weighting or `rho_0` scaling first.

## Rough size

~500–800 LOC: substage ~200–250, config + dispatch + validation ~50, unit + closure tests
~150–200, idealized-gyre integration test ~100. Single focused PR, gated by the truth-tier
tests above before the oracle re-run.
