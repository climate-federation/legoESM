# Bottom drag double-counting in explicit-substep barotropic solver

*Created: 2026-04-29*
*Affects: lat-lon C-grid and MPAS, both with `barotropic_solver="explicit_substep"`*
*Severity: ~2× drag in spun-up steady states; tens of Sv on Drake transport.*

## Problem

Linear bottom drag `r` is applied along two non-equivalent paths in the
explicit-substep solver:

1. **Path 1 — bottom-cell explicit Euler in the 3D tendency**
   - lat-lon: `src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:1273-1275`
   - MPAS:    `src/legoesm/ocean/dynamics/ocean_pe_mpas.py:321-324`

   Adds `du/dt[..., -1] += -r·u[..., -1]/dz_bot` to the 3D tendency.
   Its depth-mean (`-r·u_bot/H`) flows into `F_slow_u/v` and reaches the
   barotropic solver correctly.

2. **Path 2 — implicit decay factor on depth-mean inside barotropic substep**
   - lat-lon: `src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:339-344`
   - MPAS:    `src/legoesm/ocean/dynamics/barotropic_mpas.py:271-273`

   `U_bar ← U_bar / (1 + dt_s · r / H)` applied at every substep using
   the same `r`.  This is the "textbook shallow-water linear drag" from
   the original 2D-only formulation; it was retained when the 3D
   bottom-cell drag was added later.

Both paths use the same `bottom_drag_r`, so to leading order the model
applies ~2× the intended drag in any spun-up state.

## Why it persists

Path 2 is unconditionally stable and is what the explicit substep
relied on before the 3D bottom-cell drag was implemented; it never got
removed.  Path 1 is correct and physical (damps the actual bottom-cell
velocity, including any shear), but its depth-mean projection is what
reaches the barotropic mode — meaning if path 2 is also active, drag is
applied twice on the depth-mean.

## What's already clean

- **lat-lon implicit Crank–Nicolson solver**
  (`barotropic_implicit_latlon_cgrid.py`) — never had path 2.
- **MPAS implicit Crank–Nicolson solver**
  (`barotropic_implicit_mpas.py`) — path 2 removed 2026-04-29.

So new experiments using `barotropic_solver="implicit_cn"` on either
grid are unaffected.

## What's still affected

- Any experiment using the default `barotropic_solver="explicit_substep"`.
- This includes most ocean-test-matrix entries.

## Fix design (deferred)

Same plan as documented in `docs/ocean/experiments/global_overturning_plan.md`
Phase 2:

1. Switch path 1 from explicit Euler to implicit per-cell:
   ```
   u_bot ← u_bot / (1 + dt · r / dz_bot)
   ```
   Unconditionally stable, AD-friendly. Then take its depth-mean for
   `F_slow_u/v` as before.
2. Delete path 2 from the explicit-substep barotropic solvers
   (`barotropic_latlon_cgrid.py` and `barotropic_mpas.py`).
3. Add a config flag `bottom_drag_location: Literal["depth_mean",
   "bottom_cell"]` so existing experiments can keep their current
   behavior on request (`"depth_mean"` = both paths, back-compat;
   `"bottom_cell"` = cleaned up).
4. Two unit tests:
   - Flat-bottom rest-state spinup converges to
     `U_baro ≈ τ/(ρ_0·r)` within 5 % under `"bottom_cell"`.
   - Drake-channel `u_bot` decay timescale ≈ `dz_bot/r` under
     `"bottom_cell"`.

## Workaround for now

Use `barotropic_solver="implicit_cn"` for any experiment where bottom
drag matters quantitatively (anything spinning up to a steady state
against a wind stress; ACC; gyres).
