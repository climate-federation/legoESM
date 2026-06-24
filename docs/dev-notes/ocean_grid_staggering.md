# Ocean Grid Staggering

This document describes the grid staggering choices across all ocean
discretizations in legoESM, explains the rationale for the cubed-sphere
barotropic A-grid solver, and documents the trade-offs.

## Current staggering by grid type

| Grid | Baroclinic | Barotropic | Velocity representation |
|---|---|---|---|
| **Cubed-sphere** | C-D grid (D-grid corners for momentum, C-grid edges for transport); A-grid storage | A-grid (cell-center) | Grid-aligned (u, v) at centers `(6,n,n,nlev)`, converted to D-grid corners for dynamics |
| **Lat-lon** | A-grid (all co-located) | A-grid | Geographic (u, v) at cell centers |
| **MPAS** | C-grid (TRiSK): velocity on edges, scalars on cells, vorticity on vertices | C-grid (same) | Edge-normal scalar per edge `(nEdges, nlev)` |
| **Spectral** | Vorticity-divergence spectral; A-grid in physical space | No split-explicit — unsplit SSP-RK3 with eta hyperdiffusion | Spectral vor/div; geographic (u, v) diagnosed in grid space |

## Key observations

- **MPAS is the only true C-grid** — staggering is consistent between
  baroclinic and barotropic modes, with velocity on edges throughout.
- **Cubed-sphere has a staggering mismatch**: the baroclinic dynamics use
  C-D grid internally (D-grid corners for vorticity/Coriolis via
  Arakawa-Lamb, C-grid edges for mass/tracer transport), but the
  barotropic solver operates on A-grid cell-center data using
  centered-difference gradient/divergence. Conversions
  (`_center_to_dgrid_3d` / `_dgrid_to_center_3d`) happen at the
  tendency interface.
- **Lat-lon is consistently A-grid** everywhere.
- **Spectral doesn't split barotropic/baroclinic** — it uses unsplit
  SSP-RK3 and controls the barotropic mode via spectral hyperdiffusion
  on eta.

## Why the cubed-sphere barotropic solver uses A-grid

The mixed staggering on the cubed sphere is a deliberate trade-off:

1. **Different numerical priorities.** The baroclinic mode needs accurate
   vorticity dynamics (Coriolis, nonlinear advection, PV conservation)
   — the C-D grid excels at this via the Arakawa-Lamb scheme. The
   barotropic mode primarily resolves fast gravity waves (eta
   propagation) over many substeps, where the main requirements are mass
   conservation and stability, not high-order vorticity numerics.

2. **Complexity cost.** C-D grid on cubed-sphere panels requires vector
   rotation across panel boundaries, separate halo exchanges for corner
   vs. edge quantities, and careful treatment of cube vertices. Running
   that machinery 30x per baroclinic step (once per barotropic substep)
   would multiply implementation complexity and halo exchange cost for
   marginal accuracy gains on what is essentially a 2D shallow-water
   problem.

3. **Compensating diffusion.** The `barotropic_diffusion_alpha` parameter
   exists to damp computational modes (checkerboard noise in eta) that
   A-grid staggering is prone to. This is an acknowledged trade-off:
   simpler operators + tunable damping, rather than inherently
   noise-free C-grid staggering at higher implementation cost.

4. **Coupling only requires depth-averages to match.** The
   barotropic-baroclinic velocity reconciliation
   (`u_new = (u - U_bar_old) + U_bar_new`) preserves the baroclinic
   vertical structure regardless of what staggering the barotropic
   solver uses.

## Potential risks of the A-grid barotropic solver

The A-grid barotropic solver could be a source of issues for experiments
where:

- Barotropic Rossby waves or coastal Kelvin waves matter
  (computational mode contamination)
- Strong bathymetric gradients interact with the barotropic pressure
  gradient
- The tunable diffusion coefficient needs careful case-by-case
  adjustment

Upgrading to a C-grid barotropic solver on the cubed sphere would
address these but is a nontrivial undertaking.

## Recommendation

**Cubed-sphere C-grid barotropic** is worth doing when boundary
conditions become a practical blocker — e.g., when running experiments
with complex coastlines, realistic bathymetry, or when the Neumann fill
issues require yet another round of debugging. The C-D grid operators
already exist in `operators_cdgrid.py`.

**Lat-lon C-grid** should be deferred unless lat-lon becomes a primary
production grid. The cost is substantially higher (new operator suite)
and the benefit is proportionally the same.

In either case, **cell-center state storage should be preserved** as the
external API. C-grid staggering should be internal to the barotropic
solver, with no impact on physics, coupler, diagnostics, or ML
interfaces.

## Relevant files

| File | Description |
|---|---|
| `src/legoesm/ocean/dynamics/ocean_pe_cdgrid.py` | Cubed-sphere baroclinic (C-D grid) |
| `src/legoesm/ocean/dynamics/barotropic.py` | Cubed-sphere barotropic (**A-grid**) |
| `src/legoesm/ocean/dynamics/ocean_pe_latlon.py` | Lat-lon baroclinic (A-grid) |
| `src/legoesm/ocean/dynamics/barotropic_latlon.py` | Lat-lon barotropic (A-grid) |
| `src/legoesm/ocean/dynamics/ocean_pe_mpas.py` | MPAS baroclinic (C-grid TRiSK) |
| `src/legoesm/ocean/dynamics/barotropic_mpas.py` | MPAS barotropic (C-grid) |
| `src/legoesm/ocean/dynamics/spectral_ocean_pe.py` | Spectral (unsplit RK3) |
