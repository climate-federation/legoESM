# Partial cells for the lat-lon C-grid ocean — implementation plan

**Status (2026-04-30):** Scoping only.  No implementation in tree.

## Motivation

Phase 3a of `realistic_geometry_lat_lon_plan.md` empirically established
that the existing pure z\* coordinate fails the Beckmann-Haidvogel
seamount stress test on real ETOPO bathymetry at 2.5° resolution.  MEO
(Mellor-Ezer-Oey) preprocessing alone requires an r-factor cap of 0.03
to stabilise the model — corresponding to a +32% volume change that
eliminates continental shelves and slopes.  At that point the
"realistic geometry" promise of Phase 3.5 / 4 is not credible.

The standard production fix in modern ocean modelling is **z\* + partial
bottom cells**: keep z\* for the free-surface treatment, but allow the
deepest active cell at each column to have a reduced thickness so the
seafloor is at the correct geometric depth.  This is what MOM6 (GFDL),
modern NEMO (CMIP-class), MITgcm, and POP/CESM all use.

The PGF-error reason this works:

- For **full cells** (the upper N−1 levels), `z_k` is identical at
  every column.  ∂z_k/∂x = 0 → the PGF cancellation problem
  is **eliminated** in the bulk of the water column.
- The PGF cancellation problem is reduced to **one partial bottom cell
  per column**, where Adcroft 1997 / Adcroft & Campin 2004 derived
  special-case stencils that handle the variable thickness cleanly.

Empirically, this halves the Phase 3a r-factor problem (the partial
bottom cell still has some r-factor sensitivity, but only one layer
per column instead of every layer) and is the demonstrated production
solution at 25+ years of operational use.

## Why not d-J PGF instead

After detailed discussion (see git history of
`realistic_geometry_phase3a_results.md` and `realistic_geometry_lat_lon_plan.md`),
the chosen architectural progression is:

```
z* + MEO  (now)  →  z* + partial cells  (this plan)  →  generalized vertical coordinates  (future)
```

d-J PGF is skipped:

- It's the σ-coordinate equivalent of partial-cells-fixing-z\* (used
  by ROMS / CROCO because σ doesn't have full cells to begin with).
- ~500 LOC of work that doesn't compose with later moves to partial
  cells or generalised coordinates.
- Not needed: MITgcm and POP have run global ocean for decades on
  partial cells with their own (non-d-J) PGF formulations.

For 2.5°–1° global ocean with idealized → OMIP forcing, partial cells
alone is sufficient.  d-J PGF is a worthwhile add-on later if we ever
push to ≤1/4° eddy-resolving simulations where slope-current detail
matters.

## Strategic positioning

This plan is **infrastructure** that benefits every ocean grid in
legoESM, not just lat-lon C-grid.  Once partial cells lands:

- The realistic-geometry plan (currently paused at Phase 3.5) resumes.
- Future tripole work (`tripole_grid_plan.md`) inherits partial cells
  for free.
- Cubed-sphere C-grid ocean (`cubed_sphere_cgrid_ocean_plan.md`) —
  whenever that ships — gets partial cells without re-design.

In other words, partial cells is a **prerequisite for production-class
realistic-geometry runs** at every grid type.

## Differentiability contract

End-to-end `jax.grad` compatibility is a project-wide design goal
(per CLAUDE.md).  Partial cells **preserves AD for all current and
foreseeable workflows**:

| Workflow | AD status |
|---|---|
| Training NN parameterizations on partial-cells ocean | ✅ Preserved |
| Sensitivity of solutions to initial T/S, u, eta | ✅ Preserved |
| Sensitivity to forcing parameters (wind, restoring, drag, mixing) | ✅ Preserved |
| Sensitivity to physical-scheme tuning constants | ✅ Preserved |
| MPI-distributed AD (via `_sendrecv_vjp`) | ✅ Preserved |
| **Gradient w.r.t. bathymetry shape** (`H_bathy`) | ⚠️ **Fundamentally limited** |

**Why most workflows are preserved**:

- ``h_partial(i, j, k)`` is **static** once the coordinate is built.
  It's a constant input to all traced operations during stepping —
  identical role to ``dz_ref`` today.  No retracing, no `jit` issues.
- All partial-cell-aware operators use ``jnp.where``, ``jnp.sum``,
  ``jnp.maximum`` — all JAX-traceable.  No Python control flow on
  traced values.
- The PGF special-case at partial-cell boundaries is implemented as
  ``jnp.where(is_partial_boundary, corrected_pgf, standard_pgf)`` with
  a static boundary mask: both branches trace, no AD discontinuity.
- Cells below ``bottom_level`` masked via ``jnp.where(is_active, value,
  0.0)`` — the same pattern already used for land masking.

**The one limitation: gradient w.r.t. bathymetry**

``bottom_level(i, j) = first k where z_half_ref[k+1] < -H_bathy(i, j)``
is a **discrete index** with step transitions as ``H_bathy`` varies
continuously.  When ``H_bathy`` crosses a reference-level interface,
``bottom_level`` jumps by 1 and the column gains/loses an active cell.

Practical implication: gradient of a loss w.r.t. ``H_bathy`` has a
*non-smooth* component at every reference-level interface.  JAX will
silently compute some gradient (zero through the integer cast), but
it won't be physically meaningful for bathymetry inference.

**This is not legoESM-specific** — every partial-cell ocean model has
the same limitation.  MOM6, MITgcm, POP/CESM all treat ``H_bathy`` as
fixed input data; they don't infer it via gradient descent.  Smooth
differentiable approximations exist (sigmoid-blended thicknesses, as
in some adjoint ECCO setups) but are out of scope for this plan.

**Explicit non-goals** of the partial-cells port:

- Differentiable bathymetry inference from observations
- Sensitivity studies w.r.t. continental-margin shape
- Bayesian inversion of seafloor topography

If any of these become research priorities later, they would be a
separate workstream on top of partial cells (sigmoid-blended thickness
or differentiable-bathy reformulation).

## Scientific contract

After partial cells ships, the model must demonstrate:

1. **z\* fallback bit-exact.**  Any existing experiment configured with
   the legacy `OceanZStarCoordinate` produces *byte-for-byte identical*
   output to the pre-partial-cells code path.  Full-cells-everywhere is
   a special case of partial cells (where every partial thickness
   equals the reference thickness).
2. **Column thickness identity.**  For any (eta, H_bathy) configuration,
   `sum_k h_k(i, j) = eta(i, j) + H_bathy(i, j)` per cell to round-off.
3. **Rest state on real ETOPO at 2.5°.**  30-day rest-state stress test
   (the Phase 3a setup, but with realistic ETOPO geometry) holds
   `max|u| ≤ 5 mm/s`.  This is the headline Phase 3a re-test.
4. **Beckmann-Haidvogel seamount.**  The same regime-boundary sweep
   from `phase3a_seamount_smooth*` should now pass at the original
   B-H height (3800 m, r_max ≈ 0.5) — i.e., the model becomes much
   less sensitive to bathymetry roughness because the PGF problem
   is confined to the partial bottom cell.
5. **Hydrostatic balance preserved.**  Pressure at any level integrates
   correctly through partial cells (each layer contributes
   `ρ · g · h_k`, with the partial bottom layer using its actual
   thickness).
6. **Tracer conservation.**  Closed-domain heat and salt totals
   invariant to round-off across a 1-year integration on real ETOPO
   bathymetry under no-forcing.
7. **Realistic geometry 50-yr global overturning** (resumed from
   realistic-geometry plan Phase 4): completes cleanly with credible
   Drake transport and AMOC pathway.

## Phasing

### Phase 0 — Scoping, data structure, and partial-cell mask  (≈ 1 week)

- New ``OceanPartialCellCoordinate`` NamedTuple alongside
  ``OceanZStarCoordinate``:

  ```python
  class OceanPartialCellCoordinate(NamedTuple):
      n_levels: int
      H_max: float
      z_full_ref: jnp.ndarray         # (nlev,)  reference cell-centre depths
      z_half_ref: jnp.ndarray         # (nlev+1,) reference interface depths
      dz_ref: jnp.ndarray             # (nlev,)  reference layer thicknesses
      h_partial: jnp.ndarray          # (nlat, nlon, nlev)  per-cell layer thickness
      bottom_level: jnp.ndarray       # (nlat, nlon)  index of deepest active level
      is_partial: jnp.ndarray         # (nlat, nlon, nlev)  mask of cells partial vs full
  ```
- Factory `create_partial_cell_coordinate(z_coord, H_bathy)`:
  - For each column (i, j), find the deepest reference level whose
    interface depth is shallower than H_bathy(i,j).
  - Set h_partial[i, j, :bottom] = dz_ref[:bottom] (full cells).
  - Set h_partial[i, j, bottom] = H_bathy(i, j) − abs(z_half_ref[bottom]).
  - Set h_partial[i, j, bottom+1:] = 0 (below seafloor; treated as land).
  - Note: when ``H_bathy(i, j) >= H_max``, all cells are full
    (degenerate to the legacy path).
- Per-cell ``ocean_mask_3d``: True where the cell has any water
  (``h_partial > 0``).  This is the 3D version of the 2D land mask.
- Tests:
  - ``test_partial_cell_construction``: H_bathy = const → matches
    legacy OceanZStarCoordinate.
  - ``test_partial_cell_thickness_sum``: sum h_partial per cell =
    H_bathy.
  - ``test_partial_cell_at_step_bathy``: step bathymetry produces
    the expected ragged bottom_level array.

**Decision gate**: factory passes the construction tests; legacy
flat-bottom path is bit-exact recovered.

### Phase 1 — Layer-thickness-aware operators  (≈ 1 week)

The current ``compute_layer_thickness(eta, H_bathy, z_coord)`` returns
a uniform Jacobian-scaled thickness per cell.  Partial cells replace
this with a per-cell thickness array that is also eta-aware:

```python
h_k(i, j, k) = h_partial(i, j, k) * (eta(i, j) + H_bathy(i, j)) / H_bathy(i, j)
```

This preserves column-thickness conservation (sum over k = eta + H)
while keeping full cells full and the partial cell partial.

- Update ``compute_layer_thickness``, ``compute_ocean_jacobian``,
  ``compute_depth_mean``, and the column integration helpers in
  ``vertical.py``.
- Add a partial-cell-aware ``layer_centroid_depth(coord, eta)`` that
  returns z at each cell centre (for hydrostatic pressure integration).
- Tests:
  - All flat-bottom unit tests bit-exact regress.
  - Column thickness identity holds for arbitrary H_bathy + eta.
  - Centroid depth is monotonically deeper with k.

**Decision gate**: bit-exact regression on flat-bottom + new
identity tests pass.

### Phase 2 — Hydrostatic pressure on partial cells  (≈ 5 days)

The hydrostatic pressure at full level k is computed as:

```
p[k] = p_atm + integral from surface down to z_centre[k] of ρ · g · dz
     = sum over k' < k of (ρ[k'] · g · h[k']) + 0.5 · ρ[k] · g · h[k]
```

This formula is already correct *if* h[k] is the actual thickness.
The change is:
- Use ``h_partial`` (per-cell, partial-aware) instead of
  ``dz_ref * J`` (uniform).
- Bottom-cell contribution uses the partial thickness automatically.

- Update ``compute_hydrostatic_pressure`` in ``ocean/eos.py``.
- Update ``compute_ocean_rho_and_pressure`` in ``ocean/eos.py``.
- Tests:
  - Hydrostatic balance at rest state: ``∂p/∂z = -ρg`` to round-off
    across all layers including the partial bottom.
  - Comparison vs analytical: pressure at any depth equals
    ``ρ(z) · g · (depth from surface)`` for an isopycnal column.

**Decision gate**: hydrostatic identity passes on a step-bathymetry
test geometry.

### Phase 3 — PGF special-case for partial bottom cell  (≈ 1.5 weeks) — **CRITICAL**

This is the heart of partial-cell PGF:

The standard horizontal pressure gradient at level k (full cells):

```
∂p/∂x[k] = (p[k, i+1] - p[k, i]) / dx
```

This works for full cells (z_k constant horizontally).  At the
partial-cell boundary, where one column's bottom is at z=−2350 and
the neighbour's bottom is at z=−4000, the depths of the cell centres
differ.  The "level k" in the two columns is at different z.

The Adcroft & Campin 2004 fix:

1. Identify pairs (i, i+1) where one is a partial cell and the other
   is a full cell at the same level k.
2. For the PGF at the U-face between them, use the depth of the
   *shallower* cell's bottom interface as the reference depth.
3. Evaluate ρ from the deeper cell at this same reference depth via
   linear interpolation (or the deeper cell's value at the partial-
   cell-boundary's depth).
4. Compute the PGF using these ρ values at the consistent reference depth.

This is the special-case that turns "the partial cell still has a PGF
problem" into "the partial cell is handled correctly".  Without this
Phase 3, the partial-cells implementation will fail on the seamount
stress test.

- New helper ``_partial_cell_pgf_correction(rho, p, z_half, h_partial,
  ...)`` in ``latlon_cgrid_operators.py``.
- Modify the C-grid pressure-gradient computation in
  ``ocean_pe_latlon_cgrid.py`` to apply the correction at the partial
  cell.
- Tests:
  - ``test_partial_cell_pgf_step_bathy``: rest state on a step bathy
    has machine-zero PGF after the correction.
  - ``test_partial_cell_pgf_seamount``: B-H seamount at r_max=0.5
    integrates 30 days with max|u| < 5 mm/s — the headline test.
- Validation: re-run ``run_phase3a_seamount.py`` sweep with partial
  cells active.  Expect: pass at r_max ≈ 0.5–0.7 (vs the current
  0.24 limit).

**Decision gate**: partial-cell PGF passes Phase 3a's seamount stress
test at r_max ≥ 0.5 (matching MITgcm/MOM6 published bounds).

### Phase 4 — Tracer flux divergence on partial cells  (≈ 1 week)

The tracer transport equation
``∂(h·T)/∂t = -∂(h·u·T)/∂x - ∂(h·v·T)/∂y - ∂(w·T)/∂z`` uses per-cell
layer thickness `h_k`.  With partial cells:

- Vertical flux at the bottom interface (between partial cell and
  the seafloor) must be exactly zero.  The current
  ``flux_form_vertical_tracer_advection`` uses
  ``F[..., nlev] = 0`` by construction — already correct for
  partial cells.
- Horizontal flux at the partial-cell-to-deeper-cell boundary needs
  to use the *thinner* of the two adjacent cells (the partial cell
  is the constraint).
- Tracer-flux operators update consistently with ``h_partial``.

- Update ``flux_form_vertical_tracer_advection`` to handle the
  partial-thickness correctly (mostly a no-op since boundary fluxes
  are already zero).
- Update horizontal tracer flux computations to use the per-cell
  thickness array.
- Update SOM, DST-3, WENO advection schemes (they access ``h_k``
  via the same path).
- Tests:
  - Closed-domain tracer conservation on real-ETOPO bathymetry over
    1 year.
  - Bottom flux through partial cell = 0 to round-off.

**Decision gate**: 1-year closed-domain conservation < 1e-9 relative.

### Phase 5 — Vertical velocity and momentum advection  (≈ 1 week)

Diagnose w from continuity at partial cells:

- ``diagnose_w_from_flux_div`` already uses cumulative sum of horizontal
  flux divergence, which is independent of vertical thickness — this
  path may need only a per-cell ``h_partial`` weighting in the
  ``thickness_weighted=True`` case.
- ``vertical_advection_ocean`` and the flux-form vertical momentum
  advection use ``z_coord.dz_half_ref * jacobian`` for the upwind
  gradient.  Replace with partial-cell-aware ``dz_half`` per cell.
- Bottom boundary: w must be zero at the seafloor partial-cell
  interface.  This is enforced by the cumsum from below.
- Tests:
  - w at surface and seafloor equals 0 to round-off.
  - Vertical momentum advection at rest state has zero tendency.

**Decision gate**: w is correctly zero at the seafloor in a rest
state on step bathymetry.

### Phase 6 — Integration into LatLonCGridOceanModel  (≈ 1 week)

- ``LatLonCGridOceanState`` adds an ``h_partial`` field (per-cell layer
  thicknesses).
- ``rest_state_latlon_cgrid_ocean`` and friends populate it from
  ``H_bathy`` via ``create_partial_cell_coordinate``.
- ``LatLonCGridOceanModel.step`` threads ``h_partial`` through every
  operator that needs it.
- Backwards compatibility: a config flag
  ``use_partial_cells: bool = False`` (default False initially)
  selects the new path; when False, the legacy uniform-Jacobian path
  is used and results are bit-exact regressed.  Once Phase 6 ships
  and Phase 7 validates, the default can be flipped to True.
- Tests:
  - All existing flat-bottom tests bit-exact under
    ``use_partial_cells=False``.
  - New tests under ``use_partial_cells=True`` verify the partial-cell
    paths of every operator.

- **AD bit-exact regression** (per the differentiability contract):
  ``eqx.filter_value_and_grad`` of a representative loss function
  (e.g. mean SST after a 24-h integration) on a flat-bottom run
  produces *identical* gradient values before vs after the
  partial-cells port, when ``use_partial_cells=False``.  Catches any
  subtle pytree / JIT / VJP regression introduced during the
  refactor.  ~30-line test in
  ``tests/ocean/unit/test_partial_cells_ad_regression.py``.

**Decision gate**: legacy flat-bottom regression remains bit-exact
(forward AND adjoint); new partial-cell path is exercised by every
operator at least once.

### Phase 7 — Validation suite + documentation  (≈ 1.5 weeks)

- **Re-run Phase 3a seamount sweep with partial cells active.**
  Confirm the regime boundary moves from r ≈ 0.24 → r ≈ 0.5+.
- **Re-run real-ETOPO 30-day stress test with partial cells.**
  Should pass at r_max = 0.9 (no MEO needed).  Headline result.
- **Re-run flat-bottom 50-yr global overturning** under partial cells.
  Bit-exact regression vs the existing reference (since flat bottom
  collapses to the legacy path).
- **Run realistic-ETOPO 50-yr global overturning** (this is the
  resumption of the realistic-geometry plan's Phase 4).  Diagnostics
  per `realistic_geometry_lat_lon_plan.md` Phase 5.
- **Documentation**:
  - `docs/ocean_experiments/partial_cells_results.md` — the same
    structure as `realistic_geometry_phase3a_results.md`, reporting
    the post-port regime boundary, conservation diagnostics, and
    real-ETOPO 50-yr outcome.
  - Update `docs/ocean_grid_staggering.md` with the partial-cell
    treatment.
- **Auditing**:
  - Make ``use_partial_cells=True`` the default once flat-bottom
    bit-exact regression is verified.
  - Update CLAUDE.md with the new vertical-coordinate machinery
    notes.

**Decision gate**: 30-day rest-state on real ETOPO passes
``max|u| < 5 mm/s`` without MEO.  50-yr realistic-geometry global
overturning produces a credible Drake transport.

## Validation hierarchy

In ascending order of difficulty:

| Test | What it catches | Phase |
|---|---|---|
| Flat-bottom OceanPartialCellCoord = OceanZStarCoord | Phase 0 factory bug | 0 |
| Column thickness identity per cell | Phase 0 / 1 thickness bug | 0–1 |
| Hydrostatic balance at rest on step bathy | Phase 2 hydrostatic bug | 2 |
| Step-bathymetry PGF rest state machine-zero | Phase 3 PGF correction bug | 3 |
| Beckmann-Haidvogel seamount r=0.5 passes | Phase 3 PGF correction validation | 3 |
| Closed-domain 1-yr tracer conservation on real ETOPO | Phase 4 flux-div bug | 4 |
| w = 0 at seafloor on step bathy | Phase 5 w-diagnosis bug | 5 |
| All flat-bottom tests bit-exact under partial-cells path | Phase 6 integration bug | 6 |
| Real-ETOPO 30-day rest state max\|u\| < 5 mm/s | end-to-end production readiness | 7 |
| Realistic-ETOPO 50-yr global overturning | scientific final-product readiness | 7 |

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Phase 3 PGF correction is harder than estimated (Adcroft & Campin 2004 has subtle implementation details for staggered grids) | **High** | Extra 1 week budget; reference MITgcm + MOM6 source code as ground truth |
| Backwards-compat regression fails subtly under partial cells (flat-bottom not bit-exact) | High | Phase 6 explicitly gates on bit-exact; if it fails, isolate the floating-point ordering difference and patch |
| Partial-cell coordinate increases JIT compile time substantially | Medium | Per-cell thickness arrays are static-shape; should JIT identically.  Profile and tune if needed |
| Vertical advection at partial-cell boundary develops new instabilities | Medium | Phase 5 tests target this; if so, switch to flux-form momentum advection (we already have it) |
| KPP / convection schemes assume uniform layer thickness | Medium | They already use per-cell h_k from `compute_layer_thickness`; should work transparently after Phase 1 |
| MPI halo exchange of `h_partial` adds bandwidth | Low | h_partial is static once H_bathy is set; doesn't change per step.  No halo cost. |
| Existing GM/Redi triads + centered schemes assume uniform thickness | Medium | Phase 4 audits this; if needed, thread h_partial through the GM/Redi flux divergence |
| Integration test wall-time regressions | Low | Per-cell thickness adds a small constant overhead; profile before vs after |

## Reuse map

What stays unchanged:

| Component | Reuse | Notes |
|---|---|---|
| EOS (Wright, linear) | 100% | column-local |
| KPP, Richardson, constant vertical mixing | 100% | already use per-cell h_k |
| Convection (enhanced_diffusion) | 100% | column-local |
| Surface forcing | 100% | per-cell |
| Time integration | 100% | grid-agnostic |
| Restart I/O | ~95% | h_partial restart field added |
| Channel packing for ML | 100% | shape-preserving |
| Conservation diagnostics | ~95% | already metric-aware; thickness comes from coord |
| GM/Redi (centered + triads) | ~85% | flux divergence audit needed |
| All advection schemes | ~80% | thickness propagation through SOM/DST-3/WENO needs verification |
| `LatLonCGridOceanConfig` | 100% | adds `use_partial_cells` flag |
| `latlon_cgrid_operators.py` | ~70% | PGF correction + thickness threading |

What is genuinely new:

| Component | LOC estimate |
|---|---|
| `OceanPartialCellCoordinate` + factory | ~250 |
| Layer-thickness-aware operator updates | ~300 |
| Hydrostatic pressure partial-cell update | ~80 |
| Partial-cell PGF correction | ~250 |
| Tracer-flux partial-cell threading | ~150 |
| w-diagnosis + vertical advection update | ~100 |
| Model integration + state field | ~100 |
| Validation suite | ~600 |
| Diagnostic scripts (re-run B-H seamount, real-ETOPO) | ~200 |

**Total: ~1900–2200 LOC of new + modified code.**

## Calendar estimate

| Phase | Effort | Cumulative |
|---|---|---|
| 0 — Coordinate + factory | 1 week | 5 d |
| 1 — Layer-thickness-aware operators | 1 week | 10 d |
| 2 — Hydrostatic pressure | 5 days | 15 d |
| 3 — PGF special case (CRITICAL) | 1.5 weeks | 22 d |
| 4 — Tracer flux divergence | 1 week | 27 d |
| 5 — w + vertical advection | 1 week | 32 d |
| 6 — Model integration | 1 week | 37 d |
| 7 — Validation + documentation | 1.5 weeks | 45 d |

**Total: ~9 weeks** of focused work + some compute time for the
50-yr realistic-geometry validation.

If Phase 3 takes longer than budgeted (Adcroft & Campin 2004 has
implementation subtleties), add 1–2 weeks.

## Reference material

- **Adcroft, A., Hill, C., and Marshall, J. (1997)** — "Representation
  of topography by shaved cells in a height coordinate ocean model."
  *Mon. Wea. Rev.*, **125**, 2293–2315.  The original partial-cells
  paper.
- **Adcroft, A., and Campin, J.-M. (2004)** — "Rescaled height
  coordinates for accurate representation of free-surface flows in
  ocean circulation models."  *Ocean Modelling*, **7**, 269–284.
  z\* + partial cells combined.  *The* reference for this plan.
- **Pacanowski, R. C., and Gnanadesikan, A. (1998)** — "Transient
  response in a z-level ocean model that resolves topography with
  partial cells."  *Mon. Wea. Rev.*, **126**, 3248–3270.  The POP /
  CESM partial-cell formulation.
- **Griffies, S. M., et al. (2012)** — "Coordinated Ocean-ice Reference
  Experiments (COREs)."  *Ocean Modelling*, **51**, 76–95.  CMIP-class
  ocean model best practice including partial cells.
- **MOM6 source code** — `MOM_continuity.F90`,
  `MOM_horizontal_pressure_force.F90` for the modern z\* + partial
  cell PGF treatment.
- **MITgcm source code** — `model/src/diags_phys.F`, `pkg/seaice/...`
  for the original implementation.
- `docs/ocean_experiments/realistic_geometry_phase3a_results.md` —
  empirical regime boundary that motivated this plan.

## Open questions for decision before Phase 0

1. **Default value of ``use_partial_cells``.**  Recommendation: start
   ``False`` to preserve all existing experiments bit-exact during
   Phases 0–6.  Flip to ``True`` at Phase 7 once the 30-day real-ETOPO
   stress test passes.
2. **Precision policy.**  ``z_coord.dz_ref`` is built at storage policy
   precision (~float32) per Phase 1's earlier finding.  Should
   ``h_partial`` be float64 throughout to match production-grade
   precision targets?  Recommendation: yes — column thickness identity
   is a critical invariant; float32 introduces ~1e-7 relative error
   (acceptable for short runs, marginal for 50-yr conservation).
3. **MPAS partial cells.**  Out of scope for this plan; the partial-
   cell coordinate is grid-agnostic but the PGF correction is grid-
   specific.  Once lat-lon C-grid lands, MPAS partial cells is a
   separate, smaller plan (~500 LOC, 2 weeks).  Recommendation:
   document this scope boundary clearly.
4. **Restart compatibility.**  Existing restart files (e.g. our 50-yr
   global overturning runs) don't have ``h_partial``.  Should the
   loader auto-construct ``h_partial`` from ``H_bathy`` when missing,
   or require migration?  Recommendation: auto-construct at load time
   so existing restarts continue to work transparently.
