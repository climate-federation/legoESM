# GM/Redi Implementation Plan for Lat-Lon C-Grid Ocean

*Created: 2026-04-25*

## Context

**Goal**: Implement Gent-McWilliams / Redi (GM/Redi) mesoscale eddy parameterization
for the lat-lon Arakawa C-grid ocean, targeting the 1-degree global overturning
circulation experiment.

**Motivation**: At 1-degree resolution mesoscale eddies are unresolved. Without
GM/Redi, the overturning and stratification are maintained purely by explicit
diffusion, producing an unrealistically diffuse thermocline and excessive ACC
transport. GM parameterizes the adiabatic flattening of isopycnals by unresolved
eddies; Redi rotates tracer diffusion along (not across) isopycnal surfaces.

**Starting point**: A cubed-sphere GM/Redi exists (`gm_redi.py`, 16 tests passing)
but depends on CS-specific operators and cannot be reused for the lat-lon C-grid.
The lat-lon call site in `ocean_model_latlon_cgrid.py:526-536` expects a specific
API that does not yet exist.

**Research basis**: Literature review of Griffies (1998), Gent (2011), Ferrari et
al. (2010), Visbeck et al. (1997), Lemarié et al. (2012), and survey of MOM6,
MITgcm, NEMO, and POP2 implementations.

**Expert reviews**: Reviewed by ocean-expert (physics/validation), dycore-expert
(staggering/conservation/stability), and modularity-tester (architecture/reuse).

---

## Design Decisions

### 1. Formulation: Griffies (1998) skew-flux

Combined GM (antisymmetric) + Redi (symmetric) small-slope tensor. The skew-flux
form avoids computing a bolus velocity and its divergence-free constraint. On a
C-grid, horizontal fluxes live naturally at u/v-faces. All major z-coordinate
models (MITgcm, NEMO, POP2) use this as default. MOM6's thickness-diffusion
approach doesn't map to z-star coordinates.

### 2. Slope computation: Centered differences (not triads)

Compute drho/dx at u-faces via `gradient_x_cgrid`, average to cell centers, then
to vertical interfaces. Vertical gradient at interfaces using `dz_half_ref * J`.
The triad approach (Griffies et al. 1998, NEMO) gives exact conservation but is
complex; at 1-degree other error sources dominate. Upgrade path: isolate slope
computation in its own function.

### 3. Tapering: DM95 with S_max = 0.005

Smooth tanh tapering (Danabasoglu & McWilliams 1995):
```
taper = 0.5 * (1 + tanh((S_max - |S|) / (0.1 * S_max)))
```

Infinitely differentiable (critical for JAX autodiff). S_max = 0.005 (not 0.01)
for vertical CFL safety: at dz = 10m, kappa = 1000, dt = 300s, the vertical
stability number is 0.075 (well within the 0.5 limit). Physical slopes in the
1-deg Southern Ocean are typically 0.001-0.003.

### 4. Boundary-layer treatment: Deferred

Ferrari et al. (2010) BVP streamfunction deferred — matters mainly for
eddy-permitting models. DM95 tapering handles the mixed layer adequately at
1-degree by reducing GM where stratification is weak.

### 5. Adaptive coefficient: Visbeck (1997)

Reuse existing `compute_visbeck_kappa_gm` (grid-agnostic). kappa range
[200, 2000] m^2/s. Captures latitude dependence of eddy activity.

### 6. Land mask: Neumann fill + face masks

Following existing `ocean_pe_latlon_cgrid.py` pattern:
1. Neumann-fill density AND tracer fields before computing gradients
2. Mask horizontal fluxes with `u_mask`, `v_mask`
3. Mask final tendency with cell `mask`

### 7. AD safety

- DM95 tanh: smooth everywhere
- `jnp.minimum(drho_dz, -eps)`: zero gradient in convective regions (correct)
- `sqrt(S^2 + eps)`: eps = 1e-30 (matching Visbeck lesson)
- Division guard: eps = 1e-10 for `drho_dz_safe`
- `jnp.clip` on Visbeck kappa: zero gradient at bounds (physically correct)

---

## Tensor Formulation (Griffies 1998)

For a tracer q, the combined GM+Redi small-slope tensor gives:

**Horizontal fluxes** (at C-grid faces):
```
F_x = kappa_Redi * dq/dx + (kappa_Redi - kappa_GM) * S_x * dq/dz
F_y = kappa_Redi * dq/dy + (kappa_Redi - kappa_GM) * S_y * dq/dz
```

**Vertical flux** (at interfaces):
```
F_z = (kappa_Redi + kappa_GM) * (S_x * dq/dx + S_y * dq/dy)
    + kappa_Redi * (S_x^2 + S_y^2) * dq/dz
```

When kappa_GM = kappa_Redi (common default), the off-diagonal horizontal terms
cancel: F_x = kappa * dq/dx, simplifying to Laplacian diffusion + enhanced
vertical mixing proportional to S^2.

**Tendency**: dq/dt = div_h(F_x, F_y) + dF_z/dz

---

## Implementation Phases

### Phase 1: Extract shared code (`_gm_redi_common.py`)

Factor grid-agnostic helpers from `gm_redi.py`:
- `compute_visbeck_kappa_gm()` — adaptive coefficient
- `dm95_taper()` — identical tapering across grids
- `vertical_flux_divergence()` — zero-pad BCs + FV divergence

Update CS `gm_redi.py` to import from common. Verify existing 16 tests pass.

### Phase 2: Core implementation (`gm_redi_latlon_cgrid.py`)

Three public functions:
- `compute_isopycnal_slopes_latlon_cgrid(rho, mask, z_coord, J, grid, cfg)`
  → (S_x, S_y, taper) at interfaces
- `gm_redi_tracer_tendency_latlon_cgrid(q, S_x, S_y, masks, z_coord, J, grid, kappa_GM, kappa_Redi)`
  → dq/dt at cell centers
- `gm_redi_tracer_tendency_latlon(T, S, eta, H_bathy, grid, z_coord, cfg, ...)`
  → (dT_dt, dS_dt) — top-level orchestrator

Key implementation details:
- Use 3D-native `gradient_x_cgrid` / `gradient_y_cgrid` (no vmap)
- Combine diagonal + off-diagonal fluxes at faces, single `divergence_cgrid` call
- Neumann-fill BOTH density and tracer before gradient computation
- Also provide `LateralMixingOutput`-returning wrapper for future factory use

### Phase 3: Tests (13 categories + structural enforcement)

Unit tests: shape, finiteness, uniform-tracer zero tendency, uniform-density zero
slopes, tracer conservation, variance reduction, isopycnal-aligned tracer zero
tendency, land mask correctness, zonal symmetry, sign check, Visbeck coefficient,
JAX differentiability, vertical CFL diagnostic, LateralMixingOutput wrapper.

Structural test: verify shared imports, no inline Visbeck formula duplication.

### Phase 4: Wire into ocean model

- Update import in `ocean_model_latlon_cgrid.py:527`
- Pass mask, u_mask, v_mask, f_coriolis
- Set K_h = 0 for tracers when GM/Redi active (avoid double diffusion)
- Add clear TypeError in `integration.py` for lat-lon grid

### Phase 5: Global overturning experiment

- Add `use_gm_redi: bool = False` toggle to `GlobalOverturningConfig`
- Configure: kappa_GM = kappa_Redi = 1000, S_max = 0.005, Visbeck enabled
- Run 10-year integration, check stability, conservation, APE reduction

---

## Files

| File | Action |
|------|--------|
| `src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py` | **CREATE** |
| `src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py` | **CREATE** |
| `src/legoesm/ocean/physics/lateral_mixing/gm_redi.py` | **MODIFY** |
| `src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` | **MODIFY** |
| `src/legoesm/ocean/physics/lateral_mixing/integration.py` | **MODIFY** |
| `src/legoesm/ocean/experiments/global_overturning.py` | **MODIFY** |
| `tests/ocean/unit/test_gm_redi_latlon_cgrid.py` | **CREATE** |
| `tests/ocean/unit/test_no_gm_redi_duplication.py` | **CREATE** |

---

## Expected Physical Impact

At 1-degree in the global overturning experiment, GM/Redi should:
- **Sharpen the thermocline** (reduced diffusive spreading)
- **Reduce Eulerian MOC strength** by 20-40% (GM opposes wind-driven overturning)
- **Reduce ACC transport** by 30-50% (flattened isopycnal slopes)
- **Reduce APE** monotonically (primary diagnostic of correctness)
- **Reduce equator-to-pole SST gradient** (enhanced poleward heat transport)

**Failure modes to watch for**:
- kappa too large → thermocline destruction, nearly uniform SST
- Sign error → isopycnal steepening, runaway instability (APE increases)
- Land mask error → dipoles at coastlines
- Vertical CFL blow-up �� exponentially growing T oscillations

---

## Surface-Layer Mixing Behavior

The current implementation tapers the **slopes** (S_x, S_y) via DM95 but not
the Redi diffusivity itself. This means:

- **Interior (taper ≈ 1):** Full isopycnal mixing — adiabatic by design.
- **Surface (taper → 0):** The off-diagonal rotation terms vanish, but the
  diagonal `kappa_Redi * nabla^2(q)` persists at full strength. This is pure
  horizontal diffusion — **cross-isopycnal** wherever isopycnals slope.

In practice the mixed layer has weak stratification, so "along isopycnal" is
ill-defined there anyway. The residual horizontal diffusion effectively acts
as a background K_h in the surface layer, which is needed for numerical
stability. But `kappa_Redi = 1000 m^2/s` may be too large for that role.

### Options (future)

**Option A — taper the full tensor (quick fix):** Multiply the entire tendency
(diagonal + off-diagonal + vertical) by the taper factor. In fully tapered
regions GM/Redi produces zero tendency, and the existing K_v + convective
adjustment handle the surface. This matches MITgcm's DM95 implementation.

**Option B — Ferrari et al. (2010) BVP (correct fix):** Solve a vertical
elliptic problem for the GM streamfunction that smoothly transitions from
depth-independent (horizontal) transport in the mixed layer to adiabatic
interior transport. Eliminates the need for tapering near boundaries entirely.

**Current choice:** Proceed with the slope-only tapering. At 1-degree with
20 levels the surface layer is maintained by surface restoring + convective
adjustment, so the residual horizontal diffusion from tapered Redi is
unlikely to dominate the solution. Revisit if SST shows excessive smoothing
or if conservation diagnostics flag diapycnal drift.

---

## Known Gaps (deferred)

- **Surface-layer cross-isopycnal mixing** (see above) — Option A or B
- Factory unification (lat-lon bypasses `integration.py`)
- Config pathway unification (`LatLonCGridOceanConfig.gm_redi` vs `LateralMixingConfig`)
- Ferrari et al. (2010) BVP boundary-layer treatment
- Triad slope computation for exact conservation
- MPAS GM/Redi (no lateral mixing on Voronoi grid yet)
- Implicit vertical diffusion for the kappa * S^2 term

---

## References

- Gent, P.R. & McWilliams, J.C. (1990). Isopycnal mixing in ocean circulation
  models. *J. Phys. Oceanogr.*, 20, 150-155.
- Redi, M.H. (1982). Oceanic isopycnal mixing by coordinate rotation. *J. Phys.
  Oceanogr.*, 12, 1154-1158.
- Griffies, S.M. (1998). The Gent-McWilliams skew flux. *J. Phys. Oceanogr.*,
  28, 831-841.
- Griffies, S.M. et al. (1998). Isoneutral diffusion in a z-coordinate ocean
  model. *J. Phys. Oceanogr.*, 28, 805-830.
- Danabasoglu, G. & McWilliams, J.C. (1995). Sensitivity of the global ocean
  circulation to parameterizations of mesoscale tracer transports. *J. Climate*,
  8, 2967-2987.
- Visbeck, M. et al. (1997). Specification of eddy transfer coefficients in
  coarse-resolution ocean circulation models. *J. Phys. Oceanogr.*, 27, 381-402.
- Ferrari, R. et al. (2010). A boundary-value problem for the parameterized
  mesoscale eddy transport. *Ocean Modelling*, 35, 245-265.
- Gent, P.R. (2011). The Gent-McWilliams parameterization: 20/20 hindsight.
  *Ocean Modelling*, 39, 2-9.
- Lemarié, F. et al. (2012). On the stability and accuracy of the harmonic and
  biharmonic isoneutral mixing operators. *Ocean Modelling*, 52-53, 9-35.
