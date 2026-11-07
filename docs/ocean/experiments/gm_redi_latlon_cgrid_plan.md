# GM/Redi Implementation Plan for Lat-Lon C-Grid Ocean

*Created: 2026-04-25.  Updated: 2026-04-26 (Phase 6 — triad slope discretisation).*

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
The lat-lon call site in `ocean_model_latlon_cgrid.py:527-537` expects a specific
API that does not yet exist.

**Research basis**: Literature review of Griffies (1998), Gent (2011), Ferrari et
al. (2010), Visbeck et al. (1997), Lemarié et al. (2012), and survey of MOM6,
MITgcm, NEMO, and POP2 implementations.

**Expert reviews**: Reviewed by ocean-expert (physics/validation), dycore-expert
(staggering/conservation/stability), and modularity-tester (architecture/reuse).

---

## Current Status (2026-04-26)

### Implemented (Phases 1-6 complete)

- `_gm_redi_common.py` — shared Visbeck, DM95 taper, vertical flux divergence
- `gm_redi_latlon_cgrid.py` — slopes, tracer tendency (centered), tracer
  tendency (triads), orchestrator with `slope_scheme` dispatch
- `gm_redi.py` — CS version refactored to import from common
- `ocean_model_latlon_cgrid.py` — import fixed, masks passed
- `integration.py` — TypeError guard for lat-lon
- `global_overturning.py` — `use_gm_redi` toggle + `create_gm_redi_config()`
- 76 unit tests passing (16 CS + 28 lat-lon + 11 Eady physics + 16 Visbeck +
  5 corrections)
- 6 Eady GM/Redi test cases in ocean test matrix (3 centered + 3 triads)

### Validated

- **GM adiabatic flattening** (centered + triads): Confirmed in Eady uniform
  test — GM-only produces nonzero tendency, reduces APE, conserves tracer
  integral.
- **Redi isopycnal diffusion (centered)**: Single-step tendency is small (~1e-11
  K/s for the 30-day Eady physics integration), 1400× improvement over the
  initial implementation via the interface-level flux fix.
- **Redi isopycnal diffusion (triads)**: Per-triad cancellation is *algebraic*,
  so the residual is at the float64 round-off scale.  At kappa_Redi = 1×10⁶ the
  residual at the worst point is 4×10⁻¹³ K/s on the 12×6 Eady grid (compared to
  1.2×10⁻⁴ before the Neumann-fill of ρ inside the function).  Per-unit-kappa
  this is ~10⁻¹⁹ K/s — i.e. true machine precision.
- **Differentiability**: `jax.grad` flows through both centered and triad paths.

### Triad scheme: how to use

```python
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig

# Default (centered) — backwards compatible:
cfg = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0)

# Triad scheme — required for century-scale climate runs:
cfg = GMRediConfig(kappa_GM=1000.0, kappa_Redi=1000.0, slope_scheme="triads")
```

The orchestrator dispatches on `cfg.slope_scheme ∈ {"centered", "triads"}`.
Visbeck still runs on centred slopes (only ⟨N|S|⟩_z is needed there).

---

## Design Decisions

### 1. Formulation: Griffies (1998) skew-flux

Combined GM (antisymmetric) + Redi (symmetric) small-slope tensor. The skew-flux
form avoids computing a bolus velocity and its divergence-free constraint. On a
C-grid, horizontal fluxes live naturally at u/v-faces. All major z-coordinate
models (MITgcm, NEMO, POP2) use this as default. MOM6's thickness-diffusion
approach doesn't map to z-star coordinates.

### 2. Slope computation: Centered + triads (both implemented)

Two slope discretisations are available, selected via
`GMRediConfig.slope_scheme`:

- **Centered** (default, backwards compatible).  Compute drho/dx at u-faces via
  `gradient_x_cgrid`, average to cell centers, then to vertical interfaces.
  Vertical gradient at interfaces using `dz_half_ref * J`.  Cheap, but the Redi
  flux for ``q = f(ρ)`` carries a small residual that accumulates over long
  integrations through the pressure→velocity→advection feedback loop.

- **Triads** (Griffies, Gnanadesikan, Pacanowski et al. 1998).  Each face flux
  is the average of four quarter-cell triads.  Each triad uses *the same three*
  density/tracer values for both its slope and its gradients, so the algebraic
  identity ``dq/dx + S_x dq/dz = 0`` (and the analogous z-flux relation) holds
  *per triad* — the Redi tendency is therefore zero to float64 round-off when
  ``q = f(ρ)``.  Required for century-scale climate runs.

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

**Key implementation detail**: ALL horizontal flux terms (diagonal + off-diagonal)
are evaluated at INTERFACE levels before averaging to full levels. This ensures
exact cancellation `dq/dx + S_x * dq/dz = 0` at the interface level when
q = f(rho). See "Redi Cross-Isopycnal Diffusion" section below for why this
matters and its limitations.

---

## Eady Validation Results

### Test design

The Eady uniform experiment provides the ideal test: linear EOS makes isopycnals
equal isotherms, so T is exactly constant along isopycnals. Uniform N², linear
vertical shear, depth-uniform dT/dy. No perturbation seeded — clean background.

Three cases at low resolution (20x10, ~100 km), kappa=50000 m²/s (enhanced for
visibility), 120-day integration, no sponge layer:

| Case | kappa_GM | kappa_Redi | Expected |
|------|----------|------------|----------|
| Baseline | 0 | 0 | Dynamics only |
| GM-only | 50000 | 0 | Adiabatic flattening |
| Redi-only | 0 | 50000 | No change (T ∝ rho) |

### Results

**Single-step Redi tendency diagnostic** (computed on evolving state):
```
Day   0: 0.000007 K/day   (8.7e-11 K/s)
Day   1: 0.000005 K/day
Day  10: 0.000001 K/day
Day  30: 0.000000 K/day   (1.3e-12 K/s)
```
Redi tendency is near-zero and DECREASING — confirming the isopycnal projection.

**GM-only tendency**: 1.75 K/day — 250,000x larger than Redi residual.

**120-day integration (T change from initial)**:
```
Baseline (no GM/Redi):  0.09 K   ← dynamics only
GM-only:                2.21 K   ← GM flattening isopycnals
Redi-only:              1.17 K   ← NOT from Redi tendency (see below)
```

**GM effect** (GM - baseline): 2.27 K — clear adiabatic isopycnal flattening,
visible in T(y,z) cross-sections. Cooling at depth on the warm side, warming
in upper layers.

### The Redi-only divergence puzzle

The Redi-only case diverges from baseline by 1.17 K despite near-zero Redi
tendency. Investigation revealed:

**Not floating-point chaos from JIT**: Comparing `gm_redi=None` vs
`gm_redi=GMRediConfig(kappa_GM=0, kappa_Redi=0)` (different code paths, same
zero tendency) shows **1e-12 K divergence** at 30 days — bit-identical. The
JIT compiler does NOT produce different results based on the code path.

**Actual cause**: Comparing `kappa_Redi=0` vs `kappa_Redi=50000` (same code
path, different tendency) shows 0.41 K divergence at 30 days. The tiny Redi
residual (1e-11 K/s) modifies T → changes rho → changes pressure gradient →
changes velocity → changes advection → modifies T. This pressure-velocity
feedback amplifies the residual by ~500,000x over 30 days.

**Implication**: Even a 1e-11 K/s residual matters over long integrations with
dynamical feedback. The centered-difference approach (even with the interface-
level fix) is NOT sufficient for century-scale climate runs. **Triads are needed.**

### Verification commands

```bash
# Unit tests (41 tests, ~22s)
JAX_ENABLE_X64=1 python -m pytest tests/ocean/unit/test_gm_redi_latlon_cgrid.py \
    tests/ocean/unit/test_gm_redi_eady_physics.py tests/ocean/unit/test_visbeck_gm.py -v

# Ocean test matrix (Eady GM/Redi visual validation)
JAX_ENABLE_X64=1 python scripts/matrix/run_ocean_test_matrix.py --only eady_gm_redi_gm_only
JAX_ENABLE_X64=1 python scripts/matrix/run_ocean_test_matrix.py --only eady_gm_redi_redi_only
JAX_ENABLE_X64=1 python scripts/matrix/run_ocean_test_matrix.py --only eady_gm_redi
```

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

## Redi Cross-Isopycnal Diffusion: Root Cause & Fix Path

### The problem

The Redi isopycnal diffusion tensor should produce ZERO tendency for any
tracer that is constant along isopycnals (e.g., T with linear EOS). In the
continuous case this is guaranteed by the cancellation:

```
dq/dx + S_x * dq/dz = 0    when q = f(rho) and S_x = -(drho/dx)/(drho/dz)
```

In the discrete case, this cancellation fails if the slope `S_x` and the
tracer gradients `dq/dx`, `dq/dz` are evaluated at inconsistent stencil
locations.

### History of fixes

**Initial implementation** (split full-level diagonal + interface off-diagonal):

The diagonal term `kappa_Redi * dq/dx` was evaluated at **full levels** (where
the C-grid gradient naturally lives), while the off-diagonal cancellation term
`(kappa_Redi - kappa_GM) * S_x * dq/dz` was at **interfaces** (where S_x and
dq/dz are defined), then averaged to full levels. The stencil mismatch broke
the cancellation.

- Redi residual: **1.1e-7 K/s** (0.01 K/day at kappa=50000)
- Over 120 days with feedback: **~1.5 K spurious drift**

**Interface-level fix** (current implementation):

Both diagonal AND off-diagonal terms are now evaluated at **interface levels**
before averaging to full levels. At each interface, the cancellation
`dq/dx_half + S_x * dq/dz_half = 0` is exact (both use the same averaging
chain: face → center → interface).

- Redi residual: **8.7e-11 K/s** (0.000007 K/day at kappa=50000) — **1400x reduction**
- But still nonzero: the averaging from interfaces to full levels and then to
  faces introduces a residual because `avg(F_x_half)` is not exactly zero even
  when each `F_x_half` is zero to machine precision (the zero values have
  different floating-point representations at different interfaces).

**At realistic kappa=1000**: The residual scales linearly with kappa, so
~50x smaller: ~2e-12 K/s. For the 10-year global overturning experiment
this produces ~0.06 K of spurious Redi drift — likely acceptable. For
century-scale climate runs it is not.

### Triad approach (needed for exact isopycnal projection)

The triad discretization (Griffies et al. 1998; NEMO default) decomposes
each flux at a u-face into contributions from 4 quarter-cell "triads". Each
triad uses the SAME 3 density/tracer values to form both the slope and the
tracer gradient, guaranteeing exact cancellation by construction.

For a u-face (i+1/2, j, k), the 4 triads use:
```
Triad 1: (i,k), (i+1,k), (i,k+1)     — upper-left quarter-cell
Triad 2: (i,k), (i+1,k), (i,k-1)     — lower-left quarter-cell
Triad 3: (i+1,k), (i,k), (i+1,k+1)   — upper-right quarter-cell
Triad 4: (i+1,k), (i,k), (i+1,k-1)   — lower-right quarter-cell
```

Each triad computes a local slope from its 3 density values and applies it
to the tracer gradient from the SAME 3 values. The total flux is the average
of the 4 triad contributions.

**Properties**:
- Exact conservation of tracer mean (antisymmetric operator)
- Exact dissipation of tracer variance (no artificial variance creation)
- Zero flux for locally-referenced potential density (no spurious diapycnal mixing)
- Used by NEMO, available in MITgcm (`GM_useTriads=.TRUE.`)

**Implementation plan for triads**:
1. Add `compute_triad_slopes_latlon_cgrid()` alongside existing centered slopes
2. Add `gm_redi_tracer_tendency_triads_latlon_cgrid()` that builds fluxes from
   4 triads per face
3. Config switch: `slope_scheme: str = "centered"  # or "triads"`
4. Eady validation: Redi-only tendency must be zero to machine precision at all
   times, with zero accumulated drift over 120 days even at kappa=50000
5. Conservation test: verify `sum(dT * h * area) = 0` exactly

---

## Implementation Phases

### Phase 1: Extract shared code (`_gm_redi_common.py`) — DONE

Factor grid-agnostic helpers from `gm_redi.py`:
- `compute_visbeck_kappa_gm()` — adaptive coefficient
- `dm95_taper()` — identical tapering across grids
- `vertical_flux_divergence()` — zero-pad BCs + FV divergence

### Phase 2: Core implementation (`gm_redi_latlon_cgrid.py`) — DONE

Three public functions + `LateralMixingOutput` wrapper.
Interface-level flux evaluation for reduced Redi residual.

### Phase 3: Tests — DONE

41 tests passing: 16 CS + 21 lat-lon unit + 4 Eady physics.
3 Eady GM/Redi cases in ocean test matrix.

### Phase 4: Wire into ocean model — DONE

Import fixed, masks passed, TypeError guard in factory.

### Phase 5: Global overturning experiment config — DONE

`use_gm_redi` toggle + `create_gm_redi_config()`.

### Phase 6: Triad slope discretization — DONE (2026-04-26)

Implemented as a separate code path in `gm_redi_latlon_cgrid.py`:

- New helpers `_to_uface_west`, `_to_uface_east`, `_to_vface_south`,
  `_to_vface_north`, `_triad_taper` lift cell-centred fields onto the four
  triad anchors at every face with one `jnp.roll`/`jnp.concatenate` each.
- `gm_redi_tracer_tendency_triads_latlon_cgrid(q, rho, mask, u_mask, v_mask,
  z_coord, jacobian, grid, kappa_GM, kappa_Redi, S_max)` computes:
  - 4 triad slopes per u-face and v-face;
  - 8 triad slopes per w-face (4 in x + 4 in y);
  - **per-triad DM95 tapering applied to the *full* per-triad flux**
    (diagonal + off-diagonal together), NOT to the slope alone — see
    "Taper-on-flux" below;
  - validity-aware averaging (N_valid normalisation at top/bottom levels
    so the diagonal Redi flux stays at full strength while invalid triads
    contribute zero off-diagonal flux).
- Both ρ and q are Neumann-filled internally, matching
  `compute_isopycnal_slopes_latlon_cgrid`.  *This is critical*: passing
  un-filled ρ produces drho_dz = 0 in land columns, which clips the slope to
  S_max and breaks the algebraic cancellation.
- The orchestrator `gm_redi_tracer_tendency_latlon` dispatches on
  `cfg.slope_scheme`, raising `ValueError` for unknown values.
- 18 new unit tests cover shape, finiteness, conservation, zero-tendency for
  uniform tracers, land-mask correctness, dispatch errors, and `jax.grad`
  through the triad path.  4 new physics tests confirm the per-kappa
  cancellation, equivalence-or-better with centered, and the orchestrator
  end-to-end.
- Three triad-variant Eady cases added to `scripts/matrix/run_ocean_test_matrix.py`
  (`eady_gm_redi_*_triads`); all pass.

#### Taper-on-flux (vs taper-on-slope)

The first triad implementation tapered each triad's *slope* before
combining it with the diagonal flux:

```
F_x^(m) = K_R · ∂q/∂x + (K_R − K_GM) · (taper · S^(m)) · ∂q/∂z^(m)
```

For `q = f(ρ)` this carries a residual `K_R · (1 − taper) · ∂q/∂x` —
even at slopes well below `S_max`, the DM95 ``tanh`` saturates only as
`1 − O(exp)`, leaving (1 − taper) ≈ 4×10⁻⁶ at |S| = 4×10⁻³, which times
`K_R · ∂q/∂y_v` produces a 6.6×10⁻⁷ K/s spurious flux at the jet centre
of the production Eady setup — *exactly* what the centered scheme had
been doing from the stencil mismatch, just for a different reason.

The fix: taper the *whole* per-triad flux contribution.  Each triad's
flux is built with raw (clipped, untapered) slopes — so for `q = f(ρ)`
the flux is **algebraically zero per triad** — and only then multiplied
by the per-triad taper:

```
F_x = Σ_m  w_m · taper_m · [ K_R · ∂q/∂x + (K_R − K_GM) · S^(m)_raw · ∂q/∂z^(m) ]
```

Same structure at v-face and w-face (where the taper now multiplies
the full per-triad cross + diagonal flux, not the squared slope).

Effect on the **production** Eady setup (22×10×20, κ_R = 5×10⁴):

| residual | before fix | after fix |
|---|---|---|
| triad max\|dT\|        | 8.6×10⁻¹¹ K/s | **6×10⁻¹⁶ K/s** |
| per-κ residual          | 1.7×10⁻¹⁵     | **1.2×10⁻²⁰** |
| centered max\|dT\|      | 8.6×10⁻¹¹ K/s | unchanged |

#### 120-day Eady validation (κ_R = 5×10⁴)

`scripts/validate/validate_triad_redi_120day.py` compares baseline (no GM/Redi)
against Redi-only with both schemes.  The original framing of the test
("the Redi-only simulation should stay unchanged from baseline") relies
on `q = f(ρ)` being maintained throughout the integration — which in
turn requires **`β_S = 0`** in the EOS, because:

- Numerical noise from tracer advection / mask interactions evolves S
  away from uniform by ~10⁻⁵ PSU per step at ocean cells.
- With `β_S = 7.4×10⁻⁴` (the legacy default), this 10⁻⁵ PSU drift
  contributes ~10⁻⁸ kg/m³ to ρ, breaking `q = f(ρ)` and unmasking a
  *real* (non-canceling) Redi flux that compounds to 1.1 K over 120 days.
- This is *not* a triad bug — it's a violation of the test's premise.
  Setting `β_S = 0` (now exposed via `run_kwargs["beta_S_override"]` in
  `run_eady_gm_redi`) restores `ρ = a + b·T` exactly.

Pointwise RMS divergence from baseline (β_S = 0 case):

| day | centered − baseline | **triads − baseline** |
|---:|---:|---:|
| 12 | 9.6×10⁻⁶ K | **1.0×10⁻¹² K** |
| 60 | 1.8×10⁻⁴ K | **1.9×10⁻¹² K** |
| 84 | 4.5×10⁻⁴ K | **2.3×10⁻¹² K** |

Triads stay at float64 round-off (~10⁻¹² K) for ~84 days, ~10⁷× better
than centered.  Past day 96 the dynamics become chaotic (independent
issue with the β_S = 0 thermal-wind balance at this resolution); the
84-day window is the meaningful Phase 6 validation.

---

## Files

| File | Status |
|------|--------|
| `src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py` | DONE |
| `src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py` | DONE |
| `src/legoesm/ocean/physics/lateral_mixing/gm_redi.py` | DONE (refactored) |
| `src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py` | DONE |
| `src/legoesm/ocean/physics/lateral_mixing/integration.py` | DONE |
| `src/legoesm/ocean/experiments/global_overturning.py` | DONE |
| `tests/ocean/unit/test_gm_redi_latlon_cgrid.py` | DONE (21 tests) |
| `tests/ocean/unit/test_gm_redi_eady_physics.py` | DONE (4 tests) |
| `scripts/matrix/ocean_test_matrix/experiments.py` | DONE (3 Eady cases) |
| `scripts/matrix/ocean_test_matrix/testcase.py` | DONE |
| `scripts/matrix/ocean_test_matrix/setup.py` | DONE (gm_redi param) |

---

## Known Gaps (deferred)

- **Surface-layer cross-isopycnal mixing** (see above) — Option A (taper full
  tensor) or Option B (Ferrari et al. 2010 BVP).  Affects both centered and
  triad schemes equally.  Within the triad scheme the per-triad DM95 taper
  introduces a (1 − taper²) Redi-only residual in tapered regions; this is
  numerically benign at 1° resolution but should be revisited when the BVP
  boundary-layer treatment is added.
- Factory unification (lat-lon bypasses `integration.py`)
- Config pathway unification (`LatLonCGridOceanConfig.gm_redi` vs `LateralMixingConfig`)
- Ferrari et al. (2010) BVP boundary-layer treatment
- MPAS GM/Redi (no lateral mixing on Voronoi grid yet)
- Implicit vertical diffusion for the kappa * S^2 term
- Cubed-sphere triad scheme (`gm_redi.py` still uses centred slopes; the same
  triad helper pattern would translate but is non-trivial because the CS
  geometry has 6-fold corner triads at panel edges).

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
