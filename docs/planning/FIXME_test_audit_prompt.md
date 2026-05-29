# LegoESM Test Case Audit & Fix — Claude Code Prompt

You are working in the legoESM repository, a differentiable Earth system model built on JAX. Many atmosphere and ocean test cases have broken due to recent refactoring (grid adapters, deprecation of aliases, registry-driven kernels, physics pipeline changes). Your job is to systematically audit every physics-related test case, identify what is physically wrong, and fix the underlying solver/operator code (NOT the tests) until every test passes with physically correct behavior. The code must remain fully differentiable (`jax.grad` through multi-step integration) and parallelizable (`jax.pmap`, `jax.lax.scan`).

---

## CRITICAL RULES

1. **DO NOT STOP** until every test in the current tier passes. If a fix breaks something else, keep iterating. Run the tests, read the failures, fix the code, re-run. Loop until green.
2. **Fix the source code, not the tests** — unless a test has a genuinely wrong tolerance or incorrect physical setup. The tests encode the correct physics. If a rest state test says tendencies should be < 1e-10, that is a hard physical requirement, not a suggestion.
3. **Preserve differentiability** — never introduce `jax.lax.stop_gradient`, Python-side conditionals on array values, or non-JIT-compatible control flow. Every fix must be compatible with `jax.jit`, `jax.grad`, and `jax.lax.scan`.
4. **Preserve parallelism** — do not serialize computations that were previously vmapped or pmapped. Do not add host-side loops where device-side scans existed.
5. **Work in order of difficulty**: Spectral → Lat-Lon FV → MPAS/Icosahedral → Cubed-Sphere C-D grid. Each tier builds confidence before tackling harder grid stencils.
6. **After each file fix**, re-run that tier's tests immediately to confirm the fix didn't regress anything.

---

## REPOSITORY LAYOUT

### Source code (fix these):
```
src/legoesm/atmosphere/dynamics/
├── spectral_sw.py              # Spectral shallow water
├── spectral_pe.py              # Spectral hydrostatic primitive equations
├── spectral_nh.py              # Spectral nonhydrostatic compressible Euler
├── shallow_water_fv_latlon.py  # Lat-lon FV shallow water
├── primitive_eq_latlon.py      # Lat-lon centered primitive equations
├── primitive_eq_fv_latlon.py   # Lat-lon FV primitive equations
├── compressible_euler_fv_latlon.py  # Lat-lon FV nonhydrostatic
├── shallow_water_mpas.py       # MPAS icosahedral shallow water
├── primitive_eq_mpas.py        # MPAS hydrostatic
├── compressible_euler_mpas.py  # MPAS nonhydrostatic
├── shallow_water_fv3_cdgrid.py # Cubed-sphere C-D grid shallow water
├── primitive_eq_cdgrid.py      # Cubed-sphere C-D grid hydrostatic
├── compressible_euler_cdgrid.py # Cubed-sphere C-D grid nonhydrostatic
├── shallow_water_cgrid_latlon.py # Lat-lon C-grid shallow water
├── sfno_sw.py                  # SFNO shallow water
├── sfno_pe.py                  # SFNO hydrostatic
├── tracer_transport.py         # Tracer transport
├── edge_blending.py            # Panel edge blending for cubed sphere
├── compressible_euler.py       # Shared NH utilities
└── __init__.py                 # Module exports, solver registry, deprecation aliases

src/legoesm/ocean/dynamics/
├── ocean_model.py              # Cubed-sphere ocean wrapper
├── ocean_pe.py                 # Ocean PE tendencies (deprecated wrapper)
├── ocean_pe_cdgrid.py          # C-D grid ocean baroclinic tendencies
├── ocean_pe_fc.py              # FC discretization ocean
├── ocean_pe_fc_cgrid.py        # FC + C-grid ocean
├── ocean_pe_fv.py              # FV ocean (deprecated wrapper)
├── ocean_pe_latlon.py          # Lat-lon ocean tendencies
├── ocean_pe_mpas.py            # MPAS ocean tendencies
├── ocean_model_latlon.py       # Lat-lon ocean model
├── ocean_model_mpas.py         # MPAS ocean model
├── spectral_ocean_pe.py        # Spectral ocean
├── sfno_ocean.py               # SFNO ocean
├── barotropic.py               # Cubed-sphere barotropic substeps
├── barotropic_latlon.py        # Lat-lon barotropic substeps
├── barotropic_mpas.py          # MPAS barotropic substeps
└── latlon_operators.py         # Lat-lon ocean operators

src/legoesm/core/
├── conservation.py             # Conservation fixers (mass, energy, enstrophy)
├── constants.py                # Physical constants (g, Omega, R_earth, cp, Rd, kappa, ...)
├── grids.py                    # Grid constructors (CubedSphereGrid, LatLonGrid, GaussianGrid, VoronoiMesh)
└── ...

src/legoesm/driver/
├── grid_adapters.py            # ColumnAdapter, SingleColumnGrid, make_adapter
├── kernel_registry.py          # Registry-driven physics kernel selection
├── physics_pipeline.py         # Physics pipeline (radiation, convection, microphysics)
└── ...
```

### Tests (these encode correct physics — read them carefully):
```
tests/atmosphere/shallow_water/
├── unit/test_spectral.py                  # Spectral SW: SH transforms, Williamson TC2/5, diagnostics, differentiability
├── unit/test_shallow_water_fv_latlon.py   # Lat-lon FV SW: operators, Williamson TC2/5, mass conservation, differentiability
├── unit/test_sfno_sw.py                   # SFNO SW
├── integration/test_shallow_water.py      # Cubed-sphere SW: tendencies, stability, mass conservation, differentiability
├── integration/test_fv_cubesphere.py      # Cubed-sphere FV SW: tendencies, 100-step stability, mass conservation, differentiability
├── integration/test_fv_convergence.py     # Cubed-sphere FV convergence: Williamson TC2 error norms at C8/C16/C32
├── integration/test_shallow_water_mpas.py # MPAS SW: Williamson TC2/5/6, mass/energy conservation, differentiability

tests/atmosphere/hydrostatic/
├── unit/test_spectral_pe.py               # Spectral PE: 3D SH, geopotential, rest state, semi-implicit, differentiability
├── unit/test_primitive_eq_latlon.py       # Lat-lon PE: rest state, mass conservation, Held-Suarez, differentiability
├── unit/test_primitive_eq_fv_latlon.py    # FV lat-lon PE: rest state, stability, mass conservation
├── integration/test_fv_cubesphere.py      # Cubed-sphere PE: rest state 50-step stability, differentiability
├── integration/test_amip_smoke.py         # AMIP smoke test
├── integration/test_amip_stability.py     # AMIP stability
├── integration/test_amip_rrtmg.py         # AMIP with RRTMG radiation
├── validation/test_held_suarez_fix.py     # Held-Suarez validation
├── validation/test_stability_fix.py       # Stability fix validation

tests/atmosphere/nonhydrostatic/
├── unit/test_compressible_euler.py        # NH: height coord, reference state, acoustic modes, Kessler, differentiability
├── unit/test_compressible_euler_fv_latlon.py # NH FV lat-lon: rest state, polar filter, stability
├── unit/test_spectral_nh.py               # Spectral NH: acoustic substeps, Exner function, rest state
├── integration/test_fv_cubesphere.py      # NH cubed-sphere: 30-step stability, differentiability

tests/ocean/
├── unit/test_ocean.py                     # Cubed-sphere ocean: EOS, z-star, tendencies, conservation, long-run
├── unit/test_ocean_fv.py                  # FC discretization ocean
├── unit/test_ocean_fc.py                  # FC discretization ocean
├── unit/test_mpas_ocean.py                # MPAS ocean: state, velocity recon, barotropic, conservation
├── unit/test_sfno_ocean.py                # SFNO ocean: channel packing, conservation
├── unit/test_ocean_biogeochemistry.py     # Ocean biogeochemistry
├── unit/test_ocean_compatibility.py       # Backward compatibility
├── unit/test_bathymetry.py                # Bathymetry processing
├── unit/test_freshwater.py                # Freshwater forcing
├── validation/test_differentiability_ocean.py # JAX grad through 10 ocean steps
```

### Test case initialization (read these for correct physics):
```
tests/atmosphere/shallow_water/test_cases/williamson.py          # Cubed-sphere Williamson TC2/5
tests/atmosphere/shallow_water/test_cases/williamson_latlon.py   # Lat-lon Williamson TC2/5
tests/atmosphere/shallow_water/test_cases/williamson_mpas.py     # MPAS Williamson TC2/5/6
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/             # DCMIP-2025 test cases 1-3
```

---

## TIER 1: SPECTRAL (easiest — do this first)

### Files to test:
```bash
python -m pytest tests/atmosphere/shallow_water/unit/test_spectral.py -v --tb=short
python -m pytest tests/atmosphere/hydrostatic/unit/test_spectral_pe.py -v --tb=short
python -m pytest tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py -v --tb=short
```

### What must be physically correct:

**Shallow water (spectral_sw.py):**
- Spherical harmonic transforms must round-trip exactly (atol=1e-10)
- Williamson Test Case 2 (steady geostrophic flow): a state in exact geostrophic balance with solid-body rotation u₀·cos(lat) and the matching height field h = h₀ − (RΩu₀ + u₀²/2)sin²(lat)/g must produce near-zero tendencies. This is the single most important physical test — geostrophic balance is the foundation of GFD.
- Williamson Test Case 5 (mountain flow): must remain stable over 500 steps with mountain topography
- Mass, energy, and enstrophy must be conserved (diagnosed, not enforced)
- Solid-body rotation velocity reconstruction must be accurate (rtol=1e-3)
- Hyperdiffusion must be dissipative (eigenvalue-based, order 2)
- All operations must be differentiable

**Hydrostatic PE (spectral_pe.py):**
- 3D SH analysis→synthesis round-trip (atol=1e-10)
- Isothermal rest state (T=300K, zero wind, p_s=1e5) must have near-zero tendencies: vorticity/divergence < 1e-10, temperature < 1e-8, ln(p_s) < 1e-10
- Geopotential must decrease upward (Φ increases with pressure decrease → increases with index if top-to-bottom)
- Hydrostatic balance in the geopotential computation
- Semi-implicit time stepping must stabilize large dt
- Rest state must remain stable over 50 steps (T stays within 300±1K)
- Differentiable through tendency computation

**Nonhydrostatic (spectral_nh.py):**
- Rest state: all spectral perturbation coefficients zero, tendencies < 1e-10 (vor, div), < 1e-8 (θ', ρ')
- Acoustic substeps must preserve rest state to machine precision (atol=1e-15)
- Exner perturbation π'=0 when ρ'=θ'=0 (atol=1e-15)
- Multi-step stability: θ'_max < 1.0 after 20 steps
- Surface geopotential tendency must be exactly zero (topography is static)

### Source files to fix:
- `src/legoesm/atmosphere/dynamics/spectral_sw.py`
- `src/legoesm/atmosphere/dynamics/spectral_pe.py`
- `src/legoesm/atmosphere/dynamics/spectral_nh.py`
- `src/legoesm/core/grids.py` (GaussianGrid, SH transforms)

---

## TIER 2: LAT-LON FINITE VOLUME

### Files to test:
```bash
python -m pytest tests/atmosphere/shallow_water/unit/test_shallow_water_fv_latlon.py -v --tb=short
python -m pytest tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon.py -v --tb=short
python -m pytest tests/atmosphere/hydrostatic/unit/test_primitive_eq_fv_latlon.py -v --tb=short
python -m pytest tests/atmosphere/nonhydrostatic/unit/test_compressible_euler_fv_latlon.py -v --tb=short
```

### What must be physically correct:

**Shallow water FV (shallow_water_fv_latlon.py):**
- Constant field → zero tendency (discrete conservation of constants)
- Zero velocity → zero tendency (rest state)
- Gradient of constant field = 0
- Williamson TC2 1-day L2 error < 0.05 (5%) at 32×64 resolution
- Williamson TC2 stable over 5 days
- Williamson TC5 stable over 15 days with mountain topography preserved
- Mass conservation with fixer: relative drift < 1e-5
- Differentiable through single step and scan

**Hydrostatic lat-lon (primitive_eq_latlon.py):**
- Isothermal rest state: du/dt, dv/dt < 1e-8; dT/dt < 1e-6; dp_s/dt < 1e-4
- Rest state stays at rest: u,v drift < 1e-4 m/s after one step
- Mass conservation with fixer: relative error < 1e-5
- Held-Suarez: temperature stays within 100-400K over 50 steps
- Discrete continuity closure: residual < 1e-6 relative to divergence

**Hydrostatic FV lat-lon (primitive_eq_fv_latlon.py):**
- Same rest-state tolerances as centered lat-lon
- Multi-step stability over 10 steps
- Mass conservation: rel_err < 1e-5 over 20 steps
- Static phis: unchanged to atol=1e-12

**Nonhydrostatic FV lat-lon (compressible_euler_fv_latlon.py):**
- Rest state: du/dt, dv/dt < 1e-6
- Stable over 5 steps at dt=1s
- Polar filter must not introduce instability
- Static phis to 1e-12

### Source files to fix:
- `src/legoesm/atmosphere/dynamics/shallow_water_fv_latlon.py`
- `src/legoesm/atmosphere/dynamics/primitive_eq_latlon.py`
- `src/legoesm/atmosphere/dynamics/primitive_eq_fv_latlon.py`
- `src/legoesm/atmosphere/dynamics/compressible_euler_fv_latlon.py`

**Also run the ocean lat-lon tests:**
```bash
# If there are lat-lon ocean tests, run them here too
python -m pytest tests/ocean/unit/test_ocean.py -k "latlon or LatLon" -v --tb=short
```

---

## TIER 3: MPAS / ICOSAHEDRAL

### Files to test:
```bash
python -m pytest tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py -v --tb=short
python -m pytest tests/ocean/unit/test_mpas_ocean.py -v --tb=short
```

### What must be physically correct:

**MPAS shallow water (shallow_water_mpas.py):**
- Williamson TC2 5-day height error < 200m at level-3 mesh (642 cells)
- Williamson TC5 stable over 15 days; mass drift < 1e-10
- Williamson TC6 (Rossby-Haurwitz wave 4) stable over 5 days
- Mass conservation: relative drift < 1e-13 (near machine precision with fixer)
- Energy conservation: relative drift < 1e-10 with energy-conserving PV scheme
- Differentiable through 3 steps
- Velocity projection to edges: `u_edge = u_east · cos(angle) + v_north · sin(angle)` where angle is the edge normal angle

**MPAS ocean (ocean_model_mpas.py, ocean_pe_mpas.py):**
- Rest state: |du/dt|_max < 1.0 m/s², |dη/dt|_max < 1e-10 m/s
- Barotropic substeps: rest state |Δη|_max < 1e-10, |u_bar|_max < 1e-10
- Volume conservation with fixer: atol = 1e-10 × total_area
- Heat conservation with fixer: relative error < 1e-10
- Multi-step stability over 5 steps at dt=30s
- Land masking: zero tendencies on land cells

### Source files to fix:
- `src/legoesm/atmosphere/dynamics/shallow_water_mpas.py`
- `src/legoesm/ocean/dynamics/ocean_pe_mpas.py`
- `src/legoesm/ocean/dynamics/ocean_model_mpas.py`
- `src/legoesm/ocean/dynamics/barotropic_mpas.py`

---

## TIER 4: CUBED-SPHERE C-D GRID (hardest)

### Files to test:
```bash
# Shallow water
python -m pytest tests/atmosphere/shallow_water/integration/test_shallow_water.py -v --tb=short
python -m pytest tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py -v --tb=short
python -m pytest tests/atmosphere/shallow_water/integration/test_fv_convergence.py -v --tb=short

# Hydrostatic
python -m pytest tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py -v --tb=short

# Nonhydrostatic
python -m pytest tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py -v --tb=short

# Ocean cubed-sphere
python -m pytest tests/ocean/unit/test_ocean.py -v --tb=short
python -m pytest tests/ocean/unit/test_ocean_fv.py -v --tb=short
python -m pytest tests/ocean/unit/test_ocean_fc.py -v --tb=short
python -m pytest tests/ocean/validation/test_differentiability_ocean.py -v --tb=short
```

### What must be physically correct:

**Cubed-sphere shallow water (shallow_water_fv3_cdgrid.py):**
- Tendencies: finite for Williamson TC2/5
- 100-step stability: mean height drift < 0.1% (1e-3 relative)
- Mass conservation: drift < 1e-4 over 50 steps; < 1e-5 with high-precision fixer
- Williamson TC2 convergence: L2 error < 5% at C8/C16/C32 (note: discrete steady state differs from continuous due to A→D interpolation on C-D grid)
- Differentiable through 10 steps

**Cubed-sphere hydrostatic (primitive_eq_cdgrid.py):**
- Single step finite at dt=300s
- Rest state: p_s drift < 5e-3 relative over 50 steps
- Differentiable through 10 steps
- Divergence damping must not introduce spurious modes

**Cubed-sphere nonhydrostatic (compressible_euler_cdgrid.py):**
- Rest state: tendencies finite
- 30-step stability at dt=10s: ρ' drift < 1.0 kg/m³
- Differentiable through 10 steps
- Acoustic substeps (split-explicit) must handle fast waves correctly

**Cubed-sphere ocean (ocean_model.py, ocean_pe_cdgrid.py):**
- Wright EOS: reference density matches seawater tables
- Z-star coordinate: monotonic depth, positive thickness, Jacobian=1 when η=0
- Rest state tendencies < 1e-3 m/s²
- Split-explicit barotropic: min water column enforced (0.5m)
- Volume conservation: mean η error < 1e-6 with fixer
- Heat/salt conservation: relative error < 1e-8 long-run with fixer
- 10-step stability: T stays within -5°C to 40°C
- Differentiable through 10 steps
- FC discretization: tendencies finite, land masking correct

### Source files to fix:
- `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py`
- `src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py`
- `src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py`
- `src/legoesm/atmosphere/dynamics/edge_blending.py`
- `src/legoesm/ocean/dynamics/ocean_model.py`
- `src/legoesm/ocean/dynamics/ocean_pe_cdgrid.py`
- `src/legoesm/ocean/dynamics/ocean_pe_fc.py`
- `src/legoesm/ocean/dynamics/ocean_pe_fc_cgrid.py`
- `src/legoesm/ocean/dynamics/barotropic.py`

---

## CROSS-CUTTING CONCERNS

### After all four tiers pass, run these cross-cutting tests:
```bash
# AMIP integration (uses physics pipeline + cubed-sphere dynamics)
python -m pytest tests/atmosphere/hydrostatic/integration/test_amip_smoke.py -v --tb=short
python -m pytest tests/atmosphere/hydrostatic/integration/test_amip_stability.py -v --tb=short
python -m pytest tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py -v --tb=short

# Held-Suarez validation
python -m pytest tests/atmosphere/hydrostatic/validation/ -v --tb=short

# SFNO models
python -m pytest tests/atmosphere/shallow_water/unit/test_sfno_sw.py -v --tb=short
python -m pytest tests/ocean/unit/test_sfno_ocean.py -v --tb=short

# Physics pipeline (Task 7 tests — must not regress)
python -m pytest tests/unit/test_physics_grid_adapters.py -v --tb=short

# Ocean biogeochemistry, bathymetry, freshwater
python -m pytest tests/ocean/unit/test_ocean_biogeochemistry.py tests/ocean/unit/test_bathymetry.py tests/ocean/unit/test_freshwater.py -v --tb=short

# Differentiability (ocean)
python -m pytest tests/ocean/validation/test_differentiability_ocean.py -v --tb=short

# Full deprecation tests (must not regress from Task 8)
python -m pytest tests/unit/test_deprecation_warnings.py -v --tb=short
```

---

## COMMON FAILURE PATTERNS & FIXES

These are the most likely issues based on the recent refactoring:

1. **Import path changes**: Aliases were moved to `__getattr__`-based deprecation. If a test or source file imports the old name directly from a submodule (not through `__init__.py`), it may fail. Fix: update the import to use the canonical name.

2. **Shape mismatches from grid adapter changes**: The physics pipeline now uses `ColumnAdapter.flatten_3d/unflatten_3d` instead of raw `.reshape()`. If a solver returns data in a shape the adapter doesn't expect, you'll get reshape errors. Fix: ensure solver output shapes match what the adapter's `shape_2d` + `nlev` imply.

3. **Conservation fixer signature changes**: The `_latlon` suffixed functions are now deprecated wrappers. If the underlying `apply_conservation_fixer` changed its signature, the wrapper may not match. Fix: ensure the unified function accepts the same arguments.

4. **Discretization name changes**: `"centered"` → `"cdgrid"`, `"finite_volume"` → `"cdgrid"`, `"fv"` → `"cdgrid"`. Source code that hard-codes old names in if/elif chains will take wrong branches. Fix: update to canonical names.

5. **Registry lookup failures**: If a physics scheme name doesn't match the registry key, `resolve_kernel` raises `KeyError`. Fix: ensure config scheme names match registry keys exactly.

6. **Geostrophic balance precision**: The C-D grid stagger introduces an O(Δx²) error in geostrophic balance because velocities live at cell edges, not centers. The Williamson TC2 test accounts for this by allowing ~5% error. If the A→D interpolation operator changed, this error bound may be violated. Fix: verify the interpolation operator preserves the correct stagger.

7. **Acoustic substep stability**: Nonhydrostatic models use split-explicit time stepping with fast acoustic substeps. If the substep count or the implicit/explicit split changed, acoustic modes can blow up. Fix: verify `n_acoustic_substeps` and the pressure-gradient / divergence split.

8. **Coriolis discretization**: On the C-D grid, the Coriolis term must use energy-conserving interpolation (Arakawa & Lamb 1981). If the interpolation weights changed, rest-state balance breaks. Fix: verify the Coriolis stencil matches the expected averaging.

---

## WORKFLOW

```
For each tier (Spectral → LatLon → MPAS → CubedSphere):
    1. Run the tier's tests: `python -m pytest <test_files> -v --tb=short`
    2. Read every failure traceback carefully
    3. For each failure:
        a. Read the test to understand the physical requirement
        b. Read the corresponding source file
        c. Identify the bug (wrong formula, shape mismatch, import error, etc.)
        d. Fix the source code
        e. Re-run JUST that test file to verify
    4. Once all tests in the tier pass, re-run the ENTIRE tier to catch interactions
    5. Also re-run all PREVIOUS tiers to catch regressions
    6. Only then move to the next tier
```

**Final validation** — run the complete test suite:
```bash
python -m pytest tests/atmosphere/ tests/ocean/ -v --tb=short 2>&1 | tee test_results.txt
```

Do not stop until this command shows 0 failures (excluding any tests that are skipped due to missing optional dependencies like x64 mode or MPI).

---

## PHYSICAL REFERENCE: KEY EQUATIONS

### Shallow Water Equations (rotating sphere):
```
∂h/∂t + ∇·(hu) = 0                          (mass conservation)
∂u/∂t + (f + ζ)k×u + ∇(½|u|² + g(h+h_s)) = 0   (momentum, vector-invariant form)
```
where f = 2Ω sin(lat), ζ = k·∇×u (relative vorticity)

### Geostrophic Balance (Williamson TC2):
```
u = u₀ cos(lat)
v = 0
g·h = g·h₀ − (RΩu₀ + u₀²/2) sin²(lat)
```
Tendencies must be zero (analytically). Discrete tendencies should be < 1e-10 for spectral, < 1e-4 for finite volume.

### Hydrostatic Primitive Equations:
```
∂u/∂t = −(ζ+f)v − ∂Φ/∂x + F_u              (zonal momentum)
∂v/∂t =  (ζ+f)u − ∂Φ/∂y + F_v              (meridional momentum)
∂T/∂t = −u·∇T + κTω/p + Q                    (thermodynamic)
∂ln(p_s)/∂t = −∫₀¹ ∇·(v) dσ                  (continuity)
Φ = Φ_s + R_d ∫_{σ}^{1} T d(ln p)            (hydrostatic, integrated upward)
```

### Nonhydrostatic Compressible Euler:
```
∂ρ'/∂t = −∇·(ρu) − ∂(ρw)/∂z + ρ_ref ∂w/∂z
∂(ρu)/∂t = −∇·(ρuu) − ρ c_p θ ∇π' + fk×(ρu)
∂(ρw)/∂t = −∇·(ρuw) − ρ c_p θ ∂π'/∂z − ρ'g
∂(ρθ')/∂t = −∇·(ρθu) − ∂(ρθw)/∂z
π = (p/p₀)^(R_d/c_p),  p = ρ R_d T,  T = θ π
```

### Ocean (z-star, split-explicit):
```
∂η/∂t + ∇·(Hu_bar) = 0                       (barotropic continuity)
∂u_bar/∂t = −g∇η − fk×u_bar + ...            (barotropic momentum)
∂u'/∂t = −∇p_bc/ρ₀ − fk×u' + ...             (baroclinic momentum)
∂T/∂t = −∇·(uT) − ∂(wT)/∂z + K_v ∂²T/∂z²   (tracer transport)
```

Good luck. Do not stop until everything is green.
