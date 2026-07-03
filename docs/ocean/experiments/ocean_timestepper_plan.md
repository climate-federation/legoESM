# Plan: AB2 and RK3 Time Integration for Ocean Tracer Advection

## Motivation

The latlon C-grid ocean model uses forward Euler for tracer advection
(lines 386-387 of `ocean_model_latlon_cgrid.py`):
```python
T_new = state.T.data + dt * tend.dT_dt.data
```

This is the **only** production-relevant ocean model using 1st-order time
integration for tracers:
- **MITgcm**: Adams-Bashforth (Hill et al. 2012)
- **Oceananigans**: AB2 (3D baroclinic), RK3 (2D turbulence) (Silvestri 2024)
- **NEMO**: leapfrog + Robert-Asselin filter

Forward Euler causes two critical problems discovered during advection
convergence testing (branch `dhruv/advection-convergence-tests`):

1. **WENO5/7 instability in 2D**: Non-monotone WENO oscillations grow
   over many Euler steps, causing blowup at fine resolution. AB2/RK3
   don't have the leading-order anti-diffusive truncation error that
   triggers this.

2. **All schemes limited to O(dx) convergence**: With CFL-proportional dt,
   temporal error O(dt)=O(dx) dominates regardless of spatial order.
   WENO5 (5th order spatial) looks identical to upwind (1st order).

3. **Implicit diffusion comparisons confounded**: Hill et al. showed SOM
   is 3-4 orders of magnitude better than DST-3, but with forward Euler
   this difference is masked by the temporal error floor.

## Options

### Option A: Adams-Bashforth 2 (AB2)

```
T^{n+1} = T^n + dt * [(3/2)*F^n - (1/2)*F^{n-1}]
```

- 2nd-order accurate in time
- Only 1 tendency evaluation per step (same cost as Euler)
- Requires storing previous tendency F^{n-1} (extra memory for dT_dt, dS_dt)
- Conditionally stable — needs stabilization parameter ε ~ 0.1
  (MITgcm uses `ABepsBar = 0.1`)
- Stabilized form: `T^{n+1} = T^n + dt * [(3/2+ε)*F^n - (1/2+ε)*F^{n-1}]`
- What MITgcm and Silvestri use — well-validated in ocean modeling

### Option B: SSP-RK3

```
k1 = T^n + dt * F(T^n)
k2 = 3/4 * T^n + 1/4 * (k1 + dt * F(k1))
T^{n+1} = 1/3 * T^n + 2/3 * (k2 + dt * F(k2))
```

- 3rd-order accurate, strong stability preserving (TVD)
- 3 tendency evaluations per step (3x cost of Euler)
- No extra storage needed (no previous-tendency carry)
- Already implemented in `src/legoesm/timestepping/ssp_rk3.py`
- Unconditionally preserves monotonicity if the spatial operator does
- What Oceananigans uses for 2D turbulence

### Recommendation: Implement both

- **AB2 as default**: Cheap (1x cost), 2nd order, matches MITgcm.
  Sufficient for production runs and WENO stability.
- **RK3 as option**: For accuracy-critical work, convergence testing,
  and as a stability reference. Higher cost justified when needed.

## Implementation

### Changes needed

#### 1. Config: `src/legoesm/ocean/state.py`

Add to `LatLonCGridOceanConfig`:
```python
tracer_time_integrator: str = "euler"  # "euler", "ab2", "rk3"
ab2_epsilon: float = 0.1  # AB2 stabilization parameter
```

#### 2. State: `src/legoesm/ocean/state.py`

For AB2, add previous tendency fields to `LatLonCGridOceanState`:
```python
dT_dt_prev: Field | None = None  # Previous T tendency (AB2 only)
dS_dt_prev: Field | None = None  # Previous S tendency (AB2 only)
```

Or store in SegmentCarry if using the compiled segment path.

#### 3. Model step: `src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py`

Replace lines 386-387 with dispatch:
```python
if self.config.tracer_time_integrator == "ab2":
    eps = self.config.ab2_epsilon
    if state.dT_dt_prev is not None:
        T_new = T + dt * ((1.5 + eps) * dT_dt - (0.5 + eps) * dT_dt_prev)
        S_new = S + dt * ((1.5 + eps) * dS_dt - (0.5 + eps) * dS_dt_prev)
    else:
        # First step: fall back to Euler
        T_new = T + dt * dT_dt
        S_new = S + dt * dS_dt
elif self.config.tracer_time_integrator == "rk3":
    # 3 sub-steps, each calling tendencies()
    ...
else:
    # Forward Euler (current behavior)
    T_new = T + dt * dT_dt
    S_new = S + dt * dS_dt
```

For AB2: store current tendency for next step:
```python
state_new = state_new._replace(
    dT_dt_prev=tend.dT_dt,
    dS_dt_prev=tend.dS_dt)
```

For RK3: adapt the existing `ssp_rk3_step` from
`src/legoesm/timestepping/ssp_rk3.py`. The challenge is that
`tendencies()` computes all tendencies (momentum + tracer), but
RK3 only applies to tracers. Need to either:
(a) Call full tendencies() 3 times (expensive, also changes momentum)
(b) Factor out a `tracer_tendencies()` that only computes dT/dS
    (cleaner but requires refactoring)

Option (a) is simpler initially. Option (b) is the right long-term
approach since momentum already has its own Matsuno stepping.

#### 4. Flux-form tracer advection (lines 494-753)

The flux-form tracer update currently happens AFTER the barotropic
solver in `step()`. For AB2, the tendency from this step needs to be
stored. The "tendency" in flux form is:
```
dT/dt = -(1/h) * [div(h*u*T_face) + vert_flux_div]
```

This is computed but not stored as a tendency — it's applied inline.
Need to refactor to compute the tendency first, store it, then apply
the time integration.

#### 5. SegmentCarry (if using compiled segments)

For AB2 with `lax.scan`, the previous tendency must be part of
SegmentCarry. This is a cross-cutting change per CLAUDE.md rules:
update the NamedTuple, `pack_carry`, `unpack_carry`, and all
constructors.

#### 6. Tests

- Existing advection convergence tests: re-run with `ab2` and `rk3`
  to verify WENO5 stability and proper convergence rates
- New unit test: verify AB2 is 2nd-order on a simple ODE
- New unit test: verify RK3 matches existing `ssp_rk3_step` output
- Re-run Level 2 deformational flow with AB2 — WENO5/7 should be stable

## Estimated scope

| Component | LOC | Risk |
|---|---|---|
| Config fields | ~5 | Low |
| State fields (AB2 prev tendency) | ~10 | Low |
| AB2 dispatch in step() | ~30 | Medium (flux-form refactor) |
| RK3 dispatch in step() | ~50 | Medium (tendency factoring) |
| SegmentCarry update | ~30 | Medium (cross-cutting) |
| Tests | ~100 | Low |
| **Total** | **~225** | |

## Verification plan

1. Convergence test Level 1 with AB2: upwind should show 1st-order
   spatial + 2nd-order temporal → ~2nd order overall. WENO5 should
   show spatial order visible above temporal floor.
2. Convergence test Level 2 with AB2: WENO5/7 should be STABLE at
   128x256 (currently blows up with Euler).
3. Eady comparison: re-run the weak-forcing advection comparison with
   AB2 + implicit barotropic solver to get the definitive scheme ranking.
4. Existing test suite: all ocean tests must still pass with default
   `tracer_time_integrator="euler"`.

## References

- Hill et al. (2012), Ocean Modelling 45-46. MITgcm AB time stepping.
- Silvestri et al. (2024), JAMES. AB2 for baroclinic jet, RK3 for 2D.
- Shu & Osher (1988), J. Comp. Phys. SSP-RK3 scheme.
- Durran (2010), Numerical Methods for Fluid Dynamics. AB2 stability.
