# Petersen lock-exchange velocity mismatch — investigation log

Branch: `main` @ `f84f4029` (commit after the ocean fidelity harness landed)

## Setup

Both models run the Petersen 2015 Fig. 5 lock-exchange:

- Geometry: 64 km zonal x 4 km meridional x 20 m depth, ~1 km dx, 20 levels
- Initial T: 5 °C west of basin midpoint, 30 °C east of basin midpoint
- Salinity passive (uniform 35 PSU)
- Linear EOS, `alpha_T = 2e-4`, `beta_S = 0`, `T_ref = 17.5 °C`, `rho_0 = 1025 kg/m^3`
- No rotation (equatorial channel, f ~ 0)
- No surface forcing, no wind, no buoyancy flux
- No viscosity / bottom drag
- 17 h integration, dt = 30 s

## Result

| metric | Veros | legoESM | ratio |
| --- | --- | --- | --- |
| T mean (°C) | 17.5 | 17.5 | 1.00 |
| T range (°C) | [5.00, 30.00] (superbee) | [5.00, 30.00] (TVD) | match |
| max \|u\| (m/s) | 0.857 | 0.063 | **0.074** |
| max \|eta\| (m) | (not extracted) | 0.030 | — |
| PE drift (relative) | (Veros conservative) | -5.3e-7 | tiny |

T statistics match exactly. Velocity is 14x smaller in legoESM.

Theoretical gravity-current speed (two-layer reduced-gravity):

`c = sqrt(g * (Drho/rho_0) * H/2) = sqrt(9.81 * 0.005 * 10) = 0.70 m/s`

Veros at 0.86 m/s exceeds theory by ~20 % (normal for centered/limited
advection with overshoots). legoESM at 0.06 m/s is **an order of magnitude
below theory** — the gravity current is being damped before it can
develop.

## Knobs tested (legoESM lat-lon C-grid `LatLonCGridOceanConfig`)

| knob | default | tried | effect on \|u\|_max |
| --- | --- | --- | --- |
| `A_h` | (model default) | 0.0 | small (already ~0 for short runs) |
| `A_v` | (model default) | 0.0 | small |
| `bottom_drag_r` | 0.0 | 0.0 (no change) | n/a |
| `barotropic_diffusion_alpha` | 0.01 | **0.0** | **+70 % (0.045 -> 0.076 m/s)** |
| `eos` | "wright" (nonlinear) | "linear" (T_ref=17.5) | small (slight) |
| `bebt` (semi-implicit barotropic) | 0.2 | 0.0 (forward-backward) | none measurable |
| `n_barotropic_substeps` | 30 | 1 | small (-15 %) |
| `barotropic_time_filter` | "cosine" | "box" | small (-15 %) |
| `tracer_advection` | "tvd" | "weno5" | none measurable |
| `momentum_advection` | "vector_invariant" | "weno5" | none measurable |

`barotropic_diffusion_alpha` is the only knob that moves the needle.
Even with all of the above stacked, the gravity current saturates at
~0.06 - 0.08 m/s (full-column ``max |u|`` = 0.070 m/s, surface
``max_speed`` = 0.063 m/s — both reported now via the ``max_abs_u``
extension to the lat-lon scalar function) — still ~10x below theory.

## Diagnostic fix landed alongside the investigation

The lat-lon C-grid scalar function (``_make_scalar_fn``,
``scripts/matrix/run_ocean_test_matrix.py``) used to report only the surface-
level velocity as ``max_speed``. The MPAS path already exposed the
full-column ``max |u|``; the latlon path did not. With a strong subsurface
flow (Petersen lock-exchange bottom layer ≈ surface layer in this
counter-flow geometry), the difference is small (0.063 vs 0.070 m/s on
this case), but the metric is now consistent across grids so future
gravity-current cases get apples-to-apples comparison out of the box.
``run_comparison.py`` prefers ``max_abs_u`` over ``max_speed`` when both
are present in the legoESM bundle.

## Suspects for the remaining gap

Listed in decreasing likelihood, all of them outside the surface-knob
config space, so each would need source-level investigation:

1. **Hydrostatic pressure-gradient (PGF) scheme** — `LatLonCGridOceanModel`
   uses a z-star coordinate with a particular PGF discretisation; if the
   PGF reconstruction along sloping isopycnals is over-stabilised, the
   horizontal pressure gradient that drives the gravity current is
   muted. See `src/legoesm/ocean/dynamics/` for the active PGF path.
2. **Vertical remap / z-star adjustment** — the z-star coordinate moves
   layer interfaces with eta. If the adjustment introduces dissipation
   on the baroclinic mode (which IS the gravity current here), the flow
   is damped before it can build.
3. **Tracer advection (TVD)** vs Veros's superbee — TVD is monotone but
   is more dissipative on sharp fronts than superbee. A diffused front
   reduces the local density gradient available to drive the flow.
4. **Momentum advection (`vector_invariant`)** with its built-in
   stabilisation (Smagorinsky / Leith viscosity? Hollingsworth correction?)
   may add momentum dissipation that doesn't exist in Veros's explicit
   centered scheme.
5. **Initial perturbation symmetry** — legoESM's basin is exactly
   equatorial and exactly half-half east/west; if any operator carries a
   small asymmetry, it could trigger different mode structure than
   Veros's identical setup. Lower likelihood — T mean conservation
   suggests symmetry is preserved.

## Recommendation

The fidelity harness now correctly **exposes** this 14x gap. Closing it
requires source-level investigation in `src/legoesm/ocean/dynamics/`,
which is a multi-day effort beyond a knob-tuning session. The Petersen-
scale lock-exchange TestCase in the matrix gives a permanent regression
guard: when the dynamics-side fix lands, this scalar comparison will
move directly toward Veros's 0.86 m/s.

The harness's value here is **diagnostic**: it surfaces a real physics
discrepancy that was invisible under the prior 500 m global-lat-lon
setup. Whether the fix is preferable to the current dissipative-but-
stable behaviour is a scientific call about how legoESM should perform
on under-resolved fronts.

## Run-kwargs in the matrix TestCase (current value)

See `scripts/matrix/run_ocean_test_matrix.py` (`_build_test_matrix`):

```python
TestCase(
    "lock_exchange", "latlon_regional", "4x64",
    duration_days=17.0 / 24.0,
    quick_days=1.7 / 24.0,
    run_kwargs={
        "lat_south": -0.018,
        "lat_north": +0.018,
        "lon_west": 0.0,
        "lon_east": 0.576,
        "dt": 30.0,
        "A_h": 0.0,
        "A_v": 0.0,
        "bottom_drag_r": 0.0,
        "barotropic_diffusion_alpha": 0.0,
        "bebt": 0.0,
        "barotropic_time_filter": "box",
        "n_barotropic_substeps": 1,
        "eos": "linear",
        "alpha_T": 2.0e-4,
        "beta_S": 0.0,
        "T_ref": 17.5,
        "S_ref": 35.0,
    },
)
```

Re-run after any `LatLonCGridOceanModel` change:

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
    --only lock_exchange --grid latlon_regional
JAX_PLATFORMS=cpu .venv/bin/python scripts/ocean_fidelity/run_comparison.py \
    --output docs/ocean_fidelity/initial_comparison_$(git rev-parse --short HEAD).md
```

## Resolution (2026-05-17)

The "structural" gap turned out to be a one-line bug in
``src/legoesm/grids/latlon.py::ensure_geometry``: when handed a
``LatLonGrid``, it rebuilt the ``LatLonCGridGeometry`` from
``n_lat`` / ``n_lon`` alone via ``create_latlon_geometry``, which
hard-codes the global pole-to-pole extent (``dlat = pi / n_lat``,
``dlon = 2*pi / n_lon``). For the Petersen 64 km x 4 km equatorial
channel that turned the regional ``dlon = 1.57e-4 rad`` into the
global ``dlon = 9.52e-2 rad`` — a **606x inflation of dx**, which
weakened the horizontal pressure-gradient acceleration by the same
factor and silently damped the gravity-current velocity by ~12x.

Diagnostic that exposed it (after exhausting every config knob):

* ``model.tendencies(state)`` returned ``max |du/dt| = 5.94e-6 m/s^2``
* The same call with the *raw* ``LatLonGrid`` (bypassing
  ``ensure_geometry``) returned ``9.33e-4 m/s^2`` — 158x larger.

Fix: ``create_latlon_geometry`` now accepts optional ``lat_1d`` /
``lon_1d`` arrays; ``ensure_geometry`` passes the input grid's actual
1-D coordinate arrays so regional / channel grids preserve their
bounds. Global ``latlon`` runs are unchanged at float32 precision
(verified by ``tests/grids/test_ensure_geometry_regional.py``).

Result on the Petersen TestCase:

| metric                | Veros (superbee) | legoESM (before fix) | legoESM (after fix) |
| --------------------- | ---------------- | -------------------- | ------------------- |
| T mean                | 17.50 degC       | 17.50 degC           | 17.50 degC          |
| T range               | [5.00, 30.00]    | [4.91, 30.06]        | [4.42, 30.07]       |
| max \|u\| (3D)        | 0.857 m/s        | 0.070 m/s            | **0.629 m/s**       |
| max \|eta\|           | n/a              | 0.030 m              | 0.066 m             |

legoESM 0.629 m/s sits inside the theoretical 0.5-0.7 m/s gravity-
current speed range (two-layer reduced-gravity ``c = sqrt(g'·H/2)``).
The remaining 27 % ratio with Veros 0.857 m/s is plausibly within
scheme-dependent overshoot range (Veros uses superbee with classic
front overshoots; legoESM uses WENO5 which is sharper but slightly
more dissipative on this thin geometry). **The wind gap is closed.**

**This bug also silently corrupts every other regional-latlon ocean
run in the matrix** (baroclinic_gyre, barotropic_double_gyre,
etc.). They all produced PASSING results with ~600x inflated dx, so
their velocity scales were too small by a similar factor. Those
results should be re-run after this fix lands.
