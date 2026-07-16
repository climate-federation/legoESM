"""#1029 reproducer: lat-lon C-grid HYBRID pressure-gradient force is not
balanced over terrain.

An atmosphere initialised at rest over a mountain (DCMIP 2012 §2-0-0) must stay
at rest — the horizontal pressure-gradient force ``-∇Φ - R_d T ∇ln p_s`` has to
cancel exactly against the terrain-following coordinate slope. On the lat-lon
C-grid **hybrid** coordinate it does NOT: the rest state spuriously accelerates
and the spurious wind grows monotonically (full matrix case 72x144/7 d reaches
max|v| ≈ 4.3 m/s, vs icosahedral 0.0 and cubed-sphere 1.1). With Held-Suarez
forcing on the same mountain (``held_suarez_topo``, which the matrix runs on the
hybrid coordinate) this seed amplifies into a jet runaway → NaN — the
physics-free reproducer of the AMIP latlon-topography instability.

Mechanism (localised this session). The hybrid path multiplies the
``R_d T ∇ln p_s`` correction by a face-interpolated ``hybrid_factor = B·p_s/p``
(``primitive_eq_latlon_cgrid`` step 6): a discrete *local-derivative* average
that is not the finite-difference (secant) derivative of the nonlinear
Simmons–Burridge geopotential ``Φ(p_s)`` the ``-∇Φ`` term differences (the
correction even uses the arithmetic full-level pressure, not how ``Φ`` is
constructed). So the two large terms do not cancel discretely over a slope. (T
is uniform at rest, so this is NOT a temperature-interpolation issue.) Two
controls bracket it: pure sigma shows terrain balance, and the SAME standard
hybrid ``B`` coefficients with ``A`` zeroed also stay balanced, so the imbalance
is implicated in the ``A_half`` coefficient and the level structure it produces
(``A_half`` makes the ``B·p_s/p`` correction nontrivial — it equals 1 when A=0,
so the correction reduces to the exact sigma form) — not the sigma/hybrid code
path or the time loop.
(Zeroing ``A`` also shifts the physical levels, so this implicates ``A_half``
rather than isolating it from its level structure.)

Three tests:
  * ``test_sigma_rest_over_topo_is_balanced`` — pure-sigma coordinate: shows the
    terrain pressure-gradient cancels on the sigma path. MUST pass.
  * ``test_hybrid_A0_same_B_is_balanced`` — the standard hybrid ``B`` with
    ``A=0``: same ``B`` coefficients, only ``A_half`` differs, and it stays
    balanced — implicating ``A_half`` / its level structure. MUST pass.
  * ``test_hybrid_rest_over_topo_pgf_imbalance`` — real (A≠0) hybrid levels.
    ``xfail(strict)``: fails today; when the hybrid PGF correction is made
    discretely consistent with ``Φ(p_s)`` this flips to XPASS and the marker
    must be removed.
"""
from __future__ import annotations

import pytest

# Pin fp64 for the whole module (and restore after): the hybrid levels /
# geopotential take their dtype from the global PrecisionPolicy, whose default
# is fp32 — under which the spurious-wind threshold would move with test order /
# platform. set_policy(fp64) also enables JAX x64.
from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy


@pytest.fixture(autouse=True)
def _fp64_policy():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


# Small, deterministic config — a few seconds on CPU x64. The imbalance is
# already visible here (exact rest → ~0.18 m/s in ~8 h); the full matrix run
# grows it to 4.3 m/s over 7 days.
_N_LAT, _N_LON, _NLEV = 36, 72, 20
_N_STEPS, _DT = 200, 150.0
# A balanced PGF keeps the rest state at ~0. The hybrid case currently reaches
# ~0.18 m/s here; 0.05 m/s cleanly separates "broken now" from a future fix that
# reaches the sigma-path balance.
_REST_TOL_MS = 0.05
_BALANCED_TOL_MS = 1e-6   # sigma terrain-PGF cancels exactly; only fp64 round-off survives


def _face_max_wind_after_rest_run(coord) -> tuple[float, float, bool]:
    """(native face max|v| at t=0, after _N_STEPS, all-finite) for rest-over-topo.

    ``coord`` is the vertical coordinate (hybrid or sigma); the DCMIP 2-0-0
    mountain (h_0=2000 m) is the same for both. The metric is the raw C-grid
    face max of |u| and |v| — a cell-average could hide a checkerboard face
    mode."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )
    from tests.test_cases.dcmip2012.rest_state_topography import (
        rest_state_topography_init_latlon,
    )
    import jax.numpy as jnp

    grid = create_latlon_grid(_N_LAT, _N_LON)
    cfg = CGridLatLonPrimitiveEquationConfig(
        A_h=0.0,                       # no diffusion masking the imbalance
        fix_mass=True, anchor_mass_to_initial=True,
        use_polar_filter=True,
    )
    model = CGridLatLonPrimitiveEquationModel(grid, coord, cfg, dt=_DT)
    state = hydrostatic_to_cgrid(
        rest_state_topography_init_latlon(grid, coord, h_0=2000.0), grid)

    def face_max(s):
        return float(jnp.maximum(jnp.max(jnp.abs(s.u)), jnp.max(jnp.abs(s.v))))

    v0 = face_max(state)
    for _ in range(_N_STEPS):
        state = model.step(state, _DT)
    finite = bool(
        jnp.all(jnp.isfinite(state.u)) & jnp.all(jnp.isfinite(state.v))
        & jnp.all(jnp.isfinite(state.T)) & jnp.all(jnp.isfinite(state.p_s)))
    return v0, face_max(state), finite


def test_sigma_rest_over_topo_is_balanced():
    """Pure-sigma coordinate: terrain PGF cancels — rest state stays at rest.

    Shows the sigma path is balanced (same model, mountain, time loop; only the
    coordinate differs) — the harness is not the source of #1029."""
    from legoesm.grids.vertical import create_sigma_coordinate
    v0, vN, finite = _face_max_wind_after_rest_run(create_sigma_coordinate(_NLEV))
    assert finite
    assert v0 == 0.0
    assert vN < _BALANCED_TOL_MS, (
        f"sigma rest-over-topo drifted to {vN:.3e} m/s (should be ~0) — the "
        f"harness/time loop is not the source of #1029")


def test_hybrid_A0_same_B_is_balanced():
    """Standard hybrid B coefficients with A zeroed — implicates A_half.

    Same B_half and same hybrid code path as the failing case; only A_half
    differs (0 vs the standard pressure-based values, which also shifts the
    physical levels). It stays at machine rest, so the imbalance is implicated in
    A_half / the level structure it produces, not the sigma/hybrid code path or
    the time loop."""
    import jax.numpy as jnp
    from legoesm.grids.vertical import (
        standard_hybrid_levels, create_hybrid_coordinate)
    std = standard_hybrid_levels(_NLEV)
    coord = create_hybrid_coordinate(_NLEV, jnp.zeros_like(std.A_half), std.B_half)
    v0, vN, finite = _face_max_wind_after_rest_run(coord)
    assert finite
    assert v0 == 0.0
    assert vN < _BALANCED_TOL_MS, (
        f"A=0 (standard-B) hybrid rest-over-topo drifted to {vN:.3e} m/s "
        f"(should be ~0) — the hybrid code path is not exact even at A=0")


@pytest.mark.xfail(strict=True, reason="#1029: lat-lon C-grid HYBRID PGF "
                   "correction incompatible with Simmons-Burridge Φ(p_s) — rest "
                   "state spuriously accelerates over terrain")
def test_hybrid_rest_over_topo_pgf_imbalance():
    """Hybrid rest over the DCMIP 2-0-0 mountain must stay at rest; today it does
    not. When the hybrid PGF correction is made discretely consistent with
    ``Φ(p_s)`` this passes and the ``xfail`` must be removed."""
    from legoesm.grids.vertical import standard_hybrid_levels
    v0, vN, finite = _face_max_wind_after_rest_run(standard_hybrid_levels(_NLEV))
    assert finite                       # no blow-up at this short horizon
    assert v0 == 0.0                    # starts at exact rest
    assert vN < _REST_TOL_MS, (
        f"hybrid rest-over-topography spuriously accelerated to {vN:.3f} m/s "
        f"(> {_REST_TOL_MS} m/s) — latlon hybrid PGF imbalance (#1029)")
