"""#1029 regression: lat-lon C-grid HYBRID pressure-gradient balance over
terrain (DCMIP 2012 §2-0-0 rest state).

An atmosphere initialised at rest over a mountain must stay at rest — the
horizontal pressure-gradient force ``-∇Φ - R_d T ∇ln p`` has to cancel exactly
against the terrain-following coordinate slope.

History (#1029). The hybrid path used to multiply the ``R_d T ∇ln p_s``
correction by a face-interpolated ``hybrid_factor = B·p_s/p`` — the analytic
*local* derivative of the arithmetic full-level pressure, which is not the
finite-difference secant of the nonlinear Simmons–Burridge ``Φ(p_s)`` that
``-∇Φ`` differences.  The two large terms did not cancel discretely over a
slope: this rest case reached 0.18 m/s in 200 steps (matrix 72x144/7 d:
4.3 m/s), and with Held-Suarez forcing (``held_suarez_topo``) the seed
amplified into the jet-runaway → NaN — the physics-free reproducer of the AMIP
latlon-topography instability.

Fix: the correction now differences the SB81 full-level log-pressure
``ln p_k = ln p_{k+1/2} - α_k`` (``sb81_full_level_ln_p``), built from the SAME
half-level construction the geopotential integrates, so the pair cancels
discretely at uniform-T rest (residual here: ~9e-12 m/s, ten orders below the
broken form).

Four tests (all MUST pass):
  * ``test_sigma_rest_over_topo_is_balanced`` — pure-sigma control (machine tol).
  * ``test_hybrid_A0_same_B_is_balanced`` — standard hybrid ``B`` with ``A=0``
    (the SB81 form reduces to the sigma correction there; machine tol).
  * ``test_hybrid_rest_over_topo_is_balanced`` — real (A≠0) hybrid levels; the
    former ``xfail(strict)`` reproducer, now the regression gate for the fix
    (machine tol under fp64).
  * ``test_hybrid_rest_over_topo_fp32_policy`` — same case at the production
    fp32 policy (few-mm/s gate; fp32 round-off dominates there).
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


# Small, deterministic config — a few seconds on CPU x64. The pre-fix
# imbalance was already visible here (exact rest → ~0.18 m/s in ~8 h); the full
# matrix case grew it to 4.3 m/s over 7 days.
_N_LAT, _N_LON, _NLEV = 36, 72, 20
_N_STEPS, _DT = 200, 150.0
_BALANCED_TOL_MS = 1e-6   # terrain-PGF cancels discretely; only fp64 round-off
#                           survives (measured: sigma ~0, hybrid ~9e-12 m/s)


def _face_max_wind_after_rest_run(coord) -> tuple[float, float, bool]:
    """(max(|u|,|v|) over faces at t=0, after _N_STEPS, all-finite) for
    rest-over-topo.

    ``coord`` is the vertical coordinate (hybrid or sigma); the DCMIP 2-0-0
    mountain (h_0=2000 m) is the same for both. The metric is the raw C-grid
    face max over BOTH wind components — a cell-average could hide a
    checkerboard face mode."""
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


def test_hybrid_rest_over_topo_is_balanced():
    """Real (A≠0) hybrid rest over the DCMIP 2-0-0 mountain stays at rest.

    The former #1029 ``xfail(strict)`` reproducer: with the SB81-consistent
    correction (``sb81_full_level_ln_p``) the discrete PGF pair cancels and
    this holds at the same machine tolerance as the sigma control (measured
    ~9e-12 m/s after 200 steps; the broken analytic-``hybrid_factor`` form
    reached 0.18 m/s)."""
    from legoesm.grids.vertical import standard_hybrid_levels
    v0, vN, finite = _face_max_wind_after_rest_run(standard_hybrid_levels(_NLEV))
    assert finite
    assert v0 == 0.0                    # starts at exact rest
    assert vN < _BALANCED_TOL_MS, (
        f"hybrid rest-over-topography drifted to {vN:.3e} m/s "
        f"(> {_BALANCED_TOL_MS} m/s) — #1029 SB81 PGF regression")


def test_hybrid_rest_over_topo_fp32_policy():
    """The SB81 balance must also hold usefully under the DEFAULT fp32 policy.

    fp32 cannot reach machine rest (the PGF differences two ~1e6 m^2/s^2
    terms; log/cumsum round-off seeds ~1e-7 m/s^2 accelerations), but the
    discrete consistency still keeps the 200-step drift at the few-mm/s level
    (measured ~4.3e-3 m/s) — 40x below the broken analytic-factor form
    (0.18 m/s). Gate at 0.02 m/s: catches a revert at fp32 while leaving 5x
    headroom over the measured value.
    """
    # No fp64 fixture override here: run under whatever the default policy is
    # (fp32 in production). The autouse fixture pins fp64, so explicitly set
    # fp32 for this one test and restore via the fixture teardown ordering.
    from legoesm.core.precision import set_policy, PrecisionPolicy
    from legoesm.grids.vertical import standard_hybrid_levels
    set_policy(PrecisionPolicy.fp32())
    v0, vN, finite = _face_max_wind_after_rest_run(standard_hybrid_levels(_NLEV))
    assert finite
    assert v0 == 0.0
    assert vN < 0.02, (
        f"fp32 hybrid rest-over-topography drifted to {vN:.3e} m/s (> 0.02) — "
        f"#1029 SB81 PGF regression at production precision")


def test_sb81_omega_conversion_rest_balanced_and_gate_live():
    """#1029 ω-side opt-in flag: (a) with sb81_omega_conversion=True the
    rest-over-topo state STAYS at rest (at exact rest the flux-form mass
    divergence is exactly zero, so both conversion forms vanish — the SB81
    swap cannot disturb a balanced column); (b) the gate is LIVE: from a
    perturbed (divergent) state one step under each flag value produces
    DIFFERENT temperatures (a silently-dead flag would be the dispatch
    footgun CLAUDE.md forbids), both finite."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import standard_hybrid_levels
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
    coord = standard_hybrid_levels(_NLEV)
    state0 = hydrostatic_to_cgrid(
        rest_state_topography_init_latlon(grid, coord, h_0=2000.0), grid)

    # (a) rest balance with the SB81 conversion ON (short run suffices —
    # the pre-#1215 imbalance was visible within a few steps).
    cfg_on = CGridLatLonPrimitiveEquationConfig(
        A_h=0.0, fix_mass=True, anchor_mass_to_initial=True,
        use_polar_filter=True, sb81_omega_conversion=True,
    )
    model_on = CGridLatLonPrimitiveEquationModel(grid, coord, cfg_on, dt=_DT)
    s = state0
    for _ in range(20):
        s = model_on.step(s, _DT)
    v_rest = float(jnp.maximum(jnp.max(jnp.abs(s.u)), jnp.max(jnp.abs(s.v))))
    assert bool(jnp.all(jnp.isfinite(s.T)))
    assert v_rest < _BALANCED_TOL_MS, (
        f"SB81 ω-conversion disturbed the balanced rest state: {v_rest:.3e} "
        f"m/s (must be fp64 round-off level)")

    # (b) gate liveness on a divergent state.
    cfg_off = CGridLatLonPrimitiveEquationConfig(
        A_h=0.0, fix_mass=True, anchor_mass_to_initial=True,
        use_polar_filter=True, sb81_omega_conversion=False,
    )
    model_off = CGridLatLonPrimitiveEquationModel(grid, coord, cfg_off, dt=_DT)
    u_wave = jnp.sin(jnp.linspace(0.0, 2.0 * jnp.pi, state0.u.shape[1]))
    u_pert = state0.u.at[:, :, _NLEV // 2].add(u_wave[None, :])
    state_pert = state0._replace(u=u_pert)
    s_on = model_on.step(state_pert, _DT)
    s_off = model_off.step(state_pert, _DT)
    assert bool(jnp.all(jnp.isfinite(s_on.T)))
    assert bool(jnp.all(jnp.isfinite(s_off.T)))
    dT = float(jnp.max(jnp.abs(s_on.T - s_off.T)))
    assert dT > 0.0, (
        "sb81_omega_conversion flag is DEAD — on/off produced identical "
        "temperatures on a divergent state")
