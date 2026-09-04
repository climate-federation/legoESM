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
# The stratified variant differences two O(1e6 m^2/s^2) geopotential terms
# against a temperature integral that is no longer identically zero, so its
# fp64 round-off floor is higher than the isothermal case's. Set from the
# MEASURED sigma-control value rather than guessed; the diagnostic quantity
# here is orders of magnitude, not the last digit.
_STRATIFIED_TOL_MS = 1e-3
_BALANCED_TOL_MS = 1e-6   # terrain-PGF cancels discretely; only fp64 round-off
#                           survives (measured: sigma ~0, hybrid ~9e-12 m/s)


def _face_max_wind_after_rest_run(coord, *, stratified: bool = False,
                                  n_steps: int = _N_STEPS, h_0: float = 2000.0
                                  ) -> tuple[float, float, bool]:
    """(max(|u|,|v|) over faces at t=0, after _N_STEPS, all-finite) for
    rest-over-topo.

    ``coord`` is the vertical coordinate (hybrid or sigma); the DCMIP 2-0-0
    mountain (h_0=2000 m) is the same for both. The metric is the raw C-grid
    face max over BOTH wind components — a cell-average could hide a
    checkerboard face mode.

    ``stratified`` swaps the ISOTHERMAL DCMIP rest state for the exact
    constant-lapse-rate one over the SAME mountain — still an exact hydrostatic
    state at rest, but with a temperature that varies along the terrain-
    following coordinate surfaces, so the temperature-gradient part of the PGF
    is exercised instead of being identically zero (#1029)."""
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
    if stratified:
        from tests.test_cases.dcmip2012.rest_state_topography import (
            lapse_rate_state_topography_init_latlon,
        )
        init = lapse_rate_state_topography_init_latlon(grid, coord, h_0=h_0)
    else:
        init = rest_state_topography_init_latlon(grid, coord, h_0=h_0)
    state = hydrostatic_to_cgrid(init, grid)

    def face_max(s):
        return float(jnp.maximum(jnp.max(jnp.abs(s.u)), jnp.max(jnp.abs(s.v))))

    v0 = face_max(state)
    for _ in range(n_steps):
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


@pytest.mark.xfail(strict=True, reason=
    "#1029: the temperature-gradient term of the terrain PGF does not "
    "cancel discretely; drift is linear in mountain height and 11 orders "
    "above the isothermal case. strict=True so this flips to a loud "
    "XPASS the moment it is fixed.")
def test_sigma_stratified_rest_over_topo_is_balanced():
    """#1029: a STRATIFIED rest state over the mountain must also stay
    at rest, on the sigma control.

    The four tests above all use the isothermal DCMIP state, where grad(T) is
    identically zero, so they certify only the -grad(Phi) vs surface-pressure
    half of the terrain PGF.  The temperature-gradient integral -- the half
    that switches on the moment the atmosphere is stratified, i.e. in every
    real run -- is untested by them.  This is the same exact-rest problem with
    a constant lapse rate, so the correct answer is unchanged and any drift is
    a discretisation inconsistency in that term.
    """
    from legoesm.grids.vertical import create_sigma_coordinate
    v0, vN, finite = _face_max_wind_after_rest_run(
        create_sigma_coordinate(_NLEV), stratified=True)
    assert finite
    assert v0 == 0.0
    assert vN < _STRATIFIED_TOL_MS, (
        f"sigma STRATIFIED rest-over-topo drifted to {vN:.3e} m/s — the "
        f"temperature-gradient term of the terrain PGF does not cancel "
        f"discretely (#1029)")


@pytest.mark.xfail(strict=True, reason=
    "#1029: the temperature-gradient term of the terrain PGF does not "
    "cancel discretely; drift is linear in mountain height and 11 orders "
    "above the isothermal case. strict=True so this flips to a loud "
    "XPASS the moment it is fixed.")
def test_hybrid_stratified_rest_over_topo_is_balanced():
    """#1029: the same stratified rest state on the REAL hybrid levels.

    This is the coordinate the failing `held_suarez_topo` case runs, and the
    one whose isothermal twin was the original #1029 reproducer.

    NB the sigma and hybrid numbers are NOT a one-variable comparison — the two
    coordinates place their levels differently, so which is "worse" is not
    attributable. What both establish is that the defect is present on each,
    i.e. it is not confined to the hybrid A_half machinery.
    """
    from legoesm.grids.vertical import standard_hybrid_levels
    v0, vN, finite = _face_max_wind_after_rest_run(
        standard_hybrid_levels(_NLEV), stratified=True)
    assert finite
    assert v0 == 0.0
    assert vN < _STRATIFIED_TOL_MS, (
        f"hybrid STRATIFIED rest-over-topo drifted to {vN:.3e} m/s — the "
        f"temperature-gradient term of the terrain PGF does not cancel "
        f"discretely (#1029)")


def test_stratified_state_really_has_a_horizontal_temperature_gradient():
    """Non-vacuity: the stratified state must actually differ from the
    isothermal one ALONG a coordinate surface.

    Without this, both tests above could pass simply by reproducing the
    isothermal case (for which the balance is already known to hold), and
    would certify nothing.
    """
    import jax.numpy as jnp
    import numpy as np
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import standard_hybrid_levels
    from tests.test_cases.dcmip2012.rest_state_topography import (
        lapse_rate_state_topography_init_latlon,
        rest_state_topography_init_latlon,
    )

    grid = create_latlon_grid(_N_LAT, _N_LON)
    coord = standard_hybrid_levels(_NLEV)
    iso = rest_state_topography_init_latlon(grid, coord, h_0=2000.0)
    strat = lapse_rate_state_topography_init_latlon(grid, coord, h_0=2000.0)

    T_iso = np.asarray(iso.T.data)
    T_str = np.asarray(strat.T.data)
    # The isothermal state is flat by construction; the stratified one must
    # vary BOTH with height and, on a terrain-following surface, horizontally.
    assert float(T_iso.max() - T_iso.min()) == 0.0
    assert float(T_str.max() - T_str.min()) > 10.0
    lowest = T_str[..., -1]
    assert float(lowest.max() - lowest.min()) > 1.0, (
        "the stratified state has no horizontal temperature contrast on the "
        "lowest coordinate surface — the term under test would be inactive")
    # And it must still start at EXACT rest, or the drift metric is meaningless.
    assert float(jnp.max(jnp.abs(strat.u.data))) == 0.0
    assert float(jnp.max(jnp.abs(strat.v.data))) == 0.0


def test_stratified_drift_vanishes_without_terrain():
    """THE CONTROL for the two tests above (#1029).

    A stratified initial state is only APPROXIMATELY in discrete hydrostatic
    balance -- the model integrates the hydrostatic equation on its own levels,
    and that discrete integral does not reproduce the analytic constant-lapse-
    rate profile exactly.  That truncation alone would produce a drift with no
    terrain involved, and quoting it as a terrain-PGF error would be an
    instrument bug reported as physics.

    With ``h_0 = 0`` the mountain is gone and every coordinate surface is flat,
    so any residual here is the IC's own vertical truncation and NOTHING to do
    with the terrain pressure gradient.  It must be far below the sloped
    result for the sloped result to mean anything.
    """
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    for label, coord in (("sigma", create_sigma_coordinate(_NLEV)),
                         ("hybrid", standard_hybrid_levels(_NLEV))):
        v0, vN, finite = _face_max_wind_after_rest_run(
            coord, stratified=True, h_0=0.0)
        assert finite
        assert v0 == 0.0
        assert vN < _STRATIFIED_TOL_MS, (
            f"{label} STRATIFIED rest with NO terrain drifted to {vN:.3e} m/s "
            f"-- the initial state is not in discrete hydrostatic balance, so "
            f"the sloped result cannot be attributed to the terrain PGF")


def test_stratified_drift_scales_with_terrain_slope():
    """The mechanism must respond to the variable it is blamed on.

    A terrain pressure-gradient inconsistency is driven by the slope of the
    coordinate surfaces, so halving the mountain height must materially reduce
    the drift.  A drift that ignores ``h_0`` is not a terrain-PGF error however
    well it correlates with having a mountain present.
    """
    from legoesm.grids.vertical import create_sigma_coordinate
    coord = create_sigma_coordinate(_NLEV)
    _, v_full, ok_full = _face_max_wind_after_rest_run(
        coord, stratified=True, h_0=2000.0)
    _, v_half, ok_half = _face_max_wind_after_rest_run(
        coord, stratified=True, h_0=1000.0)
    assert ok_full and ok_half
    assert v_full > 10.0 * _STRATIFIED_TOL_MS, (
        "the full-height case does not drift enough for this scaling test to "
        "discriminate anything")
    assert v_half < 0.75 * v_full, (
        f"halving the mountain (2000 m -> 1000 m) left the drift essentially "
        f"unchanged ({v_full:.3e} -> {v_half:.3e} m/s): the terrain slope is "
        f"not what drives it, so the terrain-PGF attribution is REFUTED")


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
