"""The split-explicit barotropic solver must propagate gravity waves at
sqrt(gH) -- and the averaging window is what decides whether it does.

THE DEFECT THIS PINS (found 2026-08-12, codex CRITICAL)
-------------------------------------------------------
A split-explicit free surface runs ``n_substeps`` barotropic substeps and
hands the baroclinic step a TIME-AVERAGED state. ``compute_filter_weights``
built that average over ``i = 0 .. n-1``, i.e. over ``[t, t+dt]``, with the
box/cosine kernel centred on the middle of that range. The averaged state
therefore represented ``t + dt/2`` but was stored as the state at ``t + dt``:
every step advanced the free surface by half a step of phase.

MEASURED on the inertia-gravity-wave channel before the fix -- the model
period over the exact period:

    cosine  1.862      box  1.921
    nemo_ab3am4  1.000     implicit_cn  1.000   (neither averages)

and it got WORSE with more substeps (30 -> 1.862, 120 -> 1.951, 480 ->
1.971), converging to 2 rather than to 1, which is the signature of a fixed
half-step phase deficit rather than of a CFL or filter artefact.

The fix runs the window over ``i+1 = 1 .. 2n-1`` so it is centred on
``t + dt``. That is what NEMO's centred boxcar and the SM2005 power-law
window already did in this same module; box/cosine were the odd ones out.

WHY A WAVE-SPEED TEST AND NOT ONLY A WEIGHTS TEST
-------------------------------------------------
``test_barotropic_common`` checks the weights' arithmetic, and the centroid
assertion below is the cheap direct statement of the bug. But the weights
were arithmetically self-consistent BEFORE the fix too -- they summed to
one and satisfied the continuity identity -- so no property of the weights
in isolation could have caught this. It takes a wave whose speed is known
analytically. Hence the end-to-end measurement here, which is the test the
suite was missing (codex 2026-08-12: the existing tier-1 metric test
fabricates its own Hovmoller and the NEMO filter test is a transcription of
our own code, so neither is an independent oracle for wave speed).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# THIS FILE REQUIRES float64, EXPLICITLY (CLAUDE.md: say so, do not enable it
# silently). Two reasons, both measured: the window-centroid and transport
# identities are asserted at 1e-12..1e-14, which fp32 cannot represent; and the
# convergence study divides two L2 norms that differ by ~4x, where fp32 noise
# would show up directly in the fitted order. Run under fp32 the file
# deterministically fails six tests (codex 2026-08-12), so it enables x64 at
# import rather than relying on the caller passing JAX_ENABLE_X64=1.
import jax
jax.config.update("jax_enable_x64", True)

jnp = pytest.importorskip("jax.numpy")

if jnp.zeros(1, dtype=jnp.float64).dtype != jnp.float64:      # pragma: no cover
    pytest.skip("x64 unavailable; this file is meaningless in fp32",
                allow_module_level=True)

_REPO = Path(__file__).resolve().parents[3]


def _matrix():
    """The ocean matrix module, for its channel-mode definition.

    Imported rather than re-derived: the analytic mode, its dispersion
    relation and the channel geometry must be the SAME ones the matrix case
    uses, or this test measures a different problem than the one it claims
    to.
    """
    if "_rm_wave" in sys.modules:
        return sys.modules["_rm_wave"]
    p = _REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py"
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("_rm_wave", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rm_wave"] = mod
    spec.loader.exec_module(mod)
    return mod


#: Coarse enough to run as a unit test, fine enough that the discretisation
#: error is far below the effect (the defect is a factor ~1.9; the spatial
#: error here is a few percent).
_NLAT, _NLON = 12, 24
_PERIODS = 0.75


def _run_channel(solver: str, time_filter: str, *, n_lat: int = _NLAT,
                 n_lon: int = _NLON, n_substeps: int | None = None,
                 dt_fixed: float | None = None, theta: float | None = None,
                 coriolis: str | None = None, outer: str | None = None,
                 slow_ab2: bool = False, tides: bool = False,
                 t_seconds: float | None = None, n_steps_max: int | None = None,
                 periods: float = _PERIODS):
    """Integrate the channel eigenmode and report how well it was carried.

    Returns ``{"ratio": T_model/T_exact, "l2": relative L2 in eta at the end,
    "dt": s, "dx": m}``. ``ratio == 1`` means the right wave speed; ``l2``
    is the accuracy measure the convergence study refines.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanConfig, LatLonCGridOceanModel)
    from legoesm.ocean.init_latlon_cgrid import (
        rest_state_latlon_cgrid_ocean)
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.core.field import Field
    from legoesm.grids.latlon import create_beta_plane_cgrid_geometry

    M = _matrix()
    _lx, _ly, k, _l, omega, period = M._igw_channel_params(n_lat, n_lon)
    x_c, y_c, x_u, y_v = M._igw_channel_coords(n_lat, n_lon)
    dx_m, dy_m = M._IGWC_LX_M / n_lon, M._IGWC_LY_M / n_lat

    grid = create_beta_plane_cgrid_geometry(
        n_lat, n_lon, dx_m=dx_m, dy_m=dy_m, f0=M._IGWC_F0, beta=0.0,
        cartesian_pseudo_lat=True)
    z_coord = create_ocean_z_star(n_levels=1, H_max=M._IGWC_H)

    bt = {"barotropic_solver": solver, "barotropic_time_filter": time_filter}
    if n_substeps is not None:
        bt["n_barotropic_substeps"] = n_substeps
    if slow_ab2:
        bt["barotropic_slow_forcing_ab2"] = True
    if theta is not None:
        bt["barotropic_implicit_theta_eta"] = theta
        bt["barotropic_implicit_theta_pgf"] = theta
    config = LatLonCGridOceanConfig()
    config = config._replace(barotropic=config.barotropic._replace(**bt))
    if tides:
        config = config._replace(
            tidal_forcing=config.tidal_forcing._replace(enabled=True))
    if coriolis is not None:
        config = config._replace(coriolis_scheme=coriolis)
    if outer is not None:
        config = config._replace(outer_integrator=outer)
    model = LatLonCGridOceanModel(grid, z_coord, config)

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=M._IGWC_H,
        land_mask_override=np.ones((n_lat, n_lon)))
    # Each field sampled at ITS OWN stagger point, as the matrix case does.
    xc2, yc2 = np.meshgrid(x_c, y_c)
    xu2, yu2 = np.meshgrid(x_u, y_c)
    xv2, yv2 = np.meshgrid(x_c, y_v)
    eta0 = M.igw_channel_mode(xc2, yc2, 0.0, n_lat, n_lon)[0]
    u0 = M.igw_channel_mode(xu2, yu2, 0.0, n_lat, n_lon)[1]
    v0 = M.igw_channel_mode(xv2, yv2, 0.0, n_lat, n_lon)[2]
    state = state._replace(
        eta=Field(jnp.asarray(eta0)), u=Field(jnp.asarray(u0[..., None])),
        v=Field(jnp.asarray(v0[..., None])))
    if slow_ab2:
        # The AB2 extrapolation needs a seeded pytree carry (zeros Fields).
        state = state._replace(
            F_slow_u_prev=Field(jnp.zeros_like(state.u.data[..., 0]),
                                name="F_slow_u_prev"),
            F_slow_v_prev=Field(jnp.zeros_like(state.v.data[..., 0]),
                                name="F_slow_v_prev"))

    dt = dt_fixed if dt_fixed is not None else (
        M._IGWC_CFL_S_PER_M * min(dx_m, dy_m))
    t_final = periods * period
    n_steps = max(4, int(round(t_final / dt)))
    dt = t_final / n_steps                     # land exactly on t_final
    if n_steps_max is not None:
        n_steps = min(n_steps, n_steps_max)    # probes that need only a step

    # Same projection the matrix case uses: the mode shape in y (walled, so
    # not a Fourier harmonic) times cos/sin(kx); arctan2 of the two is the
    # phase, and its slope in time is omega.
    y_shape = M.igw_channel_mode(np.zeros_like(yc2), yc2, 0.0,
                                 n_lat, n_lon)[0]
    p_cos, p_sin = y_shape * np.cos(k * xc2), y_shape * np.sin(k * xc2)

    t_samp, ph = [], []
    s = state
    every = max(1, n_steps // 12)
    for n in range(n_steps + 1):
        if n % every == 0:
            e = np.asarray(s.eta.data, dtype=np.float64)
            ph.append(np.arctan2(float(np.sum(e * p_sin)),
                                 float(np.sum(e * p_cos))))
            t_samp.append(n * dt)
        if n < n_steps:
            s = (model.step(s, dt) if t_seconds is None
                 else model.step(s, dt, t_seconds=t_seconds + n * dt))
    ph = np.unwrap(np.asarray(ph))
    omega_fit = abs(float(np.polyfit(np.asarray(t_samp), ph, 1)[0]))
    assert np.isfinite(omega_fit) and omega_fit > 0.0, (
        f"phase fit degenerate for {solver}/{time_filter}")

    eta_end = np.asarray(s.eta.data, dtype=np.float64)
    eta_exact = M.igw_channel_mode(xc2, yc2, t_final, n_lat, n_lon)[0]
    l2 = float(np.sqrt(np.mean((eta_end - eta_exact) ** 2)) /
               max(float(np.sqrt(np.mean(eta_exact ** 2))), 1e-30))
    return {"ratio": float(omega / omega_fit), "l2": l2, "dt": dt,
            "dx": min(dx_m, dy_m)}


def _measure_period_ratio(solver: str, time_filter: str,
                          n_substeps: int | None = None) -> float:
    """Model period / exact period. 1.0 == the right wave speed."""
    return _run_channel(solver, time_filter, n_substeps=n_substeps)["ratio"]


# ---------------------------------------------------------------------------
# The cheap, direct statement of the bug: where the averaging window sits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_cosine", [False, True])
@pytest.mark.parametrize("n", [2, 3, 10, 30, 120])
def test_averaging_window_is_centred_on_the_end_of_the_step(n, use_cosine):
    """Substep ``i`` yields the state at ``t + (i+1)*dt_s``, so the weighted
    mean time of the average must be ``t + n*dt_s`` -- the END of the step,
    not its middle. This single number is the whole defect."""
    from legoesm.ocean.dynamics.barotropic_common import (
        compute_filter_weights)
    w, _w_total, _w_tr, n_loop = compute_filter_weights(
        n, jnp.float64, use_cosine=use_cosine)
    w = np.asarray(w)
    centroid = float((w * np.arange(1, n_loop + 1)).sum() / w.sum())
    assert centroid == pytest.approx(float(n), rel=1e-12), (
        f"window centred on substep {centroid:.2f} of {n}; a window centred "
        f"on n/2 returns a mid-step state as the end-of-step state and "
        f"halves the gravity-wave speed")
    assert n_loop == 2 * n - 1
    assert (w > 0.0).all(), "a zero weight wastes a substep; a negative one " \
                            "is not an average"


@pytest.mark.parametrize("use_cosine", [False, True])
def test_transport_weights_still_close_continuity(use_cosine):
    """The window moved; the continuity identity must not.

    ``div(Hu_avg) == (eta_old - eta_avg)/dt`` requires
    ``w_transport[j] = tail_j / (n_substeps * w_total)`` with the PHYSICAL
    ``n_substeps`` in the denominator -- ``dt`` is still ``n_substeps*dt_s``
    however far past ``t+dt`` the window reaches. Using ``n_loop`` there
    would rescale every transport by ~2 (codex 2026-08-12).
    """
    n = 24
    from legoesm.ocean.dynamics.barotropic_common import (
        compute_filter_weights)
    w, w_total, w_tr, _n_loop = compute_filter_weights(
        n, jnp.float64, use_cosine=use_cosine)
    w = np.asarray(w)
    tail = np.cumsum(w[::-1])[::-1]
    np.testing.assert_allclose(np.asarray(w_tr), tail / (n * float(w_total)),
                               rtol=1e-13)


def test_the_centroid_check_can_fail():
    """Non-vacuity: the OLD window, reconstructed, must trip the assertion
    above. Without this the centroid test could be passing on a tautology."""
    n = 30
    i = np.arange(n)
    for w in (np.ones(n),
              1.0 + np.cos(2.0 * np.pi * (i - 0.5 * n) / n)):
        centroid = float((w * np.arange(1, n + 1)).sum() / w.sum())
        assert abs(centroid - n) > 0.4 * n, (
            "the pre-fix window is supposed to sit near the middle of the "
            "step; if it does not, this test no longer reproduces the bug")


# ---------------------------------------------------------------------------
# The end-to-end measurement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("time_filter", ["box", "cosine"])
def test_split_explicit_propagates_at_the_right_speed(time_filter):
    """The measurement the weights alone cannot make.

    Pre-fix this returned 1.92 (box) / 1.86 (cosine). The tolerance is set
    by the SPATIAL discretisation at this coarse resolution, not by the
    time filter, and is nowhere near the factor ~1.9 being excluded.
    """
    ratio = _measure_period_ratio("explicit_substep", time_filter)
    assert ratio == pytest.approx(1.0, abs=0.15), (
        f"{time_filter} filter propagates the external gravity wave with "
        f"period {ratio:.3f}x the exact one")


def test_the_error_does_not_grow_with_substep_count():
    """The signature that told us it was a window offset and not a CFL or
    damping artefact: pre-fix the ratio moved 1.86 -> 1.95 -> 1.97 toward 2
    as substeps increased, i.e. refining the barotropic solver made the wave
    WORSE. A converging scheme must do the opposite."""
    few = _measure_period_ratio("explicit_substep", "cosine", n_substeps=12)
    many = _measure_period_ratio("explicit_substep", "cosine", n_substeps=48)
    assert abs(many - 1.0) <= abs(few - 1.0) + 0.02, (
        f"more substeps made the wave speed worse ({few:.3f} -> {many:.3f}); "
        f"that is the averaging-window signature, not convergence")


def test_it_agrees_with_the_solver_that_never_had_the_bug():
    """``implicit_cn`` does no time averaging and measured 1.000 throughout.
    It is the independent oracle for the arm under test."""
    ref = _measure_period_ratio("implicit_cn", "cosine")
    assert ref == pytest.approx(1.0, abs=0.15), (
        f"the reference solver itself is off ({ref:.3f}); the comparison "
        f"below is meaningless until that is explained")
    split = _measure_period_ratio("explicit_substep", "cosine")
    assert split == pytest.approx(ref, abs=0.10), (
        f"split-explicit {split:.3f} vs implicit_cn {ref:.3f}")


# ---------------------------------------------------------------------------
# Convergence order -- the SECOND defect, settled 2026-08-12
# ---------------------------------------------------------------------------
#
# The channel case recorded "convergence is ~1.2, not 2" as an open finding and
# guessed that coriolis_scheme="matsuno_split" was the first-order term. That
# guess was WRONG. What the measurements below pin instead:
#
#   space, dt fixed .................... 1.82   (second order, as designed)
#   time, grid fixed ................... 0.99   (FIRST order)
#   lagged nonlinear thickness ......... refuted -- the error is unchanged to
#                                        5 digits at 100x smaller amplitude
#   default lane, dt ~ dx .............. 1.31
#   fully second-order lane, dt ~ dx ... 1.95
#
# The three second-order settings are GATED IN A CHAIN --
# barotropic_slow_forcing_ab2 requires coriolis_scheme="explicit_ab2", which
# requires outer_integrator="ab2" -- so changing any ONE of them moves nothing
# and every single-knob experiment returns a null result. That is why this
# stood open: the earlier probes each turned one knob.

def _order(coarse, fine):
    """Observed order from a 2x refinement pair."""
    return float(np.log(coarse["l2"] / fine["l2"]) / np.log(2.0))


def _pair(**kw):
    return (_run_channel("implicit_cn", "cosine", n_lat=20, n_lon=40,
                         theta=0.5, **kw),
            _run_channel("implicit_cn", "cosine", n_lat=40, n_lon=80,
                         theta=0.5, **kw))


def test_space_is_second_order():
    """Refine dx with dt held FIXED: this isolates the spatial operator, and
    it is the control that says the C-grid discretisation is not the
    problem."""
    c, f = _pair(dt_fixed=30.0)
    assert _order(c, f) > 1.6, (
        f"spatial order {_order(c, f):.2f}; the C-grid should be ~2")


def test_time_is_first_order_on_the_default_lane():
    """The defect, stated directly. NOT xfail: first order is what the
    default lane is, and this test exists so that a change to it is
    noticed."""
    prev, orders = None, []
    for dt in (480.0, 240.0):
        r = _run_channel("implicit_cn", "cosine", n_lat=40, n_lon=80,
                         theta=0.5, dt_fixed=dt)
        if prev is not None:
            orders.append(float(np.log(prev / r["l2"]) / np.log(2.0)))
        prev = r["l2"]
    assert 0.8 < orders[0] < 1.25, (
        f"temporal order {orders[0]:.2f} on the default lane; it was 0.99 "
        f"when measured. If this has become ~2 the lane was made second "
        f"order and the comment in run_ocean_test_matrix.py must be updated")


def test_the_error_is_linear_so_the_lagged_thickness_is_not_the_cause():
    """The measurement that refuted the obvious suspect.

    ``H_u_old``/``H_v_old`` lag the free-surface solve by a step, which IS a
    real first-order term -- but it is NONLINEAR, so its relative
    contribution must shrink with wave amplitude. At eta/H = 1.9e-5 and
    again at 1.9e-7 the relative L2 is identical to 5 digits, so whatever is
    first order here is LINEAR in the wave amplitude.
    """
    M = _matrix()
    v0 = M._IGWC_V_AMP
    try:
        out = []
        for scale in (1.0, 0.01):
            M._IGWC_V_AMP = v0 * scale
            out.append(_run_channel("implicit_cn", "cosine", n_lat=40,
                                    n_lon=80, theta=0.5,
                                    dt_fixed=480.0)["l2"])
    finally:
        M._IGWC_V_AMP = v0
    assert out[0] == pytest.approx(out[1], rel=1e-3), (
        f"relative L2 moved with amplitude ({out[0]:.6f} vs {out[1]:.6f}); a "
        f"nonlinear term such as the lagged H would do that, and the "
        f"attribution in this module would need redoing")


def test_the_second_order_lane_actually_reaches_second_order():
    """The fix, as a measurement: all three chained settings together.

    This is the one that resolves the finding. It also documents the chain --
    if any single setting is dropped the order falls back to ~1.3.
    """
    lane = dict(coriolis="explicit_ab2", outer="ab2", slow_ab2=True)
    c, f = _pair(**lane)
    assert _order(c, f) > 1.7, (
        f"second-order lane gives order {_order(c, f):.2f}, expected ~1.95")
    # ... and it is genuinely more accurate, not just better-sloped.
    c0, f0 = _pair()
    assert f["l2"] < 0.5 * f0["l2"], (
        f"second-order lane L2 {f['l2']:.5f} vs default {f0['l2']:.5f}")


def test_one_setting_alone_does_nothing():
    """Non-vacuity for the CHAIN claim, and the reason this stayed open.

    Each single knob is a null result. Asserting that explicitly stops the
    next person from re-running the same one-knob experiment and concluding
    the setting is irrelevant.
    """
    base_c, base_f = _pair()
    base = _order(base_c, base_f)
    solo = _order(*_pair(coriolis="explicit_ab2", outer="ab2"))
    assert abs(solo - base) < 0.15, (
        f"Coriolis+outer alone moved the order {base:.2f} -> {solo:.2f}; the "
        f"chain claim needs re-deriving")


def test_the_leftover_period_error_converges_at_second_order():
    """What is the ~1% period error that survives the window fix?

    CHALLENGED by GLM-5.2 2026-08-12: "power_law reads 1.0013 on the same
    grid while cosine reads 1.0113, so your leftover cannot be spatial."
    MEASURED period ratios (model/exact), dt scaled with dx:

        grid      box       cosine    power_law   implicit_cn
        12x24     1.01373   1.01132   1.00131     1.01507
        24x48     1.00230   1.00200   0.99311     1.00389
        48x96     1.00055   1.00051   0.99176     1.00105

    RETRACTED, and this test renamed accordingly: I first read that table as
    "the residual is SPATIAL". It does not show that. Because dt scales with
    dx here, an O(dx^2) spatial phase error and an O(dt^2) temporal one
    refine identically and this measurement CANNOT separate them (codex
    2026-08-12). Separating them needs either a fixed outer dt with the grid
    refined and n_barotropic_substeps raised to hold the barotropic CFL, or
    a fixed grid with dt halved; neither is run here.

    What the table DOES establish is the thing the window fix is actually
    on the hook for: box, cosine and implicit_cn converge at ~4x per
    doubling, i.e. SECOND order in the combined refinement, so no
    first-order term survives. A residual mis-centred window would be first
    order and would show up as ~2x per doubling.

    power_law is the outlier and is genuinely different in kind: it does not
    converge to 1 at all, settling near 0.992. Its flattering coarse-grid
    1.00131 was its own bias cancelling the discretisation error, which is
    why it looked best in the comparison that prompted the challenge.
    REPORTED, NOT FIXED HERE: power_law carries a persistent ~0.8% period
    bias, out of scope for the window fix.
    """
    grids = ((12, 24), (24, 48), (48, 96))
    # Cover the two filters the fix touched AND the untouched implicit_cn
    # reference; a single-filter check could pass while a sibling regressed.
    for solver, filt in (("explicit_substep", "cosine"),
                         ("explicit_substep", "box"),
                         ("implicit_cn", "cosine")):
        err = [abs(_run_channel(solver, filt, n_lat=nl, n_lon=no)["ratio"]
                   - 1.0) for nl, no in grids]
        orders = [float(np.log(c / f) / np.log(2.0))
                  for c, f in zip(err, err[1:])]
        # Assert the ORDER, not a ratio bound: "> 1.32" would have been
        # satisfied by a masked first-order term (codex 2026-08-12).
        assert min(orders) > 1.7, (
            f"{solver}/{filt} converges at order {orders}, errors {err}; a "
            f"mis-centred window is FIRST order, so this is the assertion "
            f"that says the fix is complete")


def test_power_law_does_not_converge_to_the_right_period():
    """The outlier above, pinned separately so it is not mistaken for a
    failure of the window fix.

    Non-vacuity for the claim that power_law's coarse-grid win was a
    cancellation: if it were genuinely more accurate it would keep
    converging toward 1 like the others do.
    """
    err = [abs(_run_channel("explicit_substep", "power_law",
                            n_lat=nl, n_lon=no)["ratio"] - 1.0)
           for nl, no in ((24, 48), (48, 96))]
    assert err[1] > 0.5 * err[0], (
        f"power_law now converges ({err[0]:.5f} -> {err[1]:.5f}); its ~0.8% "
        f"persistent bias may have been fixed -- re-derive the note in "
        f"test_the_leftover_period_error_converges_at_second_order")


def test_the_frozen_tide_still_uses_loop_start_sampling(monkeypatch):
    """Pins the tide sample time, because two attempts to change it were wrong.

    Centring the window stretched the substep loop to ~t+2*dt while the
    equilibrium tide stays FROZEN at the loop's start, so a tide-enabled
    box/cosine run now carries about twice the forcing-quadrature error it
    used to (M2 at dt = 1800 s: 14.5 degrees of phase). Both cheap fixes
    were tried and reverted:

      * sampling at ``t + n_substeps*dt_s`` is the window centroid ONLY for
        the forward-frame box/cosine filters -- under MLF dt_s = dt_mom/n
        with dt_mom = 2*dt so it lands a full outer step late, power_law's
        trimmed window has centroid n + 0.0088*n, and nemo_ab3am4 does no
        averaging at all (codex 2026-08-12);
      * refusing the combination breaks the working, tested tide wiring in
        test_tidal_forcing.py::test_wire_*.

    So the sampling is deliberately UNCHANGED and the limitation is
    documented at the call site. This test exists so the next person does
    not re-attempt either fix without reading why. FOLLOW-UP: evaluate the
    tide per substep at t+(i+1)*dt_s.
    """
    import legoesm.ocean.physics.tidal_forcing as tf
    seen, real = [], tf.apply_tidal_forcing

    def _spy(du_dt, dv_dt, grid, t_seconds, config, **kw):
        seen.append(t_seconds)
        return real(du_dt, dv_dt, grid, t_seconds, config, **kw)

    monkeypatch.setattr(tf, "apply_tidal_forcing", _spy)
    t0 = 12345.0
    with jax.disable_jit():
        _run_channel("explicit_substep", "cosine", n_lat=12, n_lon=24,
                     tides=True, t_seconds=t0, n_steps_max=1)
    assert seen and seen[0] is not None, (
        "the tide path did not execute, so this pins nothing")
    assert float(seen[0]) == pytest.approx(t0), (
        f"tide sampled at {float(seen[0])} instead of the loop start {t0}; "
        f"if this was a deliberate change, it must be correct for the MLF "
        f"frame and for power_law/nemo_ab3am4 too -- see the docstring")
