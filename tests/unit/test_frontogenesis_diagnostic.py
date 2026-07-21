"""Tests for the E3SM-faithful frontogenesis producer (FRONTGF/FRONTGA).

Oracle: E3SM gravity_waves_sources.F90::compute_frontogenesis (Taylor 2011):
F = -gradth . [(gradth . grad) U] with HOMME's ugradv_sphere = the COVARIANT
directional derivative (Cartesian embedding -> scalar gradients -> tangential
projection).  Truth-tier pins:

1. Gradient operator pins (spectral, exact): grad(sin lat) = (0, cos/a);
   grad(cos lat cos lon) = (-sin lon / a, -sin lat cos lon / a).
2. COVARIANT canary: for solid-body rotation (a rigid motion),
   (gradth . grad)U = Omega x gradth, so F = -gradth.(Omega x gradth) == 0
   for ANY theta.  A naive component-wise (non-covariant) formula fails this
   at high latitude; the pin demands |F| <= 1e-10 x the |gradth|^2*Omega
   scale (observed ~4e-14).
3. Local-plane deformation closed form: near the equator with confluence
   u = -alpha x and theta = beta x, F -> +alpha beta^2 (frontogenesis);
   u = +alpha x gives -alpha beta^2 (frontolysis).  Signs + magnitude.
4. Cross-implementation consistency: spectral (exact SH) vs lat-lon
   (centered FD) producers agree on a smooth global field to FD accuracy.
5. Angle: atan2(gy, gx + 1e-10) with the oracle regulariser.
6. Dispatch: supported-predicate truth table; unsupported grid raises;
   single-column returns zeros; differentiability.

Run with JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics._shared import exner_function  # noqa: E402
from legoesm.atmosphere.physics.gravity_wave_drag.frontogenesis import (  # noqa: E402
    _frontogenesis_core,
    _latlon_gradient,
    _spectral_gradient,
    compute_frontogenesis,
    frontogenesis_supported,
)
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402

from legoesm import constants  # noqa: E402


def _ggrid():
    return create_gaussian_grid(n_max=21, dealiasing="linear")


# ---------------------------------------------------------------------------
# 1. Spectral gradient operator pins (exact analytics).
# ---------------------------------------------------------------------------
def test_spectral_gradient_analytic_pins():
    g = _ggrid()
    a = float(g.radius)
    lat = g.lat2d[:, :, None]
    lon = g.lon2d[:, :, None]

    gx, gy = _spectral_gradient(g, jnp.sin(lat))
    assert float(jnp.max(jnp.abs(gx))) < 1e-18            # zonally symmetric
    np.testing.assert_allclose(
        np.asarray(gy), np.asarray(jnp.cos(lat)) / a, rtol=0, atol=1e-11 / a)

    gx2, gy2 = _spectral_gradient(g, jnp.cos(lat) * jnp.cos(lon))
    np.testing.assert_allclose(
        np.asarray(gx2), np.asarray(-jnp.sin(lon)) / a, rtol=0, atol=1e-11 / a)
    np.testing.assert_allclose(
        np.asarray(gy2), np.asarray(-jnp.sin(lat) * jnp.cos(lon)) / a,
        rtol=0, atol=1e-11 / a)


# ---------------------------------------------------------------------------
# 2. Covariant canary: solid-body rotation => F == 0 for ANY theta.
# ---------------------------------------------------------------------------
def test_solid_body_frontogenesis_is_zero_covariant_canary():
    g = _ggrid()
    a = float(g.radius)
    lat = g.lat2d[:, :, None]
    lon = g.lon2d[:, :, None]
    omega = 7.0e-6
    u = omega * a * jnp.cos(lat) + 0.0 * lon
    v = jnp.zeros_like(u)
    theta = (300.0 + 10.0 * jnp.sin(lat)
             + 5.0 * jnp.cos(lat) * jnp.cos(2.0 * lon)
             + 3.0 * jnp.sin(lat) ** 2)

    grad_fn = lambda f: _spectral_gradient(g, f)  # noqa: E731
    fgf, _ = _frontogenesis_core(u, v, theta, g.lat2d, g.lon2d, grad_fn)
    gx, gy = grad_fn(theta)
    scale = float(jnp.max(gx * gx + gy * gy)) * omega     # |gradth|^2 * Omega
    rel = float(jnp.max(jnp.abs(fgf))) / scale
    assert rel < 1e-10, (
        f"solid-body F/|gradth|^2*Omega = {rel:.2e}: the covariant "
        "(Cartesian-embedding) derivative must annihilate rigid rotation "
        "(a component-wise/non-covariant operator does not)"
    )


# ---------------------------------------------------------------------------
# 3. Local-plane deformation closed form (sign + magnitude).
# ---------------------------------------------------------------------------
def _deformation_case(sign):
    """Fine lat-lon grid; confluence u = sign*alpha*x near the equator."""
    g = create_latlon_grid(n_lat=96, n_lon=192)
    a_r = float(getattr(g, "radius", constants.R_earth))
    x = a_r * jnp.cos(g.lat2d) * g.lon2d                    # local plane coords
    alpha, beta = 3.0e-6, 2.0e-5                          # [1/s], [K/m]
    u = (sign * alpha * (x - float(jnp.mean(x))))[:, :, None]
    v = jnp.zeros_like(u)
    theta = (300.0 + beta * (x - float(jnp.mean(x))))[:, :, None]
    grad_fn = lambda f: _latlon_gradient(g, f)  # noqa: E731
    fgf, _ = _frontogenesis_core(u, v, theta, g.lat2d, g.lon2d, grad_fn)
    # Sample near the equator + domain center where the plane limit is clean.
    i0 = g.lat2d.shape[0] // 2
    j0 = g.lat2d.shape[1] // 2
    return float(fgf[i0, j0, 0]), alpha, beta


def test_deformation_flow_closed_form_signs():
    # u = -alpha x (confluence) sharpens a theta = beta x gradient:
    # F = -gradth.[(gradth.grad)u] = -beta*(beta*(-alpha)) = +alpha*beta^2.
    f_conf, alpha, beta = _deformation_case(-1.0)
    expected = alpha * beta * beta
    np.testing.assert_allclose(f_conf, expected, rtol=2e-2)
    # u = +alpha x (diffluence) weakens it: F = -alpha*beta^2.
    f_diff, _, _ = _deformation_case(+1.0)
    np.testing.assert_allclose(f_diff, -expected, rtol=2e-2)


# ---------------------------------------------------------------------------
# 4. Cross-implementation consistency (spectral exact vs lat-lon FD).
# ---------------------------------------------------------------------------
def test_spectral_vs_latlon_consistency_smooth_field():
    gg = _ggrid()

    def fields(lat2d, lon2d):
        lat = lat2d[:, :, None]
        lon = lon2d[:, :, None]
        u = 12.0 * jnp.cos(lat) + 3.0 * jnp.sin(lat) * jnp.cos(lon)
        v = 2.0 * jnp.sin(lon) * jnp.cos(lat)
        theta = 300.0 + 8.0 * jnp.sin(lat) + 4.0 * jnp.cos(lat) * jnp.cos(lon)
        return u, v, theta

    u, v, th = fields(gg.lat2d, gg.lon2d)
    f_spec, _ = _frontogenesis_core(
        u, v, th, gg.lat2d, gg.lon2d, lambda f: _spectral_gradient(gg, f))

    gl = create_latlon_grid(n_lat=192, n_lon=384)
    ul, vl, thl = fields(gl.lat2d, gl.lon2d)
    f_ll, _ = _frontogenesis_core(
        ul, vl, thl, gl.lat2d, gl.lon2d, lambda f: _latlon_gradient(gl, f))

    # Compare at nearest lat-lon points to a subset of Gaussian nodes away
    # from the poles (FD one-sided edges are less accurate there).
    lats = np.asarray(gg.lat)
    sel = np.where(np.abs(lats) < 1.1)[0]                 # |lat| < ~63 deg
    scale = float(jnp.max(jnp.abs(f_spec)))
    for i in sel[:: max(1, len(sel) // 8)]:
        il = int(np.argmin(np.abs(np.asarray(gl.lat) - lats[i])))
        for j in range(0, gg.n_lon, gg.n_lon // 6):
            jl = int(np.argmin(np.abs(
                np.asarray(gl.lon) - float(gg.lon[j]))))
            assert abs(float(f_spec[i, j, 0]) - float(f_ll[il, jl, 0])) < 0.06 * scale, (
                f"spectral vs lat-lon FD frontogenesis disagree at "
                f"lat={lats[i]:.2f}, lon={float(gg.lon[j]):.2f}"
            )


# ---------------------------------------------------------------------------
# 5. Angle (oracle atan2 + 1e-10 regulariser).
# ---------------------------------------------------------------------------
def test_frontga_angle_convention():
    g = _ggrid()
    lat = g.lat2d[:, :, None]
    grad_fn = lambda f: _spectral_gradient(g, f)  # noqa: E731
    u = jnp.zeros_like(lat) + 1.0
    v = jnp.zeros_like(u)
    # theta increasing northward -> gradth = (0, +): angle = atan2(+, 1e-10)
    _, ga = _frontogenesis_core(u, v, 300.0 + 10.0 * jnp.sin(lat),
                                g.lat2d, g.lon2d, grad_fn)
    mid = ga[g.n_lat // 2, 0, 0]
    np.testing.assert_allclose(float(mid), np.pi / 2.0, atol=1e-4)


# ---------------------------------------------------------------------------
# 6. Public dispatch, zeros, units-shape, differentiability.
# ---------------------------------------------------------------------------
def test_supported_predicate_truth_table():
    assert frontogenesis_supported(_ggrid())
    assert frontogenesis_supported(create_latlon_grid(n_lat=8, n_lon=16))

    class _SCM:
        grid_n_columns = 1

    assert frontogenesis_supported(_SCM())

    class _Unknown:
        pass

    assert not frontogenesis_supported(_Unknown())
    with pytest.raises(TypeError, match="unsupported grid"):
        compute_frontogenesis(
            jnp.zeros((2, 2, 1)), jnp.zeros((2, 2, 1)),
            jnp.full((2, 2, 1), 280.0), jnp.full((2, 2, 1), 5e4), _Unknown())


def test_single_column_returns_zeros():
    class _SCM:
        grid_n_columns = 1

    t_col = jnp.full((1, 4), 280.0)
    fgf, fga = compute_frontogenesis(
        jnp.zeros((1, 4)), jnp.zeros((1, 4)), t_col, jnp.full((1, 4), 5e4), _SCM())
    assert fgf.shape == (1, 4) and float(jnp.max(jnp.abs(fgf))) == 0.0
    assert fga.shape == (1, 4)


def test_public_producer_uses_theta_not_raw_temperature():
    """The public producer must feed theta = T / exner(p) (E3SM
    T*(p0/p)^kappa) into the core — pinned by EQUALITY against the core
    called with theta, and by a MATERIAL difference from the core called
    with raw T (codex R1 finding 2: the previous check never touched the
    producer).  A vertically-VARYING pressure makes the two genuinely
    different (at uniform p the exner factor is a constant scale)."""
    g = _ggrid()
    lat = g.lat2d[:, :, None]
    lon = g.lon2d[:, :, None]
    nlev = 3
    ones = jnp.ones((g.n_lat, g.n_lon, nlev))
    p = jnp.stack([3.0e4 * ones[..., 0], 5.0e4 * ones[..., 0],
                   9.0e4 * ones[..., 0]], axis=-1)
    u = (-10.0 * jnp.sin(lon - jnp.pi) * jnp.cos(lat)) * ones
    v = jnp.zeros_like(u)
    T = (280.0 + 5.0 * jnp.sin(lon) * jnp.cos(lat)) * ones  # noqa: N806

    fgf, fga = compute_frontogenesis(u, v, T, p, g)
    assert fgf.shape == (g.n_lat * g.n_lon, nlev)
    assert fga.shape == fgf.shape
    assert bool(jnp.all(jnp.isfinite(fgf)))

    grad_fn = lambda f: _spectral_gradient(g, f)  # noqa: E731
    f_theta, _ = _frontogenesis_core(
        u, v, T / exner_function(p), g.lat2d, g.lon2d, grad_fn)
    np.testing.assert_allclose(
        np.asarray(fgf),
        np.asarray(f_theta.reshape(g.n_lat * g.n_lon, nlev)),
        rtol=1e-12)
    f_rawT, _ = _frontogenesis_core(  # noqa: N806
        u, v, T, g.lat2d, g.lon2d, grad_fn)
    scale = float(jnp.max(jnp.abs(f_theta)))
    assert float(jnp.max(jnp.abs(
        f_theta - f_rawT))) > 0.1 * scale, (
        "raw-T frontogenesis is indistinguishable from theta-based -> the "
        "theta pin is vacuous (make p vary in the vertical)")


def test_regional_and_stretched_latlon_rejected():
    """The centered stencil assumes UNIFORM spacing + PERIODIC longitude:
    regional (walled) and Mercator-stretched lat-lon grids must be
    unsupported (codex R1 finding 1 — grid.dlat is only representative on
    the stretched builder), so the coupled pipeline keeps rejecting the
    frontal source there rather than launching from a wrong stencil."""
    from legoesm.grids.latlon import (
        create_regional_latlon_grid,
        create_stretched_latlon_grid,
    )

    reg, _wall_mask = create_regional_latlon_grid(
        n_lat=12, n_lon=24, lat_south=10.0, lat_north=50.0,
        lon_west=0.0, lon_east=60.0)
    assert not frontogenesis_supported(reg)
    stretched = create_stretched_latlon_grid(
        dy_deg=[1.0, 2.0, 4.0], n_lon=24, lat_south=-30.0)
    assert not frontogenesis_supported(stretched)
    with pytest.raises(TypeError, match="unsupported grid"):
        compute_frontogenesis(
            jnp.zeros((reg.lat2d.shape[0], reg.lat2d.shape[1], 1)),
            jnp.zeros((reg.lat2d.shape[0], reg.lat2d.shape[1], 1)),
            jnp.full((reg.lat2d.shape[0], reg.lat2d.shape[1], 1), 280.0),
            jnp.full((reg.lat2d.shape[0], reg.lat2d.shape[1], 1), 5e4), reg)


def test_channel_band_and_mild_stretch_rejected():
    """codex R2 canaries: (a) a PERIODIC CHANNEL / SPMD latitude-band slice
    (full longitude, walled latitude band) must be rejected — its one-sided
    meridional edges sit at physical walls / internal partition edges; (b)
    even a MILD stretch (0.01 deg step change, percent-scale metric error)
    must not slip under the uniformity tolerance."""
    from types import SimpleNamespace

    g = create_latlon_grid(n_lat=32, n_lon=64, dtype=jnp.float64)
    # (a) latitude-band slice of a global grid: uniform + global-lon but the
    # lat span is a band -> rejected by the lat-span check.
    band = SimpleNamespace(
        lat=g.lat[8:24], lon=g.lon, dlat=g.dlat, dlon=g.dlon)
    assert not frontogenesis_supported(band)
    # (b) mild stretch that ISOLATES the uniformity tolerance (codex R3):
    # move exactly ONE INTERIOR point by 0.01 deg (~1.7e-4 rad) so both
    # endpoints stay global (the lat-span check passes) and cast to fp32 —
    # the perturbation must be caught by the tightened 1e2*eps tolerance
    # (it slipped under the earlier 1e4*eps32 ~ 1.2e-3 rad band).
    lat = np.asarray(g.lat).copy()
    lat[16] = lat[16] + np.deg2rad(0.01)
    mild = SimpleNamespace(lat=jnp.asarray(lat, dtype=jnp.float32),
                           lon=jnp.asarray(g.lon, dtype=jnp.float32),
                           dlat=g.dlat, dlon=g.dlon)
    assert not frontogenesis_supported(mild)
    # Control: the untouched global grid stays accepted (fp64 and fp32).
    assert frontogenesis_supported(g)
    assert frontogenesis_supported(create_latlon_grid(n_lat=32, n_lon=64))


def test_differentiable_wrt_temperature():
    g = _ggrid()
    lat = g.lat2d[:, :, None]
    lon = g.lon2d[:, :, None]
    u = 10.0 * jnp.cos(lat) + 0.0 * lon
    v = 1.0 * jnp.sin(lon) * jnp.cos(lat)
    p = jnp.full((g.n_lat, g.n_lon, 1), 5.0e4)

    def loss(amp):
        t_fld = 280.0 + amp * jnp.sin(lat) + 3.0 * jnp.cos(lat) * jnp.cos(lon)
        fgf, _ = compute_frontogenesis(u, v, t_fld, p, g)
        return jnp.sum(fgf * fgf)

    grad = float(jax.grad(loss)(8.0))
    assert np.isfinite(grad) and abs(grad) > 0.0
