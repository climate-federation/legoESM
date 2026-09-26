"""Zhang-McFarlane's net rain flux takes the surface route on the spectral bridge.

ZM emits ``dq_r_conv_dt`` as CAM's ``ntprprd``: a SIGNED per-layer net
precipitation-flux divergence whose column integral is the surface rain.  The
hydrostatic bridge and the unified pipeline never book it into a tracer; the
spectral bridge used to refuse the scheme outright.  Now it books only dq_v and
dq_c and lets the rain leave the column, as every precipitating species on
this prescribed-surface lane already does.

The first three tests fail with the change reverted (the bridge raises
"signed NET rain-flux divergence" before booking anything); the
Tiedtke test pins the >=0 per-layer rain routes byte-identical; the last test
runs the REAL ZM kernel through the bridge on convecting columns and measures
the column-water closure the surface route relies on.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    isothermal_rest_state_spectral,
)
from legoesm.atmosphere.physics.convection import integration as convint  # noqa: E402
from legoesm.atmosphere.physics.convection.config import ConvectionConfig  # noqa: E402
from legoesm.atmosphere.physics.convection.output import ConvectionOutput  # noqa: E402
from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis_3d  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402

_N_MAX = 8
_N_LEV = 8
_DT = 1800.0
_TRACER_VALUES = {"q_v": 5.0e-3, "q_c": 1.0e-4, "q_i": 2.0e-5, "q_r": 3.0e-6}


def _setup(nlev=_N_LEV, tracers=("q_v", "q_c", "q_i"), convecting=False):
    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(nlev)
    shp = (grid.n_lat, grid.n_lon, nlev)
    tr = {k: jnp.full(shp, _TRACER_VALUES[k]) for k in tracers}
    state = isothermal_rest_state_spectral(
        grid, sigma, T_init=280.0, p_s_init=1.0e5,
        perturbation_amplitude=0.0, tracers=tr)
    if convecting:
        # Warm, humid columns at low latitude index grading to cool, dry
        # ones: deep convection, marginal, and stable columns in one state.
        p_s = jnp.full((grid.n_lat, grid.n_lon), 1.0e5)
        p_full = sigma.pressure_at_full(p_s)
        z = -8000.0 * jnp.log(p_full / p_s[..., None])
        T_sfc = jnp.linspace(303.0, 286.0, grid.n_lat)[:, None, None]
        rh = jnp.linspace(0.85, 0.35, grid.n_lat)[:, None, None]
        T = jnp.maximum(T_sfc - 6.5e-3 * z, 200.0)
        state = state._replace(
            T_hat=state.T_hat.replace(data=sh_analysis_3d(grid, T)),
            tracers={**state.tracers, "q_v": rh * saturation_mixing_ratio(T, p_full)})
    ncol = int(grid.n_lat) * int(grid.n_lon)
    return grid, sigma, state, ncol


def _patterns(ncol, nlev):
    idx = jnp.arange(ncol * nlev, dtype=jnp.float64).reshape(ncol, nlev)
    dq_v = -(idx + 1.0) * 1.0e-8
    dq_c = (idx + 1.0) * 3.0e-9
    # Signed net rain flux: production aloft, evaporation of falling rain in
    # the lower layers (surface is index -1), column sum positive.
    k = jnp.arange(nlev, dtype=jnp.float64)
    dq_r = ((nlev / 2.0 - k) * 1.0e-7)[None, :] * jnp.ones((ncol, 1)) + idx * 1.0e-10
    return dq_v, dq_c, dq_r


def _fake_zm(ncol, nlev, dq_v, dq_c, dq_r):
    z = jnp.zeros((ncol, nlev))

    def fake(*, T, q_v, p_full, p_half, u, v, conv_prog_profile, dt, config,
             land_frac=None, cld_frac=None, pref_edge=None):
        return ConvectionOutput(
            dT_dt=z, dq_v_dt=dq_v, dq_c_conv_dt=dq_c, cape=jnp.zeros((ncol,)),
            convective_mask=jnp.ones((ncol,)), dq_r_conv_dt=dq_r), z
    return fake


def _run(monkeypatch, name, fake, state, grid, sigma):
    monkeypatch.setattr(convint, "_get_convection_fn",
                        lambda cfg: (name, fake, getattr(cfg, name)))
    fn = convint.make_convection_physics(ConvectionConfig(scheme=name),
                                         "spectral_pe", _DT)
    tend, _prog = fn(state, grid, sigma)
    return {k: np.asarray(getattr(v, "data", v)) for k, v in tend.tracers.items()}


def test_zm_rain_takes_the_surface_route_and_only_vapour_and_cloud_are_booked(monkeypatch):
    grid, sigma, state, ncol = _setup()
    dq_v, dq_c, dq_r = _patterns(ncol, _N_LEV)
    assert float(dq_r.min()) < 0.0 < float(dq_r.sum(axis=1).min())
    tt = _run(monkeypatch, "zhang_mcfarlane", _fake_zm(ncol, _N_LEV, dq_v, dq_c, dq_r),
              state, grid, sigma)
    shp = (grid.n_lat, grid.n_lon, _N_LEV)
    np.testing.assert_array_equal(tt["q_v"], np.asarray(dq_v).reshape(shp))
    np.testing.assert_array_equal(tt["q_c"], np.asarray(dq_c).reshape(shp))
    assert "q_r" not in tt
    np.testing.assert_array_equal(tt["q_i"], 0.0)


def test_zm_leaves_an_existing_rain_tracer_untouched(monkeypatch):
    grid, sigma, state, ncol = _setup(tracers=("q_v", "q_c", "q_r"))
    dq_v, dq_c, dq_r = _patterns(ncol, _N_LEV)
    tt = _run(monkeypatch, "zhang_mcfarlane", _fake_zm(ncol, _N_LEV, dq_v, dq_c, dq_r),
              state, grid, sigma)
    shp = (grid.n_lat, grid.n_lon, _N_LEV)
    np.testing.assert_array_equal(tt["q_r"], 0.0)
    np.testing.assert_array_equal(tt["q_c"], np.asarray(dq_c).reshape(shp))


def test_zm_refuses_a_state_without_a_cloud_tracer(monkeypatch):
    grid, sigma, state, ncol = _setup(tracers=("q_v",))
    dq_v, dq_c, dq_r = _patterns(ncol, _N_LEV)
    with pytest.raises(ValueError, match="must carry both tracers"):
        _run(monkeypatch, "zhang_mcfarlane", _fake_zm(ncol, _N_LEV, dq_v, dq_c, dq_r),
             state, grid, sigma)


@pytest.mark.parametrize("with_qr", [False, True])
def test_tiedtke_rain_split_route_is_unchanged(monkeypatch, with_qr):
    tracers = ("q_v", "q_c", "q_r") if with_qr else ("q_v", "q_c")
    grid, sigma, state, ncol = _setup(tracers=tracers)
    dq_v, dq_c, _ = _patterns(ncol, _N_LEV)
    dq_r = jnp.abs(_patterns(ncol, _N_LEV)[2])  # a >=0 per-layer rain SOURCE
    z = jnp.zeros((ncol, _N_LEV))

    def fake(*, T, q_v, p_full, p_half, u, v, conv_prog_profile, dt, config,
             moisture_convergence=None):
        return ConvectionOutput(
            dT_dt=z, dq_v_dt=dq_v, dq_c_conv_dt=dq_c, cape=jnp.zeros((ncol,)),
            convective_mask=jnp.ones((ncol,)), dq_r_conv_dt=dq_r), z
    tt = _run(monkeypatch, "tiedtke", fake, state, grid, sigma)
    shp = (grid.n_lat, grid.n_lon, _N_LEV)
    if with_qr:
        np.testing.assert_array_equal(tt["q_r"], np.asarray(dq_r).reshape(shp))
        np.testing.assert_array_equal(tt["q_c"], np.asarray(dq_c).reshape(shp))
    else:
        np.testing.assert_array_equal(tt["q_c"], np.asarray(dq_c + dq_r).reshape(shp))


def test_real_zm_kernel_closes_column_water_on_the_spectral_bridge(monkeypatch):
    nlev = 20
    grid, sigma, state, ncol = _setup(nlev=nlev, tracers=("q_v", "q_c"), convecting=True)
    real = convint.zhang_mcfarlane_convection
    seen = {}

    def spy(*a, **k):
        out, prog = real(*a, **k)
        seen["out"] = out
        return out, prog
    tt = _run(monkeypatch, "zhang_mcfarlane", spy, state, grid, sigma)
    out = seen["out"]
    n_active = int(jnp.sum(out.convective_mask > 0))
    assert n_active >= 1, "no column convected: the closure test would be vacuous"

    dp = np.asarray(sigma.layer_thickness_dp(jnp.full((ncol,), 1.0e5)))
    dq_v = np.asarray(out.dq_v_dt); dq_c = np.asarray(out.dq_c_conv_dt)
    dq_r = np.asarray(out.dq_r_conv_dt)
    # Contract: sum dp*dq_r = -sum dp*(dq_v + dq_c) per column (pressure-weighted).
    resid = np.sum(dp * (dq_v + dq_c + dq_r), axis=1)
    scale = np.sum(dp * np.abs(dq_v), axis=1)
    assert np.all(np.abs(resid) <= 1.0e-9 * scale.max() + 1.0e-12 * dp.sum(axis=1))
    # The surface rain the route implies is non-negative in every column
    # (the pipeline's clip-at-zero would be inactive) and positive somewhere.
    P = np.sum(dp * dq_r, axis=1) / constants.g
    assert P.min() >= -1.0e-15 * abs(P).max() and P.max() > 0.0
    # And the bridge booked exactly the kernel's vapour and cloud tendencies.
    shp = (grid.n_lat, grid.n_lon, nlev)
    np.testing.assert_array_equal(tt["q_v"], dq_v.reshape(shp))
    np.testing.assert_array_equal(tt["q_c"], dq_c.reshape(shp))
