"""Unit test for ``scripts/validate/ocean_fidelity/omip_eq_wind_response.py``.

``transports`` decides every number the probe prints.  Synthetic fields on the
common grid with known answers: v = +0.1 m/s north of the equator and -0.1 m/s
south over the top 50 m gives V_N = +5, V_S = -5, div_ek = 10 m^2/s; a jet of
1.2 m/s at one level/longitude is found as the undercurrent core; a uniform
-0.4 m/s surface current is the SEC.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_DIR = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "ocean_fidelity"


@pytest.fixture(scope="module")
def mod():
    import sys
    sys.path.insert(0, str(_DIR)); sys.path.insert(0, str(_DIR.parent))
    spec = importlib.util.spec_from_file_location("eq_wind", _DIR / "omip_eq_wind_response.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_known_transports_and_jets(mod):
    zi = np.array([0.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    zN = 0.5 * (zi[1:] + zi[:-1])
    shp = (zN.size, mod.GLAT.size, mod.GLON.size)
    v = np.zeros(shp); u = np.zeros(shp)
    v[:3, mod.GLAT > 0, :] = 0.1
    v[:3, mod.GLAT < 0, :] = -0.1
    v[3:, :, :] = 7.0                                    # below 50 m: must not count
    u[:2] = -0.4
    i = int(np.argmin(np.abs(mod.GLON - 230.5)))
    u[4, np.abs(mod.GLAT) <= 1.0, i] = 3.6               # core at k=4, one longitude
    r = mod.transports(u, v, zN, zi)
    assert r["V_N"] == pytest.approx(5.0) and r["V_S"] == pytest.approx(-5.0)
    assert r["div_ek"] == pytest.approx(10.0)
    assert r["u_sec"] == pytest.approx(-0.4)
    assert r["z_euc"] == pytest.approx(zN[4]) and r["lon_euc"] == pytest.approx(230.5)
    assert r["u_euc"] == pytest.approx(3.6)


def test_column_transport_and_ssh_slope(mod):
    """u = 0.5 m/s to 400 m and 9 m/s below (must not count) -> U_col = 200 in both
    boxes; eta falling 1 cm per degree eastward -> west minus east = +100 cm over
    160-260E; a missing layer in the west box makes that box's mean ignore it."""
    zi = np.array([0.0, 10.0, 50.0, 150.0, 400.0, 600.0])
    zN = 0.5 * (zi[1:] + zi[:-1])
    shp = (zN.size, mod.GLAT.size, mod.GLON.size)
    u = np.full(shp, 0.5); u[4:] = 9.0
    v = np.zeros(shp)
    eta = np.broadcast_to(-0.01 * (mod.GLON - 160.0), (mod.GLAT.size, mod.GLON.size)).copy()
    r = mod.transports(u, v, zN, zi, eta=eta)
    assert r["U_col_W"] == pytest.approx(200.0) and r["U_col_E"] == pytest.approx(200.0)
    assert r["deta_WE"] == pytest.approx(100.0)
    u[3, :, mod.GLON < 180.0] = np.nan; u[:, :, (mod.GLON >= 180.0) & (mod.GLON <= 210.0)] = 1.0
    r = mod.transports(u, v, zN, zi)
    assert r["U_col_W"] == pytest.approx(400.0) and r["U_col_E"] == pytest.approx(200.0)
    u2 = np.full(shp, 0.5); u2[3, mod.GLAT > 0] = np.nan   # one row missing a layer: drop that row only
    u2[:, mod.GLAT < 0] = 1.0
    assert mod.transports(u2, v, zN, zi)["U_col_E"] == pytest.approx(400.0)
    zi3 = np.array([0.0, 100.0, 300.0, 500.0]); zN3 = 0.5 * (zi3[1:] + zi3[:-1])
    u3 = np.ones((3, mod.GLAT.size, mod.GLON.size)); u3[2] = 2.0   # 300-500 m layer: only 100 m counts
    assert mod.transports(u3, np.zeros_like(u3), zN3, zi3)["U_col_W"] == pytest.approx(500.0)
    with pytest.raises(ValueError, match="levels reach only"):
        mod.transports(u[:3], v[:3], zN[:3], zi[:4])


def test_pressure_force_and_geostrophy(mod):
    """Uniform water, SSH falling 1 cm/deg eastward -> eastward force
    g*0.01/(1 deg in m)*400 m, all of it SSH; SSH rising 2 cm/deg northward ->
    ug_N = -(g/f)*0.02/(1 deg in m); warm-west water, flat SSH -> westward
    density force and zero SSH force."""
    from legoesm import constants
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    g, R = constants.g, constants.R_earth
    dm = np.deg2rad(1.0) * R
    zi = np.array([0.0, 10.0, 50.0, 150.0, 300.0, 400.0, 600.0]); zN = 0.5 * (zi[1:] + zi[:-1])
    shp = (zN.size, mod.GLAT.size, mod.GLON.size)
    T = np.full(shp, 20.0); S = np.full(shp, 35.0); u = np.zeros(shp)
    LA, LO = np.meshgrid(mod.GLAT, mod.GLON, indexing="ij")
    r = mod.pressure_terms(T, S, -0.01 * (LO - 160.0), u, zN, zi)
    assert r["F_ssh"] == pytest.approx(g * 0.01 / dm * 400.0 * 1e5, rel=1e-9)
    assert r["F_tot"] == pytest.approx(r["F_ssh"], rel=1e-9)
    r = mod.pressure_terms(T, S, 0.02 * LA, u, zN, zi)
    f = 2.0 * constants.Omega * np.sin(np.deg2rad(2.0))
    assert r["ug_N"] == pytest.approx(-g / f * 0.02 / dm, rel=1e-9)
    assert r["ug_S"] == pytest.approx(-g / -f * 0.02 / dm, rel=1e-9)
    assert r["u_N"] == 0.0
    Tw = 25.0 - 0.05 * (LO - 160.0)[None] + 0.0 * T
    r = mod.pressure_terms(Tw, S, 0.0 * LA, u, zN, zi)
    assert r["F_ssh"] == pytest.approx(0.0, abs=1e-12) and r["F_tot"] < -1.0


def test_density_force_magnitude_with_clipped_layer(mod):
    """Warm-west water on levels whose last layer (300-500 m) straddles 400 m:
    the force equals an explicit loop over layers, pressure taken at each
    layer's centre and at 350 m (the overlap midpoint) for the clipped one."""
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.eos import wright_eos
    set_policy(PrecisionPolicy.fp64())
    g, r0, R = constants.g, constants.rho_ocean, constants.R_earth
    zi = np.array([0.0, 20.0, 100.0, 300.0, 500.0]); zN = np.array([8.0, 55.0, 190.0, 420.0])
    shp = (zN.size, mod.GLAT.size, mod.GLON.size)
    LA, LO = np.meshgrid(mod.GLAT, mod.GLON, indexing="ij")
    T = 25.0 - 0.05 * (LO - 160.0)[None] - 0.02 * zN[:, None, None]
    S = np.full(shp, 35.0)
    r = mod.pressure_terms(T, S, 0.0 * LA, np.zeros(shp), zN, zi)
    ls = (mod.GLON >= 160.0) & (mod.GLON <= 260.0)
    x = np.deg2rad(mod.GLON[ls]) * R
    rho = np.asarray(wright_eos(jnp.asarray(T), jnp.asarray(S),
                                jnp.asarray(r0 * g * zN[:, None, None] + 0.0 * T)))[:, 10, ls]
    zc = [8.0, 55.0, 190.0, 350.0]; thick = [20.0, 80.0, 200.0, 100.0]
    F = 0.0
    for k in range(4):
        Pk = sum(g * (rho[j] - r0) * (zi[j + 1] - zi[j]) for j in range(k)) + g * (rho[k] - r0) * (zc[k] - zi[k])
        F += -np.polyfit(x, Pk, 1)[0] / r0 * thick[k]
    assert r["F_tot"] == pytest.approx(F * 1e5, rel=1e-9)


def test_momentum_terms_known_answers(mod):
    """Depth-uniform fields to 400 m (9 m/s below must not count).
    (1) u = 0.3 + 0.004 (lon - 210), v = 0: A_x = H (u160^2 - u260^2) / Lx,
    upwelling w = H a / dx lifts the box-mean u = 0.3 -> A_z = 0.3 H a / dx;
    curvature, Coriolis and A_y are zero.
    (2) u = 0.5, v = 0.02 * lat: A_y = -2 c u H / Ly, Cor = H c 0.5 f(0.5)."""
    from legoesm import constants
    R, Om = float(constants.R_earth), float(constants.Omega)
    zi = np.array([0.0, 10.0, 50.0, 150.0, 400.0, 600.0])
    zN = 0.5 * (zi[1:] + zi[:-1]); H = 400.0
    shp = (zN.size, mod.GLAT.size, mod.GLON.size)
    a = 0.004
    u = np.broadcast_to(0.3 + a * (mod.GLON - 210.0), shp).copy(); u[4:] = 9.0
    v = np.zeros(shp)
    r = mod.momentum_terms(u, v, zN, zi)
    uW, uE = 0.3 + a * (160.0 - 210.0), 0.3 + a * (260.0 - 210.0)
    assert r["A_x"] == pytest.approx(1e5 * H * (uW ** 2 - uE ** 2) / (np.deg2rad(100.0) * R))
    dx = np.deg2rad(1.0) * R * np.cos(np.deg2rad(0.5))
    assert r["A_z"] == pytest.approx(1e5 * 0.3 * H * a / dx)
    for t in ("A_y", "Cor", "lapU"):
        assert abs(r[t]) < 1e-9, t
    assert r["U_box"] == pytest.approx(0.3 * H)
    c = 0.02
    u = np.full(shp, 0.5); u[4:] = 9.0
    v = np.broadcast_to((c * mod.GLAT)[:, None], shp[1:]).copy() * np.ones((zN.size, 1, 1)); v[4:] = 9.0
    r = mod.momentum_terms(u, v, zN, zi)
    assert r["A_y"] == pytest.approx(1e5 * -2.0 * c * 0.5 * H / (np.deg2rad(2.0) * R))
    assert r["Cor"] == pytest.approx(1e5 * H * c * 0.5 * 2.0 * Om * np.sin(np.deg2rad(0.5)))
    assert abs(r["A_x"]) < 1e-9 and abs(r["lapU"]) < 1e-9
    u[2, :, 30] = np.nan                                 # 160.5E: a missing west-face column must raise
    with pytest.raises(ValueError, match="box face"):
        mod.momentum_terms(u, v, zN, zi)
