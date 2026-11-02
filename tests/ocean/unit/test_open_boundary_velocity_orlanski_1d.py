"""MOM6's 1-D Orlanski radiation for the baroclinic velocities at open sides (OpenBoundaryData.velocity_rx_max).

Checked on hand values (inflow, outward phase speed, the cap, a flat interior), on random fields (the boundary value
stays between the model's own boundary value, the first interior value and the data: no new extremes), through the
full baroclinic boundary step (None leaves the CROCO oblique form unchanged; a value changes only the velocity
rows, never T and S), and for a finite gradient.
"""
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.grids.factory import create_regional_grid  # noqa: E402
from legoesm.ocean.dynamics.open_boundary_latlon import (  # noqa: E402
    OpenBoundaryData, _orlanski_1d, apply_baroclinic_open_boundary)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean  # noqa: E402
from legoesm.ocean.vertical import compute_layer_thickness, create_z_star_from_thicknesses  # noqa: E402

set_policy(PrecisionPolicy.fp64())
RI, RO = 1800.0 / 86400.0, 1800.0 / (360 * 86400.0)


def test_hand_values():
    a = lambda x: jnp.asarray([x], dtype=float)  # noqa: E731
    # inflow (f1 rising toward f2, phase speed inward): rx = 0, the model's own boundary value, nudged with tau_in
    g = RI / (1 + RI)
    out = _orlanski_1d(a(0.3), a(1.0), a(1.2), a(0.5), a(0.0), RI, RO, 1.0)
    assert np.isclose(float(out[0]), (1 - g) * 0.3)
    # outward, rx = (1.0 - 0.8) / (0.8 - 0.4) = 0.5
    g = RO / (1 + RO)
    out = _orlanski_1d(a(0.3), a(1.0), a(0.8), a(0.4), a(0.0), RI, RO, 1.0)
    assert np.isclose(float(out[0]), (1 - g) * (0.3 + 0.5 * 0.8) / 1.5)
    # outward, rx = 0.6 / 0.1 = 6 capped at 1
    out = _orlanski_1d(a(0.3), a(1.0), a(0.4), a(0.3), a(0.0), RI, RO, 1.0)
    assert np.isclose(float(out[0]), (1 - g) * (0.3 + 0.4) / 2.0)
    # flat interior (f1 = f2): no division by zero, treated as inflow
    out = _orlanski_1d(a(0.3), a(1.0), a(0.4), a(0.4), a(0.2), RI, RO, 1.0)
    assert np.isfinite(float(out[0]))


def test_no_new_extremes():
    rng = np.random.default_rng(0)
    fb, f1o, f1n, f2n, d = (jnp.asarray(rng.standard_normal((6, 40))) for _ in range(5))
    out = np.asarray(_orlanski_1d(fb, f1o, f1n, f2n, d, RI, RO, 1.0))
    lo = np.minimum.reduce([np.asarray(fb), np.asarray(f1n), np.asarray(d)])
    hi = np.maximum.reduce([np.asarray(fb), np.asarray(f1n), np.asarray(d)])
    assert np.all(out >= lo - 1e-14) and np.all(out <= hi + 1e-14)


def test_gradient_is_finite():
    rng = np.random.default_rng(1)
    args = [jnp.asarray(rng.standard_normal((3, 8))) for _ in range(5)]
    g = jax.grad(lambda x: jnp.sum(_orlanski_1d(x, *args[1:], RI, RO, 1.0) ** 2))(args[0])
    assert np.all(np.isfinite(np.asarray(g))) and np.abs(np.asarray(g)).max() > 0


def _states():
    grid, wall = create_regional_grid("latlon", n_lat=8, n_lon=10, lat_south=-30.0, lat_north=-26.0,
                                      lon_west=200.0, lon_east=205.0)
    dz = np.geomspace(10.0, 400.0, 6)
    zc = create_z_star_from_thicknesses(dz)
    H = np.full(np.asarray(wall).shape, float(dz.sum()))
    lm = np.ones_like(H)
    st = rest_state_latlon_cgrid_ocean(grid, zc, land_mask_override=jnp.asarray(lm),
                                       H_bathy_override=jnp.asarray(H), H_max=float(zc.H_max))
    rng = np.random.default_rng(2)

    def perturb(s, seed):
        r = np.random.default_rng(seed)
        return s._replace(u=s.u.replace(data=jnp.asarray(0.5 * r.standard_normal(s.u.data.shape))),
                          v=s.v.replace(data=jnp.asarray(0.5 * r.standard_normal(s.v.data.shape))),
                          T=s.T.replace(data=s.T.data + jnp.asarray(r.standard_normal(s.T.data.shape))))
    old, new = perturb(st, 3), perturb(st, 4)
    nl, nx = len(dz), st.T.data.shape[1]
    ob = OpenBoundaryData(south_T=jnp.asarray(rng.standard_normal((nl, nx))), south_S=jnp.full((nl, nx), 35.0),
                          south_u3=jnp.asarray(0.1 * rng.standard_normal((nl, nx))),
                          south_t3=jnp.asarray(0.1 * rng.standard_normal((nl, nx + 1))),
                          tau_in=jnp.asarray(86400.0), tau_out=jnp.asarray(360 * 86400.0))
    h = compute_layer_thickness(new.eta.data, new.H_bathy.data, zc)
    return old, new, ob, h


def test_full_step_changes_only_the_velocity_rows():
    old, new, ob, h = _states()
    a = apply_baroclinic_open_boundary(old, new, ob, 1800.0, h)
    b = apply_baroclinic_open_boundary(old, new, ob._replace(velocity_rx_max=1.0), 1800.0, h)
    assert np.array_equal(np.asarray(a.T.data), np.asarray(b.T.data))
    assert np.array_equal(np.asarray(a.S.data), np.asarray(b.S.data))
    du = np.abs(np.asarray(a.u.data) - np.asarray(b.u.data))
    dv = np.abs(np.asarray(a.v.data) - np.asarray(b.v.data))
    assert du[0].max() > 0 and dv[1].max() > 0          # south ring tangential u, open face normal v
    assert du[1:].max() == 0 and dv[np.r_[0, 2:dv.shape[0]]].max() == 0
    # the depth mean of the open face normal velocity is kept from the barotropic condition in both forms
    w = np.asarray(jnp.minimum(h[0], h[1]).T)
    w = w / w.sum(0, keepdims=True)
    assert np.allclose((w * np.asarray(a.v.data[1]).T).sum(0), (w * np.asarray(new.v.data[1]).T).sum(0))
    assert np.allclose((w * np.asarray(b.v.data[1]).T).sum(0), (w * np.asarray(new.v.data[1]).T).sum(0))


def test_tangential_nudge_only():
    """The ring-row tangential velocity is the previous value nudged toward the data with tau_in (before the
    depth mean correction, which keeps the model's depth mean); the normal velocity and T, S are unchanged."""
    old, new, ob, h = _states()
    b = apply_baroclinic_open_boundary(old, new, ob._replace(velocity_rx_max=1.0), 1800.0, h)
    c = apply_baroclinic_open_boundary(old, new, ob._replace(velocity_rx_max=1.0, tangential_nudge_only=True), 1800.0, h)
    assert np.array_equal(np.asarray(b.v.data), np.asarray(c.v.data))
    assert np.array_equal(np.asarray(b.T.data), np.asarray(c.T.data))
    g = RI / (1 + RI)
    rad = (1 - g) * np.asarray(old.u.data[0]).T + g * np.asarray(ob.south_t3)          # (nlev, along)
    ring = np.asarray(h[0]).T
    inner = np.minimum(ring[:, :-1], ring[:, 1:])
    hf = np.concatenate([np.zeros((ring.shape[0], 1)), inner, np.zeros((ring.shape[0], 1))], axis=1)
    w = hf / np.maximum(hf.sum(0, keepdims=True), 1e-20)
    expect = (rad - (w * rad).sum(0, keepdims=True) + (w * np.asarray(new.u.data[0]).T).sum(0, keepdims=True)) * (hf > 0)
    assert np.allclose(np.asarray(c.u.data[0]).T, expect, rtol=0, atol=1e-13)
    assert np.max(np.abs(np.asarray(c.u.data[0]) - np.asarray(b.u.data[0]))) > 0
