"""Differentiable land-parameter calibration core (synthetic data, no network)."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from scripts.run.train_land_params_era5 import (
    constrain, init_raw_params, forward, loss_fn, train, BOUNDS, _N_PFT)


def _synthetic_data(ncol=16):
    rng = np.random.default_rng(0)
    lat = np.deg2rad(rng.uniform(-80, 80, ncol))
    pft = rng.random((ncol, _N_PFT)); pft /= pft.sum(1, keepdims=True)
    z = lambda v: jnp.full((ncol,), v)
    f = [AtmToSurface(
        sw_down=z(200.0), lw_down=z(320.0), precip_total=z(2e-5), precip_snow=z(0.0),
        T_lowest=jnp.asarray(285 - 0.3 * np.abs(np.rad2deg(lat))),
        q_lowest=z(5e-3), u_lowest=z(3.0), v_lowest=z(2.0), p_lowest=z(9.9e4),
        p_surface=z(1.0e5), rho_lowest=z(1.2), cos_zenith=z(0.5), co2_ppmv=z(412.0),
        has_radiation=z(1.0), has_precipitation=z(1.0)) for _ in range(12)]
    dom = pft.argmax(1); oh = np.zeros((ncol, _N_PFT)); oh[np.arange(ncol), dom] = 1.0
    return dict(forc=f, lat=jnp.asarray(lat), pft=jnp.asarray(pft),
                fg=jnp.asarray(rng.random(ncol) * 0.3), wp=z(0.12), fc=z(0.30),
                skt=jnp.asarray(285 - 0.3 * np.abs(np.rad2deg(lat)))[None].repeat(12, 0),
                alb=jnp.full((12, ncol), 0.2), t0=jnp.asarray(285.0 * np.ones(ncol)),
                dom_onehot=jnp.asarray(oh), w=jnp.cos(jnp.asarray(lat)))


def test_constrain_bounds():
    cp = constrain(init_raw_params())
    for k, (lo, hi) in BOUNDS.items():
        v = np.asarray(cp[k])
        assert np.all(v >= lo - 1e-9) and np.all(v <= hi + 1e-9)


def test_forward_finite():
    data = _synthetic_data()
    T, A = forward(constrain(init_raw_params()), data, n_spin_years=1)
    assert T.shape == (12, 16) and A.shape == (12, 16)
    assert jnp.all(jnp.isfinite(T)) and jnp.all(jnp.isfinite(A))
    assert 200.0 < float(T.mean()) < 340.0


def test_loss_is_differentiable():
    data = _synthetic_data()
    g = jax.grad(lambda p: loss_fn(p, data, n_spin_years=1)[0])(init_raw_params())
    assert all(jnp.all(jnp.isfinite(v)) for v in g.values())
    # gradient is non-trivial for at least the albedo params
    assert float(jnp.max(jnp.abs(g["pft_alb"]))) > 0.0


def test_adam_reduces_loss():
    data = _synthetic_data()
    p = init_raw_params()
    vg = jax.value_and_grad(lambda q: loss_fn(q, data, n_spin_years=1)[0])
    opt = optax.adam(3e-2); s = opt.init(p)
    l0 = float(vg(p)[0])
    for _ in range(4):
        l, gr = vg(p); u, s = opt.update(gr, s); p = optax.apply_updates(p, u)
    assert float(l) < l0          # loss decreased


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
