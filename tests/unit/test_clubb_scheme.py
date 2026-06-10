"""End-to-end tests for the runnable CLUBB turbulence scheme (``clubb.py``).

Confirms ``scheme="clubb"`` dispatches, runs through the column interface, is
JIT/grad-clean, and behaves physically (positive diffusivities, more vigorous
mixing in unstable than stable columns). The full prognostic moment coupling is
deepened in later phases (see ``PORT_CLUBB.md``); this locks the runnable entry.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import clubb_turbulence  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn  # noqa: E402
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput  # noqa: E402

from legoesm import constants  # noqa: E402


def _column(ncol=4, nlev=24, dtheta_dz=4e-3, seed=0):
    """Synthetic top-down column; dtheta_dz controls static stability."""
    rng = np.random.default_rng(seed)
    p_half = np.linspace(2.0e4, 1.0e5, nlev + 1)[None, :] * np.ones((ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    z_full = jnp.asarray(np.tile(np.linspace(15000.0, 50.0, nlev), (ncol, 1)))
    z_half = jnp.asarray(np.tile(np.linspace(16000.0, 0.0, nlev + 1), (ncol, 1)))
    exner = (p_full / constants.p_ref) ** constants.kappa
    # theta increasing upward (height increases with decreasing index) by dtheta_dz.
    z = np.asarray(z_full)
    theta = 290.0 + dtheta_dz * z
    T = jnp.asarray(theta * np.asarray(exner))
    u = jnp.asarray(8.0 + 4.0 * rng.standard_normal((ncol, nlev)))
    v = jnp.asarray(2.0 * rng.standard_normal((ncol, nlev)))
    q_v = jnp.asarray(2e-3 + 4e-3 * rng.random((ncol, nlev)))
    p_full = jnp.asarray(p_full)
    p_half = jnp.asarray(p_half)
    rho = p_full / (constants.R_d * T)
    return dict(u=u, v=v, T=T, q_v=q_v, tke=jnp.full((ncol, nlev), 0.4),
                p_full=p_full, p_half=p_half, z_full=z_full, z_half=z_half,
                T_sfc=T[:, -1] + 1.0, q_sfc=q_v[:, -1], rho=rho, dt=300.0,
                config=CLUBBConfig())


def test_dispatch_selects_clubb():
    name, fn, cfg = get_turbulence_fn(TurbulenceConfig(scheme="clubb"))
    assert name == "clubb"
    assert fn is clubb_turbulence
    assert isinstance(cfg, CLUBBConfig)


def test_runs_and_outputs_are_physical():
    out, wp2_new = clubb_turbulence(**_column())
    assert isinstance(out, TurbulenceOutput)
    ncol, nlev = _column()["T"].shape
    for arr in (out.du_dt, out.dv_dt, out.dT_dt, out.dq_v_dt, out.Km, out.Kh):
        assert arr.shape == (ncol, nlev)
        assert jnp.all(jnp.isfinite(arr))
    for arr in (out.shflx, out.lhflx, out.ustar, out.h_pbl):
        assert arr.shape == (ncol,)
        assert jnp.all(jnp.isfinite(arr))
    # Positive diffusivities and TKE.
    assert jnp.all(out.Km > 0.0) and jnp.all(out.Kh > 0.0)
    assert jnp.all(wp2_new > 0.0)
    assert wp2_new.shape == (ncol, nlev)


def test_unstable_mixes_more_than_stable():
    """Unstable stratification -> larger eddy diffusivity than strongly stable."""
    out_unstable, _ = clubb_turbulence(**_column(dtheta_dz=-2e-3))
    out_stable, _ = clubb_turbulence(**_column(dtheta_dz=1.5e-2))
    assert float(jnp.mean(out_unstable.Km)) > float(jnp.mean(out_stable.Km))


def test_custom_config_is_used():
    """A custom CLUBBConfig threads through TurbulenceConfig.clubb."""
    cfg = CLUBBConfig()
    name, fn, scheme_cfg = get_turbulence_fn(TurbulenceConfig(scheme="clubb", clubb=cfg))
    assert scheme_cfg is cfg


def test_jit_and_grad():
    kw = _column()

    def loss(T):
        out, wp2 = clubb_turbulence(**dict(kw, T=T))
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.Km) + jnp.sum(wp2)

    assert jnp.isfinite(jax.jit(loss)(kw["T"]))
    g = jax.grad(loss)(kw["T"])
    assert g.shape == kw["T"].shape and jnp.all(jnp.isfinite(g))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
