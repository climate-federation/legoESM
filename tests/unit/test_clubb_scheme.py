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

from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    clubb_step,
    clubb_turbulence,
)
from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb_core import CLUBBMomentState  # noqa: E402
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


def _init_moments(ncol, nlev, seed=1):
    """A plausible start-of-step CLUBBMomentState on the CLUBB grids.

    Means on zt (nlev) are placeholders — clubb_step overwrites them from the
    live column; moments/fluxes on zm (nlev+1), wp3 on zt.
    """
    rng = np.random.default_rng(seed)
    nzm = nlev + 1

    def zt(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ncol, nlev)))

    def zm_pos(lo, hi):
        return jnp.asarray(lo + (hi - lo) * rng.random((ncol, nzm)))

    return CLUBBMomentState(
        rtm=zt(1e-3, 9e-3), thlm=zt(2.0, 298.0), um=zt(2.0, 6.0), vm=zt(2.0),
        wp2=zm_pos(0.05, 0.6), wp3=jnp.asarray(0.05 * rng.standard_normal((ncol, nlev))),
        up2=zm_pos(0.1, 0.4), vp2=zm_pos(0.1, 0.4),
        wprtp=jnp.asarray(1e-4 * rng.standard_normal((ncol, nzm))),
        wpthlp=jnp.asarray(1e-2 * rng.standard_normal((ncol, nzm))),
        upwp=jnp.asarray(1e-2 * rng.standard_normal((ncol, nzm))),
        vpwp=jnp.asarray(1e-2 * rng.standard_normal((ncol, nzm))),
        rtp2=zm_pos(1e-8, 2e-6), thlp2=zm_pos(1e-3, 0.1),
        rtpthlp=jnp.asarray(1e-6 * rng.standard_normal((ncol, nzm))))


def test_clubb_step_one_prognostic_step():
    """clubb_step advances the full moment set and returns finite, correctly
    shaped top-down tendencies; the carried moments stay finite/shape-stable."""
    kw = _column(ncol=4, nlev=24)
    kw.pop("tke")
    # Real model columns are grid-consistent (thermo level = midpoint of the
    # bracketing momentum half-levels); _column uses independent linspaces, so
    # rebuild z_full as the half-level midpoint to avoid spurious zt2zm extrapolation.
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    ncol, nlev = kw["T"].shape
    moments = _init_moments(ncol, nlev)
    du, dv, dT, dq, new_moments, diags = clubb_step(moments=moments, **kw)
    for name, t in (("du", du), ("dv", dv), ("dT", dT), ("dq", dq)):
        assert t.shape == (ncol, nlev) and np.all(np.isfinite(np.asarray(t))), name
    for name, v in new_moments._asdict().items():
        arr = np.asarray(v)
        assert np.all(np.isfinite(arr)), name
        assert arr.shape == np.asarray(getattr(moments, name)).shape, name
    # positive-definite variances survive
    for name in ("wp2", "up2", "vp2", "rtp2", "thlp2"):
        assert np.all(np.asarray(getattr(new_moments, name)) >= 0.0), name
    cf = np.asarray(diags["cloud_frac"])
    assert np.all((cf >= 0.0) & (cf <= 1.0)) and np.all(np.asarray(diags["rcm"]) >= 0.0)


def test_clubb_step_surface_drag_decelerates():
    """Surface drag sign check: with a UNIFORM positive zonal wind (no internal
    shear, so the only momentum sink is the surface stress BC u'w'_sfc=tau_x/rho,
    NEGATIVE for u>0), CLUBB must DECELERATE the near-surface wind (du_dt<0). The
    earlier -tau_x/rho sign bug would instead accelerate it."""
    kw = _column(ncol=4, nlev=24)
    kw.pop("tke")
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    ncol, nlev = kw["T"].shape
    moments = _init_moments(ncol, nlev)
    kw["u"] = jnp.full((ncol, nlev), 12.0)   # uniform -> no shear, only drag
    kw["v"] = jnp.zeros((ncol, nlev))
    du, _, _, _, new_moments, _ = clubb_step(moments=moments, **kw)
    # Direct sign guard: the persisted surface (index 0, ascending) u'w' must be
    # NEGATIVE — this fails if upwp_sfc is flipped back to -tau_x/rho. (The
    # advance_windm mean-wind drag uses sqrt(upwp^2+vpwp^2), so du<0 alone is
    # sign-insensitive and cannot guard the fix.)
    assert np.all(np.asarray(new_moments.upwp)[:, 0] < 0.0)
    # Physical consequence: drag decelerates the near-surface wind (top-down last).
    assert np.all(np.asarray(du)[:, -1] < 0.0)


def test_clubb_step_jit_and_grad():
    kw = _column(ncol=3, nlev=20)
    kw.pop("tke")
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    ncol, nlev = kw["T"].shape
    moments = _init_moments(ncol, nlev)

    def loss(T):
        du, dv, dT, dq, ns, _ = clubb_step(moments=moments, **dict(kw, T=T))
        return jnp.sum(dT ** 2) + jnp.sum(ns.wp2 ** 2) + jnp.sum(du ** 2)

    assert jnp.isfinite(jax.jit(loss)(kw["T"]))
    g = jax.grad(loss)(kw["T"])
    assert g.shape == kw["T"].shape and jnp.all(jnp.isfinite(g))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
