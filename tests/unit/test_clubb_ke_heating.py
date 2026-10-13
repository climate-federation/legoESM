"""CLUBB returns the kinetic energy its momentum mixing removes as heat (F14).

Review 2026-10-10 F14: CLUBB's momentum mixing and the surface stress it applies
as its lower boundary condition remove resolved kinetic energy that was never
returned as heat, a systematic total-energy sink on every CLUBB lane.  CAM6
closes it with ``clubb_do_energyfix`` (default .true.).  The port is the local
discrete closure ``_shared.ke_dissipation_heating`` applied inside the kernel
(always on, like CAM's default; no switch: CLUBB's flags are the fixed CAM
defaults).  The heating-off reference below patches the helper to zero in the
clubb module, so both arms run the same code.

Invariant pinned here (x64): with the heating on, the column
``sum((c_pd*T + KE) dp/g)`` change attributable to the momentum tendency is
returned exactly -- the extra heating integrates to minus the column KE change
of the applied update ``u += dt*du``, to round-off.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics._shared import ke_dissipation_heating  # noqa: E402
from legoesm.atmosphere.physics.turbulence import clubb as clubb_mod  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBConfig,
    clubb_turbulence,
    clubb_turbulence_prognostic,
    init_clubb_moments,
    pack_clubb_moments,
)


def _column(ncol=3, nlev=24, seed=0):
    """Top-down sheared column with a surface wind (so the drag BC is active)
    and a near-neutral surface (small heat flux)."""
    rng = np.random.default_rng(seed)
    p_half = np.linspace(2.0e4, 1.0e5, nlev + 1)[None, :] * np.ones((ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    z_half = np.tile(np.linspace(16000.0, 0.0, nlev + 1), (ncol, 1))
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    exner = (p_full / constants.p_ref) ** constants.kappa
    T = (290.0 + 4e-3 * z_full) * exner
    u = 8.0 + 4.0 * rng.standard_normal((ncol, nlev))
    v = 3.0 + 2.0 * rng.standard_normal((ncol, nlev))
    q_v = 2e-3 + 4e-3 * rng.random((ncol, nlev))
    rho = p_full / (constants.R_d * T)
    a = lambda x: jnp.asarray(x)  # noqa: E731
    return dict(u=a(u), v=a(v), T=a(T), q_v=a(q_v), p_full=a(p_full),
                p_half=a(p_half), z_full=a(z_full), z_half=a(z_half),
                T_sfc=a(T[:, -1] + 0.5), q_sfc=a(q_v[:, -1]), rho=a(rho))


def _dp(kw):
    ph = np.asarray(kw["p_half"])
    return ph[:, 1:] - ph[:, :-1]


def _column_ke_rate(kw, du, dv, dt):
    """Column KE change [W/m^2] of the applied update u += dt*du."""
    u, v = np.asarray(kw["u"]), np.asarray(kw["v"])
    du, dv = np.asarray(du), np.asarray(dv)
    ke0 = 0.5 * (u ** 2 + v ** 2)
    ke1 = 0.5 * ((u + dt * du) ** 2 + (v + dt * dv) ** 2)
    return np.sum((ke1 - ke0) * _dp(kw), axis=1) / (constants.g * dt)


def _run(kw, cfg, dt, heating=True, monkeypatch=None):
    if not heating:              # the same kernel with the KE heating zeroed
        monkeypatch.setattr(clubb_mod, "ke_dissipation_heating",
                            lambda u, v, du, dv, dt: jnp.zeros_like(u))
        jax.clear_caches()
        try:
            return _run(kw, cfg, dt)
        finally:
            monkeypatch.undo()
            jax.clear_caches()
    if cfg.prognostic:
        ncol, nlev = kw["T"].shape
        carry = pack_clubb_moments(init_clubb_moments(ncol, nlev, cfg))
        out, _ = clubb_turbulence_prognostic(
            kw["u"], kw["v"], kw["T"], kw["q_v"], carry, kw["p_full"],
            kw["p_half"], kw["z_full"], kw["z_half"], kw["T_sfc"], kw["q_sfc"],
            kw["rho"], dt, cfg)
        return out
    out, _ = clubb_turbulence(
        kw["u"], kw["v"], kw["T"], kw["q_v"],
        jnp.full(kw["T"].shape, 0.4), kw["p_full"], kw["p_half"], kw["z_full"],
        kw["z_half"], kw["T_sfc"], kw["q_sfc"], kw["rho"], dt, cfg)
    return out


def test_helper_is_discretely_exact():
    """sum(c_pd*heat*dp/g) == -column KE change of u += dt*du, to round-off."""
    rng = np.random.default_rng(3)
    shape = (5, 30)
    u, v, du, dv = (jnp.asarray(rng.standard_normal(shape) * s)
                    for s in (10.0, 10.0, 1e-3, 1e-3))
    dp = jnp.asarray(rng.random(shape) * 3000.0 + 100.0)
    dt = 600.0
    heat = ke_dissipation_heating(u, v, du, dv, dt)
    col_heat = np.sum(np.asarray(constants.c_pd * heat * dp), axis=1)
    ke0 = 0.5 * (u ** 2 + v ** 2)
    ke1 = 0.5 * ((u + dt * du) ** 2 + (v + dt * dv) ** 2)
    col_dke = np.sum(np.asarray((ke1 - ke0) * dp), axis=1) / dt
    np.testing.assert_allclose(col_heat, -col_dke, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("prognostic", [False, True])
def test_clubb_returns_its_ke_loss_as_heat(prognostic, monkeypatch):
    """The extra heating equals the KE the step removes (drag + mixing).

    dt <= clubb_dt keeps the prognostic path at one sub-step, so the momentum
    tendency is identical with the heating on or off (it cannot feed back
    within the step) and the comparison isolates the closure.  The prognostic
    column energy fixer (CAM clubb_do_energyfix, test_clubb_column_budget.py)
    would return the same column total in BOTH arms, so it is zeroed here in
    both to isolate the local heating."""
    kw = _column()
    dt = 150.0
    cfg_on = CLUBBConfig(prognostic=prognostic, clubb_dt=300.0)
    monkeypatch.setattr(clubb_mod, "column_energy_fixer_heating",
                        lambda r, m, w: jnp.zeros_like(w * r[..., None]))
    jax.clear_caches()
    on = _run(kw, cfg_on, dt)
    off = _run(kw, cfg_on, dt, heating=False, monkeypatch=monkeypatch)
    np.testing.assert_array_equal(np.asarray(on.du_dt), np.asarray(off.du_dt))
    np.testing.assert_array_equal(np.asarray(on.dv_dt), np.asarray(off.dv_dt))
    dke = _column_ke_rate(kw, on.du_dt, on.dv_dt, dt)        # W/m^2
    # non-vacuous: the step genuinely dissipates KE in every column
    assert np.all(dke < -1e-2), dke
    extra_heat = np.sum(constants.c_pd * np.asarray(on.dT_dt - off.dT_dt)
                        * _dp(kw), axis=1) / constants.g    # W/m^2
    np.testing.assert_allclose(extra_heat, -dke, rtol=1e-11, atol=0.0)


@pytest.mark.parametrize("prognostic", [False, True])
def test_ke_heating_is_differentiable(prognostic, monkeypatch):
    """jax.grad through the new heating is finite and the heating carries
    gradient (d/du of the column heating differs from the heating-off path)."""
    kw = _column(ncol=2, nlev=16)
    dt = 150.0
    cfg = CLUBBConfig(prognostic=prognostic, clubb_dt=300.0)

    def loss(u, heating):
        out = _run(dict(kw, u=u), cfg, dt, heating, monkeypatch)
        return jnp.sum(out.dT_dt * jnp.asarray(_dp(kw)))

    g_on = jax.grad(loss)(kw["u"], True)
    g_off = jax.grad(loss)(kw["u"], False)
    assert np.all(np.isfinite(np.asarray(g_on)))
    assert float(np.max(np.abs(np.asarray(g_on - g_off)))) > 0.0
