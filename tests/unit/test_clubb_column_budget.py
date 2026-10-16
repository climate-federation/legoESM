"""Prognostic CLUBB closes the column energy and water budgets (review 2026-10-11).

Measured on the production MPAS deck (day-1 checkpoint, one CLUBB call of
600 s = 2 sub-steps): after the KE heating (F14) the column total energy
``sum(dp/g (c_pd T + KE + L_v q_v))`` still gained +0.21..0.24 W/m2 global
mean beyond the surface supply ``shflx + lhflx`` -- CLUBB conserves the
mass-weighted theta-l, not the host enthalpy -- and the sub-cycle created
1.5e-4 kg/m2/day of water because each sub-step re-derived the density from
the evolving virtual temperature with the heights held fixed.

Pinned here (x64), on a column built with the lane's own height/density
helpers (so CLUBB's ``rho*dz`` equals the host ``dp/g``, as on the lane):

* energy: the change of the INDEPENDENT diagnostic
  ``process_ledger.column_total_energy`` across the host update
  ``x += dt*dx_dt`` equals ``(shflx + lhflx)*dt`` to round-off (CAM6
  ``clubb_do_energyfix``); a control with the fixer patched to zero shows the
  residual the fixer removes is real;
* water: ``sum(dp/g (dq_v + dq_c))*dt == evap*dt`` to round-off with
  sub-cycling on (CAM holds ``rho_ds`` through its ``nadv`` loop).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics._shared import (  # noqa: E402
    column_energy_fixer_heating,
    compute_heights_from_sigma,
    compute_rho,
)
from legoesm.atmosphere.physics.turbulence import clubb as clubb_mod  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBConfig,
    clubb_turbulence_prognostic,
    init_clubb_moments,
    pack_clubb_moments,
)
from legoesm.diagnostics.process_ledger import column_total_energy  # noqa: E402

DT = 600.0          # the production CLUBB call (1800 s / 3 macmic)
CLUBB_DT = 300.0    # -> 2 sub-steps, the production sub-cycle


def _column(ncol=3, nlev=24, seed=0, p_top=2.0e4, log_p=False):
    """Sheared, stably stratified moist column with a thin cloud layer; heights
    and density from the lane's helpers so sum(rho*dz) == sum(dp/g)."""
    rng = np.random.default_rng(seed)
    space = np.geomspace if log_p else np.linspace
    p_half = space(p_top, 1.0e5, nlev + 1)[None, :] * np.ones((ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    exner = (p_full / constants.p_ref) ** constants.kappa
    theta = 290.0 + 25.0 * (1.0 - p_full / 1.0e5) + 0.3 * rng.standard_normal((ncol, nlev))
    T = theta * exner
    q_v = 1.2e-2 * (p_full / 1.0e5) ** 3
    q_c = np.where(np.abs(p_full - 8.5e4) < 4.0e3, 2.0e-4, 0.0)
    u = 8.0 + 4.0 * rng.standard_normal((ncol, nlev))
    v = 3.0 + 2.0 * rng.standard_normal((ncol, nlev))
    a = lambda x: jnp.asarray(x, dtype=jnp.float64)  # noqa: E731
    T, q_v, p_half, p_full = a(T), a(q_v), a(p_half), a(p_full)
    z_full, z_half = compute_heights_from_sigma(T, p_half, q_v=q_v)
    rho = compute_rho(T, p_full, q_v=q_v)
    return dict(u=a(u), v=a(v), T=T, q_v=q_v, q_c=a(q_c), p_full=p_full,
                p_half=p_half, z_full=z_full, z_half=z_half,
                T_sfc=T[:, -1] + 2.0, q_sfc=1.1 * q_v[:, -1], rho=rho,
                lat=jnp.zeros((ncol,)))           # equator [rad]


def _run(kw, dt=DT, cfg=None, carry=None, with_carry=False):
    cfg = cfg or CLUBBConfig(prognostic=True, clubb_dt=CLUBB_DT,
                             trop_cloud_top_press=15000.0)
    ncol, nlev = kw["T"].shape
    if carry is None:
        carry = pack_clubb_moments(init_clubb_moments(ncol, nlev, cfg))
    out, carry = clubb_turbulence_prognostic(
        kw["u"], kw["v"], kw["T"], kw["q_v"], carry, kw["p_full"],
        kw["p_half"], kw["z_full"], kw["z_half"], kw["T_sfc"], kw["q_sfc"],
        kw["rho"], dt, cfg, q_c=kw["q_c"], lat=kw["lat"])
    return (out, carry) if with_carry else out


def _energy_residual(kw, out, dt=DT):
    """[W/m2, + = created] of the host update against shflx + lhflx."""
    p_s = kw["p_half"][:, -1]
    dp = kw["p_half"][:, 1:] - kw["p_half"][:, :-1]
    dsig = dp / p_s[:, None]

    def energy(T, u, v, q_v):
        return np.asarray(column_total_energy(p_s, dsig, T, u=u, v=v, q_v=q_v, dp=dp))

    e0 = energy(kw["T"], kw["u"], kw["v"], kw["q_v"])
    e1 = energy(kw["T"] + dt * out.dT_dt, kw["u"] + dt * out.du_dt,
                kw["v"] + dt * out.dv_dt, kw["q_v"] + dt * out.dq_v_dt)
    return (e1 - e0) / dt - np.asarray(out.shflx + out.lhflx)


def test_fixer_helper_is_exact():
    rng = np.random.default_rng(1)
    resid = jnp.asarray(rng.standard_normal(4) * 3.0)
    m = jnp.asarray(rng.random((4, 20)) * 300.0 + 10.0)
    w = jnp.asarray(rng.random((4, 20)))
    heat = column_energy_fixer_heating(resid, m, w)
    np.testing.assert_allclose(
        np.sum(np.asarray(constants.c_pd * heat * m), axis=1), -np.asarray(resid),
        rtol=1e-12)


# 2.0e3 Pa: a column reaching well above the 15000 Pa cutoff (CAM top_lev).
@pytest.mark.parametrize("p_top", [2.0e4, 2.0e3])
def test_prognostic_clubb_closes_column_energy(monkeypatch, p_top):
    kw = _column(p_top=p_top)
    out = _run(kw)
    flux = np.asarray(out.shflx + out.lhflx)
    assert np.all(np.abs(flux) > 10.0), flux          # a real surface supply
    res = _energy_residual(kw, out)
    assert np.all(np.abs(res) < 1e-7), res            # W/m2, round-off
    # Control: the same kernel with the fixer zeroed leaves a real residual.
    monkeypatch.setattr(clubb_mod, "column_energy_fixer_heating",
                        lambda r, m, w: jnp.zeros_like(w * r[..., None]))
    jax.clear_caches()
    res_off = _energy_residual(kw, _run(kw))
    monkeypatch.undo()
    jax.clear_caches()
    assert np.max(np.abs(res_off)) > 1e-3, res_off   # ~1e-2 W/m2 measured


@pytest.mark.parametrize("p_top", [2.0e4, 2.0e3])
@pytest.mark.parametrize("dt", [CLUBB_DT, DT])
def test_prognostic_clubb_conserves_column_water(dt, p_top):
    """Total water (vapour + the partitioned liquid) changes by exactly the
    surface evaporation, with and without sub-cycling."""
    kw = _column(p_top=p_top)
    out = _run(kw, dt=dt)
    m = np.asarray((kw["p_half"][:, 1:] - kw["p_half"][:, :-1]) / constants.g)
    dW = np.sum(m * np.asarray(out.dq_v_dt + out.dq_c_dt), axis=1)
    evap = np.asarray(out.evap_sfc)
    assert np.all(evap > 1e-6), evap                  # a real surface supply
    np.testing.assert_allclose(dW, evap, rtol=1e-9, atol=0.0)


_OFF = CLUBBConfig(prognostic=True, clubb_dt=CLUBB_DT)   # no cutoff (CAM top_lev = 1)


def test_prognostic_clubb_leaves_the_state_above_the_cutoff_alone():
    """CAM runs CLUBB only on k >= top_lev (clubb_intr.F90): u, v, thl, rt and
    the host liquid keep the host state above it.  Without the cutoff the same
    column IS mixed there, so the comparison cannot pass vacuously."""
    kw = _column(p_top=2.0e3, nlev=30, log_p=True)
    above = np.asarray(kw["p_full"][0]) < 5.0e3      # taper weight < 7e-4
    assert above.sum() >= 3
    carry = None                                     # moments spun up aloft
    for _ in range(4):
        _, carry = _run(kw, cfg=_OFF, carry=carry, with_carry=True)
    on, off = _run(kw, carry=carry), _run(kw, cfg=_OFF, carry=carry)
    for name, floor in (("du_dt", 1e-4), ("dv_dt", 1e-4), ("dT_dt", 1e-4),
                        ("dq_v_dt", 1e-12)):
        x_on = np.abs(np.asarray(getattr(on, name))[:, above]) * DT
        x_off = np.abs(np.asarray(getattr(off, name))[:, above]) * DT
        assert x_off.max() > floor, (name, x_off.max())
        assert x_on.max() < 1e-3 * x_off.max(), (name, x_on.max(), x_off.max())


def test_cutoff_returns_the_removed_momentum_to_the_column():
    """The truncated CAM grid passes no flux across top_lev, so the column
    momentum changes only by the surface stress.  One CLUBB step (no
    sub-cycle) runs the identical solve with and without the cutoff, so the
    column sums must agree to round-off."""
    kw = _column(p_top=2.0e3, nlev=30)
    on, off = _run(kw, dt=CLUBB_DT), _run(kw, dt=CLUBB_DT, cfg=_OFF)
    m = np.asarray((kw["p_half"][:, 1:] - kw["p_half"][:, :-1]) / constants.g)
    for name in ("du_dt", "dv_dt"):
        s_on = np.sum(m * np.asarray(getattr(on, name)), axis=1)
        s_off = np.sum(m * np.asarray(getattr(off, name)), axis=1)
        np.testing.assert_allclose(s_on, s_off, rtol=1e-10, atol=1e-14)


def test_water_pushed_down_from_above_the_cutoff_cannot_dry_a_layer_negative():
    """CAM apply_to_surface: when the column water returned from above the
    cutoff is NEGATIVE (liquid aloft mixed down) and spreading it would drive
    any q_v negative, it all goes into the bottom layer instead."""
    kw = _column(p_top=2.0e3, nlev=30)
    p = np.asarray(kw["p_full"])
    kw["q_v"] = jnp.zeros_like(kw["q_v"])
    kw["q_c"] = jnp.asarray(np.where(p < 5.0e3, 1.0e-4, 0.0))
    kw["q_sfc"] = jnp.zeros_like(kw["q_sfc"])
    out = _run(kw, dt=CLUBB_DT)
    q_new = np.asarray(kw["q_v"] + CLUBB_DT * out.dq_v_dt)
    assert np.all(q_new[:, :-1] >= 0.0), q_new.min(axis=1)


def test_removed_water_lands_and_is_tested_only_below_clubbtop():
    """CAM clubb_intr.F90 (cesm2_1_rel 2075-2095) spreads the returned water
    uniformly per unit mass over k = clubbtop+1..pver and tests negativity on
    those same levels only; otherwise it all goes into the bottom layer."""
    below = jnp.array([[False, True, True]])              # clubbtop = 0
    dp_g = jnp.array([[1.0e3, 4.0e3, 4.0e3]])
    dry, wet = -1.0e-3, 1.0e-3                             # removed [kg/m2/s]

    def run(dq, removed, q_v):
        return np.asarray(clubb_mod.return_removed_water(
            jnp.array([dq]), jnp.array([[removed]]), dp_g,
            jnp.array([q_v]), 1.0, below))[0]

    # Nothing lands above clubbtop; a level above it going negative (here
    # from its own tendency) is not tested, as in CAM.
    np.testing.assert_allclose(
        run([-1.0e-9, 0.0, 0.0], dry, [0.0, 1.0e-2, 1.0e-2]),
        [-1.0e-9, dry / 8.0e3, dry / 8.0e3], rtol=1e-12)
    # A level below clubbtop would go negative -> all into the bottom.
    np.testing.assert_allclose(
        run([0.0, 0.0, 0.0], dry, [1.0e-2, 0.0, 1.0e-2]),
        [0.0, 0.0, dry / 4.0e3], rtol=1e-12)
    # Positive return never falls back.
    np.testing.assert_allclose(
        run([0.0, 0.0, 0.0], wet, [0.0, 0.0, 0.0]),
        [0.0, wet / 8.0e3, wet / 8.0e3], rtol=1e-12)
