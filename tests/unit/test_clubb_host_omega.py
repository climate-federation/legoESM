"""clubb_step: CLUBB's mean vertical velocity from the host omega, and the
l_implemented = .true. call, as CAM's clubb_intr does.

CESM2.1 (clubb_intr.F90:1676-1679, 1778, cam_cesm2_1_rel_60):
``wm_zt = -omega / (rho_ds_zt * gravit)`` (no ``omega_bot``), ``wm_zm =
zt2zm(wm_zt)``; CLUBB is called with ``l_implemented = .true.`` (:113).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.turbulence import clubb as clubb_mod  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBConfig, clubb_step, init_clubb_moments, zt2zm)

from tests.unit.test_clubb_liquid_partition import _NCOL, _NLEV, _column  # noqa: E402

_CFG = CLUBBConfig(trop_cloud_top_press=0.0, prognostic=True)


def _omega():
    """Top-down (ncol, nlev) [Pa/s]: sinking aloft, rising near the ground."""
    s = (np.arange(_NLEV) + 0.5) / _NLEV
    return jnp.asarray(np.broadcast_to(0.3 * np.sin(np.pi * s) - 0.1 * s,
                                       (_NCOL, _NLEV)).copy())


def _run(col, omega, monkeypatch=None, got=None):
    if monkeypatch is not None:
        real_core = clubb_mod.advance_clubb_core

        def core(*a, **k):
            got.update(k)
            return real_core(*a, **k)

        monkeypatch.setattr(clubb_mod, "advance_clubb_core", core)
    return clubb_step(col["u"], col["v"], col["T"], col["q_v"],
                      init_clubb_moments(_NCOL, _NLEV, _CFG, dtype=jnp.float64),
                      col["p_full"], col["p_half"], col["z_full"], col["z_half"],
                      col["T_sfc"], col["q_sfc"], col["rho"], 300.0, _CFG,
                      omega=omega)


def test_wm_is_minus_omega_over_rho_g_and_core_is_implemented(monkeypatch):
    col, got = _column(), {}
    omega = _omega()
    _run(col, omega, monkeypatch, got)
    rho_ds_zt = jnp.flip(col["rho"], axis=1)
    wm_zt = -jnp.flip(omega, axis=1) / (rho_ds_zt * constants.g)
    np.testing.assert_array_equal(np.asarray(got["wm_zt"]), np.asarray(wm_zt))
    np.testing.assert_array_equal(np.asarray(got["wm_zm"]),
                                  np.asarray(zt2zm(wm_zt, got["gr"])))
    assert got["l_implemented"] is True
    # Sign: omega > 0 (sinking) -> wm < 0, and vice versa.
    w = np.asarray(got["wm_zt"])
    o = np.asarray(jnp.flip(omega, axis=1))
    assert np.all(np.sign(w[o != 0]) == -np.sign(o[o != 0]))


def test_no_omega_keeps_wm_zero(monkeypatch):
    got = {}
    _run(_column(), None, monkeypatch, got)
    assert not np.any(np.asarray(got["wm_zt"])) and not np.any(np.asarray(got["wm_zm"]))
    assert got["l_implemented"] is True


def test_zero_omega_is_bit_identical_to_none():
    col = _column()
    a = _run(col, None)
    b = _run(col, jnp.zeros_like(col["T"]))
    for x, y in zip(a[:4], b[:4]):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_omega_changes_the_step_but_not_the_column_water():
    """wm reaches the step (flux/variance equations) while the host keeps the
    mean advection: the flux-form rtm advance telescopes, so the column
    (rho dz) total-water change is the same with and without omega.  The
    standalone path's advective-form mean advection would change it.  Bound
    as test_advance_clubb_core_conserves_thlm_rtm (1e-9 relative)."""
    col = _column()
    a = _run(col, None)
    b = _run(col, _omega())
    assert float(jnp.max(jnp.abs(a[3] - b[3]))) > 0.0
    dz = -jnp.diff(col["z_half"], axis=1)          # top-down, positive
    w = col["rho"] * dz
    ca = jnp.sum(a[3] * w, axis=1)
    cb = jnp.sum(b[3] * w, axis=1)
    rel = float(jnp.max(jnp.abs(cb - ca) / jnp.abs(ca)))
    assert rel < 1e-9, rel


def test_grad_through_omega_is_finite():
    col = _column()

    def loss(omega):
        out = _run(col, omega)
        return jnp.sum(out[2] ** 2) + jnp.sum(out[3] ** 2)

    g = jax.grad(loss)(_omega())
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.max(jnp.abs(g))) > 0.0


def test_mpas_column_lane_calls_the_core_implemented(monkeypatch):
    """The MPAS turbulence factory (the MPAS _run_column lane and the FV3-duo
    column lane both build their physics with make_physics(model_type="mpas"))
    reaches the core with l_implemented = .true.  No host omega reaches it yet
    on this lane, so wm is zero there (pending the dycore-omega decision)."""
    import types

    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas)
    from legoesm.atmosphere.physics.turbulence.clubb import pack_clubb_moments
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.integration import (
        _make_mpas_turbulence)
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    sigma = create_sigma_coordinate(8, sigma_top=0.05)
    state = held_suarez_init_mpas(mesh, sigma, T_init=290.0,
                                  perturbation_amplitude=0.0)
    ncell, nlev = state.T.data.shape
    state = state._replace(tracers={"q_v": jnp.full((ncell, nlev), 0.008,
                                                    dtype=state.T.data.dtype)})
    cfg = CLUBBConfig(prognostic=True)
    seen = []
    real_core = clubb_mod.advance_clubb_core

    def core(*a, **k):
        seen.append((k["l_implemented"], np.asarray(k["wm_zt"])))
        return real_core(*a, **k)

    monkeypatch.setattr(clubb_mod, "advance_clubb_core", core)
    fn = _make_mpas_turbulence(TurbulenceConfig(scheme="clubb", clubb=cfg), 300.0)
    phys_state = types.SimpleNamespace(
        clubb_moments=pack_clubb_moments(init_clubb_moments(ncell, nlev, cfg)))
    fn(state, mesh, sigma, phys_state=phys_state,
       forcing={"T_sfc": jnp.full((ncell,), 295.0)})
    assert seen, "the CLUBB core was not called"
    assert all(impl is True and not np.any(wm) for impl, wm in seen)
