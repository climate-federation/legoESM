"""A prescribed surface flux must reach the CLOSURE, not just the column.

A case deck that prescribes its surface heat and moisture fluxes used to have
them injected as a separate tendency on the lowest cell AFTER turbulence ran,
with the bulk exchange coefficient zeroed to avoid double-counting. The
closure therefore computed its own surface flux as exactly zero. Local
closures still see the resulting gradient; every flux-driven NONLOCAL scheme
loses its defining pathway, because YSU and friends derive the convective
velocity scale, PBL depth, entrainment and countergradient from ``shflx``.

``SurfaceLayerConfig.prescribed_shflx_w_m2`` / ``prescribed_lhflx_w_m2`` hand
the deck value to the closure instead. These tests pin that it arrives, that
momentum is deliberately untouched, that it reaches the MOST branch too, and
that leaving it unset is byte-identical to before.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import (  # noqa: E402
    SurfaceLayerConfig,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import (  # noqa: E402
    compute_surface_fluxes,
)

NCOL = 3
# The Nieuwstadt CBL deck's prescribed kinematic heat flux [K m/s].
CBL_W_THETA_K_M_S = 0.06
CBL_RHO = 1.15


def _column():
    u = jnp.full((NCOL,), 6.0)
    v = jnp.full((NCOL,), -2.0)
    T = jnp.full((NCOL,), 295.0)
    q_v = jnp.full((NCOL,), 8.0e-3)
    T_sfc = jnp.full((NCOL,), 300.0)
    q_sfc = jnp.full((NCOL,), 1.5e-2)
    rho = jnp.full((NCOL,), 1.2)
    return u, v, T, q_v, T_sfc, q_sfc, rho


def test_unset_is_byte_identical_to_the_historical_config():
    """Default None must not perturb any existing run."""
    args = _column()
    base = SurfaceLayerConfig(Cd_neutral=1.1e-3, Ch_neutral=1.1e-3)
    explicit_none = SurfaceLayerConfig(
        Cd_neutral=1.1e-3, Ch_neutral=1.1e-3,
        prescribed_shflx_w_m2=None, prescribed_lhflx_w_m2=None)
    a = compute_surface_fluxes(*args, base)
    b = compute_surface_fluxes(*args, explicit_none)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_prescribed_scalar_fluxes_are_returned_verbatim():
    args = _column()
    cfg = SurfaceLayerConfig(
        Cd_neutral=1.1e-3, Ch_neutral=0.0,
        prescribed_shflx_w_m2=9.46, prescribed_lhflx_w_m2=153.4)
    _tx, _ty, shflx, lhflx, _ustar = compute_surface_fluxes(*args, cfg)
    assert np.allclose(np.asarray(shflx), 9.46)
    assert np.allclose(np.asarray(lhflx), 153.4)


def test_the_zeroed_coefficient_is_what_the_override_repairs():
    """Without the override, Ch=0 gives a surface flux of exactly zero.

    This is the defect, stated as a test: it is the state every prescribed-flux
    case ran in.
    """
    args = _column()
    cfg = SurfaceLayerConfig(Cd_neutral=1.1e-3, Ch_neutral=0.0)
    _tx, _ty, shflx, lhflx, _ustar = compute_surface_fluxes(*args, cfg)
    assert np.allclose(np.asarray(shflx), 0.0)
    assert np.allclose(np.asarray(lhflx), 0.0)


def test_momentum_is_not_overridden():
    """Decks that fix scalar fluxes leave the stress interactive."""
    args = _column()
    plain = SurfaceLayerConfig(Cd_neutral=1.1e-3, Ch_neutral=1.1e-3)
    pinned = SurfaceLayerConfig(
        Cd_neutral=1.1e-3, Ch_neutral=1.1e-3,
        prescribed_shflx_w_m2=9.46, prescribed_lhflx_w_m2=153.4)
    a = compute_surface_fluxes(*args, plain)
    b = compute_surface_fluxes(*args, pinned)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))   # tau_x
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))   # tau_y
    assert np.array_equal(np.asarray(a[4]), np.asarray(b[4]))   # ustar
    assert not np.allclose(np.asarray(a[2]), np.asarray(b[2]))  # shflx differs


def test_override_also_applies_on_the_most_branch():
    """The MOST path returns early; it must not skip the prescription."""
    args = _column()
    cfg = SurfaceLayerConfig(
        bulk_scheme="most", z0=1.0e-4, z_ref=20.0,
        prescribed_shflx_w_m2=15.0, prescribed_lhflx_w_m2=115.0)
    _tx, _ty, shflx, lhflx, _ustar = compute_surface_fluxes(*args, cfg)
    assert np.allclose(np.asarray(shflx), 15.0)
    assert np.allclose(np.asarray(lhflx), 115.0)


def test_only_one_channel_can_be_prescribed():
    """Sensible-only is legitimate: the dry analytic cases have no latent flux."""
    args = _column()
    cfg = SurfaceLayerConfig(
        Cd_neutral=1.1e-3, Ch_neutral=1.1e-3,
        prescribed_shflx_w_m2=-7.5, prescribed_lhflx_w_m2=None)
    plain = SurfaceLayerConfig(Cd_neutral=1.1e-3, Ch_neutral=1.1e-3)
    _tx, _ty, shflx, lhflx, _ = compute_surface_fluxes(*args, cfg)
    ref = compute_surface_fluxes(*args, plain)
    assert np.allclose(np.asarray(shflx), -7.5)
    assert np.array_equal(np.asarray(lhflx), np.asarray(ref[3]))


def test_ysu_convective_pathway_switches_on_with_the_flux():
    """The consequence, on the scheme codex named.

    YSU builds its convective velocity scale from shflx. With the historical
    Ch=0 the dry convective case runs it with no convection at all; handing it
    the deck flux must change its tendencies.
    """
    from legoesm.atmosphere.physics.turbulence.config import YSUConfig
    from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence

    nlev = 24
    ncol = 1
    z_half = jnp.linspace(1600.0, 0.0, nlev + 1)[None, :]
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    p_half = jnp.linspace(8.0e4, 1.0e5, nlev + 1)[None, :]
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    # Well-mixed dry column, zero wind: the CBL case's defining configuration.
    T = jnp.full((ncol, nlev), 300.0)
    u = jnp.zeros((ncol, nlev))
    v = jnp.zeros((ncol, nlev))
    q_v = jnp.full((ncol, nlev), 1.0e-3)
    rho = jnp.full((ncol, nlev), CBL_RHO)
    T_sfc = jnp.full((ncol,), 300.0)
    q_sfc = jnp.full((ncol,), 1.0e-3)

    deck_shf = CBL_W_THETA_K_M_S * CBL_RHO * constants.c_pd
    off = YSUConfig(surface=SurfaceLayerConfig(
        Cd_neutral=7.5e-3, Ch_neutral=0.0))
    on = YSUConfig(surface=SurfaceLayerConfig(
        Cd_neutral=7.5e-3, Ch_neutral=0.0,
        prescribed_shflx_w_m2=deck_shf, prescribed_lhflx_w_m2=0.0))

    out_off = ysu_turbulence(u, v, T, q_v, p_full, p_half, z_full, z_half,
                             T_sfc, q_sfc, rho, 10.0, off)
    out_on = ysu_turbulence(u, v, T, q_v, p_full, p_half, z_full, z_half,
                            T_sfc, q_sfc, rho, 10.0, on)

    assert float(out_off.shflx[0]) == pytest.approx(0.0), (
        "Ch=0 with no prescription gives the closure a zero surface flux -- "
        "this is the defect being repaired")
    assert float(out_on.shflx[0]) == pytest.approx(deck_shf)

    # h_pbl is the sharp discriminator: YSU's convective branch grows the PBL
    # from the surface buoyancy flux, so with no flux it sits at its floor.
    # (dT_dt is NOT exactly zero without the flux -- residual background
    # diffusion acts on the initial profile at ~1e-7 K/s -- so asserting an
    # exact zero there would be asserting the wrong thing.)
    h_off = float(out_off.h_pbl[0])
    h_on = float(out_on.h_pbl[0])
    assert h_on > h_off, (
        f"the prescribed surface flux must deepen the convective PBL: "
        f"h_pbl {h_off:.1f} -> {h_on:.1f} m")

    amp_off = float(np.max(np.abs(np.asarray(out_off.dT_dt))))
    amp_on = float(np.max(np.abs(np.asarray(out_on.dT_dt))))
    assert amp_on > 100.0 * amp_off, (
        f"the flux-driven tendency must dominate the residual background "
        f"diffusion: {amp_on:.3e} vs {amp_off:.3e} K/s")
