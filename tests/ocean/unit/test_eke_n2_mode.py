"""Tests for the EKE/Visbeck Eady-chain ``n2_mode`` (in-situ vs adiabatic N²).

The legacy ``compute_buoyancy_frequency`` (in-situ ∂_zρ) carries the adiabatic
compressibility term and is biased ~6x too stable, which inflates the column
buoyancy integral ∫N dz → the Rossby/deformation radius → the EKE mixing length
``eke_len``. A too-long eke_len weakens the EKE dissipation (∝ 1/L) and is the
dominant lever of the Phase-G EKE runaway. ``n2_mode="adiabatic"`` switches the
Eady chain to the Veros parcel-displacement N² (``compute_buoyancy_frequency_
adiabatic``).

These tests cover:
  1. default-off BYTE-IDENTITY: the explicit-kwargs (None) path == the no-kwargs
     path bit-exactly, for both ``_eady_growth_and_length`` and the public
     ``compute_eke_kappa_gm`` / ``compute_visbeck_kappa_gm``.
  2. compressibility-bias DIRECTION: on a synthetic stratified column with the
     nonlinear Wright EOS, the adiabatic-N² eke_len is strictly SHORTER than the
     in-situ-N² eke_len (both finite + positive).
  3. dispatch discipline: unknown ``n2_mode`` raises; ``"adiabatic"`` without
     T/S/p_cell/eos raises.
  4. AD: ``jax.grad`` through ``compute_eke_kappa_gm`` in adiabatic mode is
     finite.

Run with:
    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_eke_n2_mode.py -v
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ocean.eos import (
    compute_hydrostatic_pressure,
    make_eos_fn,
    wright_eos,
)
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    _eady_growth_and_length,
    compute_eke_kappa_gm,
    compute_visbeck_kappa_gm,
)
from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig


# ---------------------------------------------------------------------------
# Synthetic stratified column with a nonlinear EOS
# ---------------------------------------------------------------------------

def _column(nlev=20, n_lat=3, n_lon=4, H_max=4000.0):
    """A statically-stable, meridionally-tilted T/S column on a small grid.

    Returns (z_coord, jacobian, f_coriolis, beta, T, S, rho, p_cell, eos_fn).
    Deep, warm-over-cold + a meridional gradient so |S| > 0 and N² > 0 with a
    genuine compressibility contrast between the in-situ and adiabatic forms.
    """
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=H_max)
    shape = (n_lat, n_lon, nlev)
    jacobian = jnp.ones((n_lat, n_lon))
    eos_fn = wright_eos

    # Warm, salty surface -> cold, fresh deep (stable). Add a meridional T
    # gradient so the isopycnal slopes are nonzero.
    Tz = jnp.linspace(18.0, 2.0, nlev)
    Sz = jnp.linspace(35.5, 34.6, nlev)
    lat_grad = 2.0 * (jnp.arange(n_lat, dtype=jnp.float64) - n_lat / 2.0)
    T = (Tz[None, None, :] + lat_grad[:, None, None]
         + jnp.zeros(shape, dtype=jnp.float64))
    S = jnp.broadcast_to(Sz[None, None, :], shape) + jnp.zeros(shape)

    # eta = 0; cell-centre pressure from the in-situ density column.
    eta = jnp.zeros((n_lat, n_lon))
    rho = eos_fn(
        T, S,
        # in-situ pressure approximated by depth pressure for the column build
        constants.rho_ocean * constants.g * jnp.abs(z_coord.z_full_ref)[None, None, :]
        + jnp.zeros(shape),
    )
    p_cell = compute_hydrostatic_pressure(
        rho, eta, z_coord.dz_ref, jacobian, constants.rho_ocean,
    )

    # f, beta at ~30S band.
    f_coriolis = jnp.full((n_lat, n_lon), -7.3e-5)
    beta = jnp.full((n_lat, n_lon), 2.0e-11)
    return z_coord, jacobian, f_coriolis, beta, T, S, rho, p_cell, eos_fn


def _slopes(rho, z_coord, jacobian):
    """Crude centred interface slopes (|S| ~ 1e-3) for the column."""
    # drho along lat (axis 0) and a vertical gradient -> S ~ -drho_h/drho_z.
    drho_dy = (jnp.roll(rho, -1, axis=0) - jnp.roll(rho, 1, axis=0)) / (2.0 * 5.0e4)
    dz_actual = z_coord.dz_ref * jacobian[..., None]
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    drho_dz = (rho[..., :-1] - rho[..., 1:]) / dz_half
    drho_dy_h = 0.5 * (drho_dy[..., :-1] + drho_dy[..., 1:])
    S_y = -drho_dy_h / jnp.where(jnp.abs(drho_dz) > 1e-12, drho_dz, 1e-12)
    S_y = jnp.clip(S_y, -1e-2, 1e-2)
    S_x = jnp.zeros_like(S_y)
    return S_x, S_y


# ---------------------------------------------------------------------------
# 1. Byte-identity (default off)
# ---------------------------------------------------------------------------

def test_eady_insitu_kwargs_byte_identical():
    z_coord, jac, f, beta, T, S, rho, p_cell, eos_fn = _column()
    S_x, S_y = _slopes(rho, z_coord, jac)
    cfg = VisbeckConfig(enabled=True, use_rossby_radius=True)

    base = _eady_growth_and_length(rho, S_x, S_y, z_coord, jac, f, cfg)
    # explicit insitu + the new (ignored) kwargs supplied
    new = _eady_growth_and_length(
        rho, S_x, S_y, z_coord, jac, f, cfg,
        n2_mode="insitu", T=T, S=S, p_cell=p_cell, eos_fn=eos_fn,
    )
    for a, b in zip(base, new):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_compute_eke_kappa_gm_insitu_byte_identical():
    z_coord, jac, f, beta, T, S, rho, p_cell, eos_fn = _column()
    S_x, S_y = _slopes(rho, z_coord, jac)
    vb = VisbeckConfig(enabled=True, use_rossby_radius=True)
    E = jnp.full(f.shape, 1.0e-3)

    for scheme, kw in [("rossby", {}), ("rhines", {"beta": beta})]:
        eke = EKEConfig(mixing_length_scheme=scheme)
        base = compute_eke_kappa_gm(E, rho, S_x, S_y, z_coord, jac, f, vb, eke, **kw)
        new = compute_eke_kappa_gm(
            E, rho, S_x, S_y, z_coord, jac, f, vb, eke,
            T=T, S=S, p_cell=p_cell, eos_fn=eos_fn, **kw,
        )
        for a, b in zip(base, new):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_compute_visbeck_kappa_gm_insitu_byte_identical():
    z_coord, jac, f, beta, T, S, rho, p_cell, eos_fn = _column()
    S_x, S_y = _slopes(rho, z_coord, jac)
    cfg = VisbeckConfig(enabled=True, use_rossby_radius=True)
    base = compute_visbeck_kappa_gm(rho, S_x, S_y, z_coord, jac, f, cfg)
    new = compute_visbeck_kappa_gm(
        rho, S_x, S_y, z_coord, jac, f, cfg,
        T=T, S=S, p_cell=p_cell, eos_fn=eos_fn,
    )
    np.testing.assert_array_equal(np.asarray(base), np.asarray(new))


# ---------------------------------------------------------------------------
# 2. Compressibility-bias direction (the physics)
# ---------------------------------------------------------------------------

def test_adiabatic_eke_len_shorter_than_insitu():
    """In-situ N² is biased too stable -> ∫N dz too large -> eke_len too long.

    On a deep stratified column with the nonlinear Wright EOS, the adiabatic
    eke_len must be strictly SHORTER than the in-situ one, both finite/positive.
    """
    z_coord, jac, f, beta, T, S, rho, p_cell, eos_fn = _column(H_max=5000.0)
    S_x, S_y = _slopes(rho, z_coord, jac)
    # "rossby" length L = max(N̄·H/|f|, l_min) is purely N-dependent (no Rhines
    # min that would mask the N² difference) — the cleanest probe of the
    # compressibility bias on the mixing length itself.
    vb = VisbeckConfig(enabled=True, use_rossby_radius=True)
    eke = EKEConfig(mixing_length_scheme="rossby")
    E = jnp.full(f.shape, 1.0e-3)

    _, _, L_insitu = compute_eke_kappa_gm(
        E, rho, S_x, S_y, z_coord, jac, f, vb, eke,
    )
    eke_ad = EKEConfig(mixing_length_scheme="rossby", n2_mode="adiabatic")
    _, _, L_adia = compute_eke_kappa_gm(
        E, rho, S_x, S_y, z_coord, jac, f, vb, eke_ad,
        T=T, S=S, p_cell=p_cell, eos_fn=eos_fn,
    )
    L_insitu = np.asarray(L_insitu)
    L_adia = np.asarray(L_adia)
    assert np.all(np.isfinite(L_insitu)) and np.all(L_insitu > 0)
    assert np.all(np.isfinite(L_adia)) and np.all(L_adia > 0)
    # mean over wet interior cells (interior lats avoid the roll wrap)
    m = slice(1, -1)
    assert L_adia[m].mean() < L_insitu[m].mean(), (
        f"adiabatic eke_len {L_adia[m].mean():.1f} should be < in-situ "
        f"{L_insitu[m].mean():.1f}"
    )


def test_adiabatic_intNdz_smaller():
    """The driver of (2): ∫N dz is smaller in adiabatic mode (compressibility)."""
    z_coord, jac, f, beta, T, S, rho, p_cell, eos_fn = _column(H_max=5000.0)
    S_x, S_y = _slopes(rho, z_coord, jac)
    cfg = VisbeckConfig(enabled=True, use_rossby_radius=True)
    _, _, _, int_N_insitu, _, _dzh = _eady_growth_and_length(
        rho, S_x, S_y, z_coord, jac, f, cfg, n2_mode="insitu",
    )
    _, _, _, int_N_adia, _, _dzh = _eady_growth_and_length(
        rho, S_x, S_y, z_coord, jac, f, cfg, n2_mode="adiabatic",
        T=T, S=S, p_cell=p_cell, eos_fn=eos_fn,
    )
    assert float(int_N_adia.mean()) < float(int_N_insitu.mean())
    assert float(int_N_adia.mean()) > 0.0


# ---------------------------------------------------------------------------
# 3. Dispatch discipline
# ---------------------------------------------------------------------------

def test_unknown_n2_mode_raises():
    z_coord, jac, f, beta, T, S, rho, p_cell, eos_fn = _column()
    S_x, S_y = _slopes(rho, z_coord, jac)
    cfg = VisbeckConfig(enabled=True)
    with pytest.raises(ValueError, match="insitu.*adiabatic|adiabatic.*insitu"):
        _eady_growth_and_length(
            rho, S_x, S_y, z_coord, jac, f, cfg, n2_mode="bogus",
        )


def test_adiabatic_without_TS_raises():
    z_coord, jac, f, beta, T, S, rho, p_cell, eos_fn = _column()
    S_x, S_y = _slopes(rho, z_coord, jac)
    cfg = VisbeckConfig(enabled=True)
    with pytest.raises(ValueError, match="adiabatic.*requires"):
        _eady_growth_and_length(
            rho, S_x, S_y, z_coord, jac, f, cfg, n2_mode="adiabatic",
        )
    # via the public EKE entry too
    eke = EKEConfig(mixing_length_scheme="rossby", n2_mode="adiabatic")
    E = jnp.full(f.shape, 1.0e-3)
    with pytest.raises(ValueError, match="adiabatic.*requires"):
        compute_eke_kappa_gm(E, rho, S_x, S_y, z_coord, jac, f, cfg, eke)


# ---------------------------------------------------------------------------
# 4. AD
# ---------------------------------------------------------------------------

def test_grad_finite_through_adiabatic_kappa():
    z_coord, jac, f, beta, T, S, rho, p_cell, eos_fn = _column()
    S_x, S_y = _slopes(rho, z_coord, jac)
    vb = VisbeckConfig(enabled=True, use_rossby_radius=True)
    eke = EKEConfig(mixing_length_scheme="rhines", n2_mode="adiabatic")
    E = jnp.full(f.shape, 1.0e-3)

    def loss(T_in, S_in):
        rho_in = eos_fn(T_in, S_in, p_cell)
        kappa, _, _ = compute_eke_kappa_gm(
            E, rho_in, S_x, S_y, z_coord, jac, f, vb, eke, beta=beta,
            T=T_in, S=S_in, p_cell=p_cell, eos_fn=eos_fn,
        )
        return jnp.sum(kappa)

    gT, gS = jax.grad(loss, argnums=(0, 1))(T, S)
    assert np.all(np.isfinite(np.asarray(gT)))
    assert np.all(np.isfinite(np.asarray(gS)))
    # non-trivial sensitivity somewhere
    assert float(jnp.max(jnp.abs(gT))) > 0.0
