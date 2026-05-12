"""FV3_3D iter 722: compute_pkz_fv3 port.

Faithful JAX port of FV3 pkz (Exner factor per layer) computation
(model/fv_mapz.F90:457 hydrostatic, :481 non-hydrostatic dry).

Tests
-----

1. ``test_pkz_isothermal_isobaric_matches_p_kappa``.
2. ``test_pkz_hydrostatic_isothermal_column``.
3. ``test_pkz_nonhydrostatic_dry``.
4. ``test_pkz_hydro_vs_nonhydro_isothermal_match``.
5. ``test_pkz_custom_cappa``.
6. ``test_pkz_shapes_3d``.
7. ``test_pkz_missing_args_raises``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import compute_pkz_fv3


def test_pkz_isothermal_isobaric_matches_p_kappa():
    """Single layer with thin Δp around p=1e5 → pkz ≈ p^κ.

    For Δp small: pkz = (p_bot^κ − p_top^κ) / (κ·ln(p_bot/p_top))
              → d(p^κ)/d(ln p) = κ·p^κ / κ = p^κ in the limit Δp → 0.
    """
    pe = jnp.array([0.99e5, 1.01e5])
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    pkz = compute_pkz_fv3(delp, peln=peln, hydrostatic=True)
    expected = 1.0e5 ** constants.kappa
    assert jnp.allclose(pkz, expected, rtol=1e-4)


def test_pkz_hydrostatic_isothermal_column():
    """Multi-layer isothermal column → pkz monotone with pressure."""
    km = 10
    pe = jnp.linspace(1.0e4, 1.0e5, km + 1)
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    pkz = compute_pkz_fv3(delp, peln=peln, hydrostatic=True)
    # pkz should increase from low p (top) to high p (bottom)
    assert jnp.all(jnp.diff(pkz) > 0.0)
    assert jnp.all(pkz > 0.0)


def test_pkz_nonhydrostatic_dry():
    """Non-hydrostatic dry: pkz = (R_d·delp·pt/(g·|delz|))^κ.
    Tested against direct computation."""
    km = 5
    pt = jnp.full((km,), 280.0)
    delp = jnp.full((km,), 5000.0)
    delz = jnp.full((km,), -500.0)
    pkz = compute_pkz_fv3(
        delp, pt=pt, delz=delz, hydrostatic=False,
    )
    expected = (
        constants.R_d * delp[0] * pt[0] / (constants.g * 500.0)
    ) ** constants.kappa
    assert jnp.allclose(pkz, expected, rtol=1e-10)


def test_pkz_hydro_vs_nonhydro_isothermal_match():
    """Isothermal hydrostatic column: hydrostatic = R_d·T·ln(p_bot/p_top)·g
    relates delz, so hydro and non-hydro pkz should match.

    For isothermal T=280, p_bot=1e5, p_top=9e4:
        H = R_d·T/g, delz = -H·ln(p_bot/p_top), |delz| = H·ln(p_bot/p_top)
        Non-hydro pkz = (R_d·(p_bot-p_top)·T/(g·|delz|))^κ
                     = ((p_bot-p_top)/ln(p_bot/p_top))^κ
        Hydrostatic pkz = (p_bot^κ - p_top^κ)/(κ·ln(p_bot/p_top))
    These match analytically.
    """
    T = 280.0
    p_top, p_bot = 9.0e4, 1.0e5
    H = constants.R_d * T / constants.g
    delz_val = -H * jnp.log(p_bot / p_top)
    delp = jnp.array([p_bot - p_top])
    pt = jnp.array([T])
    delz = jnp.array([delz_val])
    peln = jnp.log(jnp.array([p_top, p_bot]))
    pkz_hydro = compute_pkz_fv3(delp, peln=peln, hydrostatic=True)
    pkz_nonh = compute_pkz_fv3(delp, pt=pt, delz=delz, hydrostatic=False)
    # Two formulas differ at higher order in delp/p (each uses different
    # discretization).  Should agree to ~1e-4 relative for delp/p ~ 0.1.
    assert jnp.allclose(pkz_hydro, pkz_nonh, rtol=1e-3)


def test_pkz_custom_cappa():
    """Custom cappa override (FV3 moist nwat path)."""
    km = 5
    pt = jnp.full((km,), 280.0)
    delp = jnp.full((km,), 5000.0)
    delz = jnp.full((km,), -500.0)
    cappa_array = jnp.full((km,), 0.30)  # Slightly moist cappa
    pkz = compute_pkz_fv3(
        delp, pt=pt, delz=delz, hydrostatic=False, cappa=cappa_array,
    )
    expected = (
        constants.R_d * delp[0] * pt[0] / (constants.g * 500.0)
    ) ** 0.30
    assert jnp.allclose(pkz, expected, rtol=1e-10)


def test_pkz_shapes_3d():
    """3-D inputs → 3-D output."""
    rng = np.random.default_rng(seed=722)
    n_x, n_y, km = 4, 5, 20
    pe_col = jnp.linspace(1.0e4, 1.0e5, km + 1)
    peln = jnp.broadcast_to(jnp.log(pe_col)[None, None, :], (n_x, n_y, km + 1))
    delp = jnp.broadcast_to((pe_col[1:] - pe_col[:-1])[None, None, :],
                            (n_x, n_y, km))
    pkz = compute_pkz_fv3(delp, peln=peln, hydrostatic=True)
    assert pkz.shape == (n_x, n_y, km)


def test_pkz_missing_args_raises():
    """Missing peln (hydrostatic) / pt / delz raises."""
    km = 5
    delp = jnp.full((km,), 1000.0)
    with pytest.raises(ValueError):
        compute_pkz_fv3(delp, hydrostatic=True)
    with pytest.raises(ValueError):
        compute_pkz_fv3(delp, hydrostatic=False)
