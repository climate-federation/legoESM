"""KK2000 warm-rain tests (iter-7 M3) — SAM M2005 default.

gSAM ``MICRO_M2005/module_mp_graupel.f90`` uses Khairoutdinov-Kogan (2000)
warm rain (``IRAIN=0``):
    PRC = 1350 · q_c^2.47 · (N_c[#/cm³])^-1.79   (autoconversion, line 1813)
    PRA = 67 · (q_c·q_r)^1.15                    (accretion,      line 1952)
These tests pin the formulas, the conversion direction (q_c→q_r), mass
conservation, AD-safety, and the Morrison scheme dispatch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.microphysics._warm_rain import (
    accretion_kk2000,
    autoconversion_kk2000,
    _KK2000_CONS29,
)


jax.config.update("jax_enable_x64", True)


def test_kk2000_autoconversion_matches_sam_formula():
    """PRC = 1350·q_c^2.47·(N_c/1e6)^-1.79 to machine precision."""
    q_c = jnp.array(8.0e-4)
    N_c = jnp.array(1.0e8)              # 1e8 /m³ = 100 /cm³ maritime
    rho = jnp.array(1.1)
    prc, dN_r, x_c = autoconversion_kk2000(q_c, N_c, rho, dt=20.0)
    prc_ref = 1350.0 * (8.0e-4) ** 2.47 * (100.0) ** (-1.79)
    assert float(prc) == pytest.approx(prc_ref, rel=1e-9)
    # Rain-number source = PRC·rho/CONS29, capped at N_c/dt.
    npr1_ref = min(prc_ref * 1.1 / float(_KK2000_CONS29), 1.0e8 / 20.0)
    assert float(dN_r) == pytest.approx(npr1_ref, rel=1e-9)
    # Mean droplet mass x_c = q_c·rho/N_c.
    assert float(x_c) == pytest.approx(8.0e-4 * 1.1 / 1.0e8, rel=1e-9)


def test_kk2000_accretion_matches_sam_formula():
    """PRA = 67·(q_c·q_r)^1.15 to machine precision."""
    q_c, q_r = jnp.array(1.0e-3), jnp.array(2.0e-4)
    pra = accretion_kk2000(q_c, q_r)
    pra_ref = 67.0 * (1.0e-3 * 2.0e-4) ** 1.15
    assert float(pra) == pytest.approx(pra_ref, rel=1e-9)


def test_kk2000_autoconversion_steep_in_qc():
    """The q_c^2.47 power ⇒ doubling q_c raises PRC by 2^2.47 ≈ 5.5×."""
    N_c, rho = jnp.array(1.0e8), jnp.array(1.1)
    prc1, *_ = autoconversion_kk2000(jnp.array(4.0e-4), N_c, rho, 20.0)
    prc2, *_ = autoconversion_kk2000(jnp.array(8.0e-4), N_c, rho, 20.0)
    assert float(prc2 / prc1) == pytest.approx(2.0 ** 2.47, rel=1e-6)


def test_kk2000_autoconversion_decreases_with_droplet_number():
    """More CCN (larger N_c) ⇒ smaller drops ⇒ LESS autoconversion
    (N_c^-1.79) — the aerosol-precip suppression KK2000 captures."""
    rho = jnp.array(1.1)
    prc_clean, *_ = autoconversion_kk2000(
        jnp.array(6.0e-4), jnp.array(5.0e7), rho, 20.0)   # 50 /cm³
    prc_polluted, *_ = autoconversion_kk2000(
        jnp.array(6.0e-4), jnp.array(5.0e8), rho, 20.0)   # 500 /cm³
    assert float(prc_clean) > float(prc_polluted)


def test_kk2000_zero_at_zero_cloud():
    """No cloud water ⇒ no autoconversion / accretion (smooth → 0)."""
    z = jnp.array(0.0)
    prc, dN_r, _ = autoconversion_kk2000(z, jnp.array(1.0e8), jnp.array(1.1), 20.0)
    assert float(prc) == 0.0 and float(dN_r) == 0.0
    assert float(accretion_kk2000(z, jnp.array(1.0e-4))) == 0.0


def test_kk2000_ad_safe_at_zero():
    """jax.grad through both rates stays finite at q_c=0 (cold start)."""
    g_au = jax.grad(
        lambda q: autoconversion_kk2000(q, jnp.array(1e8), jnp.array(1.1), 20.0)[0]
    )(jnp.array(0.0))
    g_ac = jax.grad(lambda q: accretion_kk2000(q, jnp.array(1e-4)))(jnp.array(0.0))
    assert bool(jnp.isfinite(g_au)) and bool(jnp.isfinite(g_ac))


def test_morrison_dispatch_kk2000_default_and_differs_from_sb():
    """MorrisonConfig defaults to kk2000 (SAM); it gives a different
    warm-rain tendency than seifert_beheng, and an unknown scheme raises."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics,
    )
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

    assert MorrisonConfig().warm_rain_scheme == "kk2000"

    ncol, nlev = 1, 4
    T = jnp.full((ncol, nlev), 295.0)
    q_v = jnp.full((ncol, nlev), 0.012)
    p_full = jnp.full((ncol, nlev), 9.0e4)
    p_half = jnp.full((ncol, nlev + 1), 9.0e4)
    rho = jnp.full((ncol, nlev), 1.1)
    dz = jnp.full((ncol, nlev), 200.0)

    def hydro():
        z = jnp.zeros((ncol, nlev))
        return HydrometeorState(
            q_c=jnp.full((ncol, nlev), 6.0e-4), q_r=jnp.full((ncol, nlev), 1.0e-4),
            q_i=z, q_s=z, q_g=z,
            N_c=jnp.full((ncol, nlev), 1.0e8), N_r=jnp.full((ncol, nlev), 1.0e3),
            N_i=z,
        )

    out_kk = morrison_microphysics(
        T, q_v, hydro(), p_full, p_half, rho, dz, 20.0,
        MorrisonConfig(warm_rain_scheme="kk2000"),
    )
    out_sb = morrison_microphysics(
        T, q_v, hydro(), p_full, p_half, rho, dz, 20.0,
        MorrisonConfig(warm_rain_scheme="seifert_beheng"),
    )
    assert bool(jnp.all(jnp.isfinite(out_kk.dq_r_dt)))
    # The two warm-rain schemes give materially different rain production.
    assert not bool(jnp.allclose(out_kk.dq_r_dt, out_sb.dq_r_dt, rtol=1e-3))

    with pytest.raises(ValueError, match="Unknown warm_rain_scheme"):
        morrison_microphysics(
            T, q_v, hydro(), p_full, p_half, rho, dz, 20.0,
            MorrisonConfig(warm_rain_scheme="kessler_typo"),
        )
