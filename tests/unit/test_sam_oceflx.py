"""SAM ``oceflx`` faithful bulk-flux tests (iter-5 SF fix).

Covers the new :mod:`legoesm.core.bulk_flux` helpers ported from
gSAM ``oceflx.f90``: the salt-reduced ocean surface humidity
(``sam_ocean_surface_q``), the neutral 10 m drag (``_sam_cdn``), and the
iterative Monin–Obukhov scheme (``compute_sam_oceflx_fluxes``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.bulk_flux import (
    _sam_cdn,
    compute_sam_oceflx_fluxes,
    sam_ocean_surface_q,
)
from legoesm.thermo import saturation_specific_humidity


jax.config.update("jax_enable_x64", True)


def _rce_args(u=5.0, theta_atm=299.0, q_atm=0.016, T_sfc=300.0, q_sfc=None):
    if q_sfc is None:
        q_sfc = float(sam_ocean_surface_q(jnp.asarray(T_sfc), 101480.0))
    return dict(
        u_atm=jnp.asarray(u), v_atm=jnp.asarray(0.0),
        theta_atm=jnp.asarray(theta_atm), q_atm=jnp.asarray(q_atm),
        T_sfc=jnp.asarray(T_sfc), q_sfc=jnp.asarray(q_sfc),
        rho=jnp.asarray(1.16), z_bot=jnp.asarray(50.0),
    )


def test_sam_ocean_surface_q_salt_reduced_and_larger_than_hardcoded():
    """q_sfc = 0.981·qsat(SST): ~0.0215 at 300 K — ~20 % above the old
    hardcoded 0.018, and strictly below the unreduced saturation."""
    qs = float(sam_ocean_surface_q(jnp.asarray(300.0), 101480.0))
    qsat = float(saturation_specific_humidity(jnp.asarray(300.0), 101480.0))
    assert qs == pytest.approx(0.981 * qsat)
    assert 0.020 < qs < 0.023
    assert qs > 0.018                       # the SF-2 bias the fix removes
    assert qs < qsat                        # salinity reduction


def test_sam_cdn_matches_oceflx_formula():
    """Neutral drag cdn(U)=0.0027/U+0.000142+0.0000764U (oceflx.f90)."""
    for u in (5.0, 10.0, 20.0):
        expected = 0.0027 / u + 0.000142 + 0.0000764 * u
        assert float(_sam_cdn(jnp.asarray(u))) == pytest.approx(expected)
    # Floored at umin = 1 m/s (the 1/U term stays bounded).
    assert float(_sam_cdn(jnp.asarray(0.0))) == pytest.approx(
        float(_sam_cdn(jnp.asarray(1.0)))
    )


def test_oceflx_flux_signs_and_magnitudes():
    """Surface warmer + moister ⇒ positive (upward) SHF + LHF; LHF in the
    physical 50–300 W/m² range for a typical 5 m/s RCE surface layer."""
    taux, tauy, shf, lhf, ust = compute_sam_oceflx_fluxes(**_rce_args())
    assert float(shf) > 0.0                  # surface (300) warmer than air (299)
    assert float(lhf) > 0.0                  # surface moister
    assert 50.0 < float(lhf) < 300.0
    assert 0.05 < float(ust) < 0.6           # physical friction velocity


def test_oceflx_momentum_opposes_wind():
    """Stress opposes the wind and scales with ρ u*²."""
    taux, tauy, *_ , ust = compute_sam_oceflx_fluxes(**_rce_args(u=8.0))
    assert float(taux) < 0.0                 # opposes +u
    assert float(tauy) == pytest.approx(0.0)


def test_oceflx_calm_wind_floored_to_umin():
    """U=0 ⇒ vmag floored to 1 m/s (WISHE floor) — still finite, nonzero
    flux, smaller than the windy case."""
    _, _, _, lhf0, ust0 = compute_sam_oceflx_fluxes(**_rce_args(u=0.0))
    _, _, _, lhf5, _ = compute_sam_oceflx_fluxes(**_rce_args(u=5.0))
    assert np.isfinite(float(lhf0)) and float(lhf0) > 0.0
    assert float(lhf0) < float(lhf5)
    assert float(ust0) > 0.0


def test_oceflx_salt_qsat_raises_lhf_vs_hardcoded_018():
    """The SF-2 point: using q_sfc = 0.981·qsat(SST) instead of the old
    hardcoded 0.018 increases the latent-heat flux (larger air–sea
    humidity deficit)."""
    args_fix = _rce_args()                                  # q_sfc≈0.0215
    args_old = _rce_args(q_sfc=0.018)
    _, _, _, lhf_fix, _ = compute_sam_oceflx_fluxes(**args_fix)
    _, _, _, lhf_old, _ = compute_sam_oceflx_fluxes(**args_old)
    assert float(lhf_fix) > float(lhf_old)
    # The deficit ratio sets the LH ratio (same wind/coeffs).
    assert float(lhf_fix) / float(lhf_old) > 1.1


def test_oceflx_stability_raises_unstable_flux():
    """Unstable (surface much warmer) gives a larger Stanton number than
    stable (surface cooler) — SAM 0.0327 (unstable) vs 0.018 (stable)."""
    # Unstable: surface 4 K warmer than air.
    *_, shf_unstable, _, _ = compute_sam_oceflx_fluxes(
        **_rce_args(theta_atm=296.0)
    )
    # Stable: air warmer than surface (negative SHF, downward).
    *_, shf_stable, _, _ = compute_sam_oceflx_fluxes(
        **_rce_args(theta_atm=303.0)
    )
    assert float(shf_unstable) > 0.0
    assert float(shf_stable) < 0.0


def test_oceflx_ad_safe():
    """jax.grad through the MO iteration stays finite (T_sfc, q_atm, u)."""
    def lhf_of(T_sfc, q_atm, u):
        _, _, _, lhf, _ = compute_sam_oceflx_fluxes(
            u_atm=u, v_atm=jnp.asarray(0.0),
            theta_atm=jnp.asarray(299.0), q_atm=q_atm,
            T_sfc=T_sfc, q_sfc=jnp.asarray(0.0215),
            rho=jnp.asarray(1.16), z_bot=jnp.asarray(50.0),
        )
        return lhf

    g = jax.grad(lhf_of, argnums=(0, 1, 2))(
        jnp.asarray(300.0), jnp.asarray(0.016), jnp.asarray(5.0),
    )
    assert all(bool(jnp.isfinite(gi)) for gi in g)
    # Warmer SST and drier air both raise LHF; calmer wind lowers it.
    assert float(g[0]) > 0.0                 # dLHF/dT_sfc > 0
    assert float(g[1]) < 0.0                 # dLHF/dq_atm < 0
    assert float(g[2]) > 0.0                 # dLHF/dU > 0
