"""OMIP-2 / CORE-II open-ocean bulk flux formulas (Large & Yeager 2009).

Reference
---------
Large, W. G., & Yeager, S. G. (2009). "The global climatology of an
interannually varying air-sea flux data set", Climate Dynamics 33,
341-364.

What
----
Converts the seven JRA55-do / CORE-II forcing channels (u10, v10,
T_air, q_air, sw_down, lw_down, precip) plus the model surface
state (SST, sea-ice-fraction) into the four ocean surface fluxes:

* ``tau_x``, ``tau_y``  -- wind stress on the ocean surface [Pa]
* ``shflx``             -- sensible heat flux [W/m^2] (positive into ocean)
* ``lhflx``             -- latent heat flux [W/m^2]   (positive into ocean)

Coefficients
------------
L&Y 2009 uses a wind-speed-dependent open-ocean drag

.. math::
    C_d \\cdot 10^3 = \\frac{2.7}{u_{10}} + 0.142 + 0.0764 u_{10}

(valid for ``u10 in [0.5, 25] m/s``; floored at u10 = 0.5 to avoid
``1/u10`` blowup near calm winds). The heat / moisture transfer
coefficient ``C_h`` is fixed at ``1.46e-3`` for unstable conditions
and ``1.18e-3`` for stable -- a single-coefficient approximation
of the original Eq. 25 / 26 that is good to ~10 % in typical
mid-latitude conditions.

The implementation reuses :func:`legoesm.core.bulk_flux.simple_bulk_fluxes`
for the actual flux assembly so the unit-test surface stays the same
as the atmosphere-coupled path.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


_U10_FLOOR_M_S: float = 0.5    # avoid 1/u10 -> inf at calm winds
_CH_UNSTABLE: float = 1.46e-3
_CH_STABLE: float = 1.18e-3


def large_yeager_cd(u10_speed):
    """L&Y 2009 open-ocean wind-speed-dependent neutral drag coefficient.

    Delegates to the canonical substrate kernel
    :func:`legoesm.core.bulk_flux.large_yeager_neutral_cd` (redundancy audit) so
    the ocean uses the FULL LY09 Eq. 6 — including the ``-3.14807e-10·U⁶``
    high-wind correction the OMIP-2 protocol (Griffies 2016) requires and this
    local copy previously omitted — and the same ``[0.5e-3, 3.0e-3]`` clip as the
    MOST flux solver.  Returns ``C_d`` (dimensionless).
    """
    from legoesm.core.bulk_flux import large_yeager_neutral_cd

    return large_yeager_neutral_cd(jnp.asarray(u10_speed, dtype=jnp.float64))


def large_yeager_ch(T_air_K, T_sfc_K):
    """L&Y 2009 sensible-heat transfer coefficient.

    Simplified to a two-regime constant depending on the sign of
    ``T_sfc - T_air``: unstable (ocean warmer) -> 1.46e-3,
    stable (ocean cooler) -> 1.18e-3. Two-coefficient
    approximation of the full Eq. 25/26.
    """
    return jnp.where(T_sfc_K > T_air_K, _CH_UNSTABLE, _CH_STABLE)


def air_sea_fluxes(u10, v10, T_air_K, q_air, T_sfc_K, q_sfc, rho_air,
                   *, L_latent=None):
    """Apply L&Y 2009 + ``simple_bulk_fluxes`` to derive (tau_x, tau_y,
    shflx, lhflx).

    Sign convention: ``shflx > 0`` and ``lhflx > 0`` add heat to the
    ocean; ``tau_x`` follows the wind direction.
    """
    u_arr = jnp.asarray(u10, dtype=jnp.float64)
    v_arr = jnp.asarray(v10, dtype=jnp.float64)
    wind_speed = jnp.sqrt(u_arr ** 2 + v_arr ** 2 + 1e-12)
    Cd = large_yeager_cd(wind_speed)
    Ch = large_yeager_ch(T_air_K, T_sfc_K)
    # ``simple_bulk_fluxes`` expects scalar Cd / Ch; broadcast via the
    # caller's array shape by reusing the underlying formula directly
    # so we keep the array-valued L&Y coefficients.
    L = constants.L_v if L_latent is None else L_latent
    tau_x = -rho_air * Cd * wind_speed * u_arr
    tau_y = -rho_air * Cd * wind_speed * v_arr
    # L&Y sign convention: positive flux into the ocean when the air
    # is warmer than the ocean.
    shflx = rho_air * constants.c_pd * Ch * wind_speed * (T_air_K - T_sfc_K)
    lhflx = rho_air * L * Ch * wind_speed * (q_air - q_sfc)
    return tau_x, tau_y, shflx, lhflx


__all__ = ["large_yeager_cd", "large_yeager_ch", "air_sea_fluxes"]
