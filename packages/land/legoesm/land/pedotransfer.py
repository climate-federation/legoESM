"""Pedotransfer: soil texture (sand/clay) -> Clapp-Hornberger hydraulic params.

The surfdata gives soil **texture** (percent sand/clay per layer); the soil
hydrology needs retention-curve parameters.  This module implements the Cosby et
al. (1984) regressions as used by CLM (CLM4 Tech Note eqs. 7.82, 7.84, 7.87,
7.90), producing Clapp & Hornberger (1978) parameters:

    theta_sat = 0.489 - 0.00126 * %sand                       [m3/m3]   (7.82)
    b         = 2.91  + 0.159   * %clay                        [-]       (7.84)
    psi_sat   = -10.0 * 10^(1.88 - 0.0131 * %sand)            [mm] ->[m] (7.87)
    K_sat     = 0.0070556 * 10^(-0.884 + 0.0153 * %sand)      [mm/s]->[m/s] (7.90)

Pure JAX and differentiable in (sand, clay) — the regression coefficients live in
:class:`CosbyPedotransferConfig` so they can later be *learned* rather than fixed.

References
----------
- Cosby, B. J., Hornberger, G. M., Clapp, R. B., and Ginn, T. R. (1984): A
  statistical exploration of the relationships of soil moisture characteristics
  to the physical properties of soils. Water Resour. Res., 20(6), 682-690.
  https://doi.org/10.1029/WR020i006p00682
- Clapp, R. B. and Hornberger, G. M. (1978): Empirical equations for some soil
  hydraulic properties. Water Resour. Res., 14(4), 601-604.
- Oleson et al. (2010): CLM4 Technical Note, section 7.4.

Faithfulness
------------
``tests/land/unit/test_pedotransfer_faithful.py`` pins all four Cosby/CLM4 forms
to round-off (rel 1e-9) against an independent scalar reimplementation and
canaries the regression coefficients against ``CosbyPedotransferConfig``.  The
oracle literals are transcribed from the on-disk gSAM SLM reference
``SLM/slm_vars.f90:1227-1236`` (the identical Cosby fit: ``poro_soil = -0.00126
SAND + 0.489``, ``Bconst = 0.159 CLAY + 2.91``, ``m_pot_sat = min(-150, -10*10^(
1.88 - 0.0131 SAND))`` mm, ``ks = 10^(0.0153 SAND - 0.884) * 25.4/3600`` mm/s).

Two documented DEPARTURES from the gSAM reference (both leave legoesm closer to
the raw published Cosby/CLM4 eqs):
  * psi_sat is UNCAPPED here (raw eq. 7.87), whereas gSAM floors its MAGNITUDE at
    150 mm (``min(-150, raw)`` — this binds for HIGH sand / small |psi_sat|, e.g.
    at 90% sand legoesm gives -50 mm but gSAM -150 mm);
  * the K_sat prefactor is CLM4's rounded ``0.0070556`` mm/s vs gSAM's exact
    ``25.4/3600`` (inch/hr -> mm/s), a ~6.3e-6 relative difference.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.land.soil_hydraulics import SoilHydraulicsConfig

_MM_TO_M = 1.0e-3


class CosbyPedotransferConfig(NamedTuple):
    """Cosby et al. (1984) regression coefficients (CLM4 defaults)."""

    theta_sat_intercept: float = 0.489
    theta_sat_sand: float = 0.00126
    b_intercept: float = 2.91
    b_clay: float = 0.159
    psi_sat_intercept: float = 1.88     # exponent base-10
    psi_sat_sand: float = 0.0131
    psi_sat_coeff_mm: float = 10.0      # |psi_sat| prefactor [mm]
    ksat_intercept: float = -0.884      # exponent base-10
    ksat_sand: float = 0.0153
    ksat_coeff_mm_s: float = 0.0070556  # K_sat prefactor [mm/s]


# Cosby et al. (1984) pedotransfer regression coefficients: a fixed published
# fit, not a trained model closure (treated as constants).
__param_spec__ = {
    "CosbyPedotransferConfig": {
        "scheme_key": "land.pedotransfer",
        "excluded": {
            "theta_sat_intercept": "Cosby et al. 1984 pedotransfer fit (fixed)",
            "theta_sat_sand": "Cosby et al. 1984 pedotransfer fit (fixed)",
            "b_intercept": "Cosby et al. 1984 pedotransfer fit (fixed)",
            "b_clay": "Cosby et al. 1984 pedotransfer fit (fixed)",
            "psi_sat_intercept": "Cosby et al. 1984 pedotransfer fit (fixed)",
            "psi_sat_sand": "Cosby et al. 1984 pedotransfer fit (fixed)",
            "psi_sat_coeff_mm": "conversion: |psi_sat| prefactor [mm]",
            "ksat_intercept": "Cosby et al. 1984 pedotransfer fit (fixed)",
            "ksat_sand": "Cosby et al. 1984 pedotransfer fit (fixed)",
            "ksat_coeff_mm_s": "conversion: K_sat prefactor [mm/s]",
        },
        "params": {},
    },
}


class CosbyParams(NamedTuple):
    """Clapp-Hornberger parameters from the Cosby pedotransfer."""

    theta_sat: jnp.ndarray   # [m3/m3]
    psi_sat: jnp.ndarray     # [m] (negative)
    b_ch: jnp.ndarray        # [-]
    K_sat: jnp.ndarray       # [m/s]


def cosby_hydraulic_params(
    sand_pct: jnp.ndarray,
    clay_pct: jnp.ndarray,
    config: CosbyPedotransferConfig = CosbyPedotransferConfig(),
) -> CosbyParams:
    """Clapp-Hornberger params from percent sand/clay (element-wise, JAX)."""
    sand = jnp.asarray(sand_pct)
    clay = jnp.asarray(clay_pct)
    theta_sat = config.theta_sat_intercept - config.theta_sat_sand * sand
    b_ch = config.b_intercept + config.b_clay * clay
    psi_sat = -(config.psi_sat_coeff_mm
                * 10.0 ** (config.psi_sat_intercept - config.psi_sat_sand * sand)) * _MM_TO_M
    K_sat = (config.ksat_coeff_mm_s
             * 10.0 ** (config.ksat_intercept + config.ksat_sand * sand)) * _MM_TO_M
    return CosbyParams(theta_sat=theta_sat, psi_sat=psi_sat, b_ch=b_ch, K_sat=K_sat)


def soil_hydraulics_config_from_texture(
    sand_pct: float,
    clay_pct: float,
    *,
    base: SoilHydraulicsConfig = SoilHydraulicsConfig(),
    config: CosbyPedotransferConfig = CosbyPedotransferConfig(),
) -> SoilHydraulicsConfig:
    """Build a Clapp-Hornberger :class:`SoilHydraulicsConfig` from scalar texture.

    For a single representative texture (e.g. a column-mean sand/clay); the
    retention curve is switched to ``"clapp_hornberger"`` and theta_r set to 0
    (Clapp-Hornberger has no residual term).
    """
    p = cosby_hydraulic_params(jnp.asarray(float(sand_pct)), jnp.asarray(float(clay_pct)), config)
    return base._replace(
        retention_curve="clapp_hornberger",
        theta_sat=float(p.theta_sat),
        psi_sat=float(p.psi_sat),
        b_ch=float(p.b_ch),
        K_sat=float(p.K_sat),
        theta_r=0.0,
    )
