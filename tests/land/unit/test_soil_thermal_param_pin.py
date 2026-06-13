"""Pin test for the soil-thermal coefficient migration (param-hygiene).

Byte-identical guard: the de Vries dry-conductivity and Johansen Kersten
coefficients moved from inline literals in ``compute_thermal_conductivity`` into
``SoilThermalConfig`` fields (+ ``constants.rho_soil_particle``). At default config
the output MUST equal the pre-migration inline formula exactly (rtol=0). A
sensitivity check proves the new fields are live, not decorative.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import SoilThermalConfig, compute_thermal_conductivity


def _oracle(theta, hcfg, rho_b=1400.0):
    """Pre-migration inline formula with the original literal constants."""
    Sr = jnp.clip((theta - hcfg.theta_r) / (hcfg.theta_sat - hcfg.theta_r + 1e-10), 0.01, 1.0)
    k_dry = (0.135 * rho_b + 64.7) / (2700.0 - 0.947 * rho_b)
    k_sat = 2.0 ** (1.0 - hcfg.theta_sat) * 0.57 ** hcfg.theta_sat
    K_e = jnp.log10(jnp.clip(Sr, 0.1, None)) + 1.0  # loam (default texture)
    K_e = jnp.clip(K_e, 0.0, 1.0)
    return k_dry + (k_sat - k_dry) * K_e


def test_conductivity_byte_identical_at_defaults() -> None:
    hcfg = SoilHydraulicsConfig()
    tcfg = SoilThermalConfig()  # default = loam, rho_bulk 1400
    theta = jnp.linspace(0.0, hcfg.theta_sat, 11)
    got = compute_thermal_conductivity(theta, hcfg, tcfg)
    want = _oracle(theta, hcfg, rho_b=tcfg.rho_bulk)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_dry_conductivity_coeffs_are_live() -> None:
    hcfg = SoilHydraulicsConfig()
    base = SoilThermalConfig()
    bumped = base._replace(k_dry_coeff_a=base.k_dry_coeff_a * 1.5)
    theta = jnp.full((5,), 0.15)
    k_base = compute_thermal_conductivity(theta, hcfg, base)
    k_bumped = compute_thermal_conductivity(theta, hcfg, bumped)
    assert not np.allclose(np.asarray(k_base), np.asarray(k_bumped))


def test_kersten_slope_live_for_sand() -> None:
    hcfg = SoilHydraulicsConfig()
    base = SoilThermalConfig(soil_texture="sand")
    bumped = base._replace(kersten_slope_coarse=0.4)
    theta = jnp.full((5,), 0.20)
    assert not np.allclose(
        np.asarray(compute_thermal_conductivity(theta, hcfg, base)),
        np.asarray(compute_thermal_conductivity(theta, hcfg, bumped)),
    )


def test_conductivity_differentiable_wrt_new_field() -> None:
    hcfg = SoilHydraulicsConfig()
    theta = jnp.full((4,), 0.18)

    def loss(a: float) -> jax.Array:
        cfg = SoilThermalConfig(k_dry_coeff_a=a)
        return jnp.sum(compute_thermal_conductivity(theta, hcfg, cfg))

    g = jax.grad(loss)(0.135)
    assert np.isfinite(float(g)) and abs(float(g)) > 0.0
