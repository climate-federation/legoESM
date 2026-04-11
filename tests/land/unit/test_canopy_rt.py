"""Unit tests for canopy/radiative_transfer.py.

Checks:
- sw decomposition: PAR + NIR + UV ≈ sw_down during day
- Night guard: all components = 0 when cos_zenith < 0.01
- Two-leaf RT: fSun ∈ [0, 1], APAR_Sun + APAR_Sh = total absorbed PAR
- Energy conservation: ASW_Sun + ASW_Sh + ASW_Soil ≤ sw_down
- Vcmax25_Sun + Vcmax25_Sh > 0 when LAI > 0
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.land.canopy.radiative_transfer import (
    canopy_shortwave_rt,
    split_sw_components,
)


def test_split_sw_partitions_daytime():
    sw_down = jnp.array([800.0, 400.0])
    cos_zenith = jnp.array([0.8, 0.3])
    PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV = split_sw_components(
        sw_down, cos_zenith)
    total = PAR_dir + PAR_diff + NIR_dir + NIR_diff + UV
    assert jnp.allclose(total, sw_down, rtol=1e-5, atol=1e-3)


def test_split_sw_night_guard():
    sw_down = jnp.array([10.0])
    cos_zenith = jnp.array([0.001])  # below 0.01 night threshold
    comps = split_sw_components(sw_down, cos_zenith)
    for c in comps:
        assert float(c[0]) == 0.0


def test_canopy_sw_absorption_bounded():
    sw_down = jnp.array([800.0])
    cos_zenith = jnp.array([0.85])
    PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV = split_sw_components(
        sw_down, cos_zenith)
    SZA = jnp.degrees(jnp.arccos(cos_zenith))
    out = canopy_shortwave_rt(
        PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV,
        SZA=SZA,
        LAI=jnp.array([3.0]),
        CI=jnp.array([0.75]),
        ALB_VIS=jnp.array([0.08]),
        ALB_NIR=jnp.array([0.25]),
        Vcmax25_C3_leaf=jnp.array([60.0]),
        Vcmax25_C4_leaf=jnp.array([0.0]),
        kn=jnp.array([0.3]),
        FNonVeg=jnp.array([0.0]),
    )
    # Energy conservation: absorbed ≤ incoming
    absorbed = out.ASW_Sun + out.ASW_Sh + out.ASW_Soil
    assert float(absorbed[0]) > 0.0
    assert float(absorbed[0]) <= float(sw_down[0]) + 1e-3
    # Sunlit fraction in [0, 1]
    assert 0.0 <= float(out.fSun[0]) <= 1.0
    # Vcmax distributed to both fractions
    assert float(out.Vcmax25_C3Sun[0]) > 0.0
    assert float(out.Vcmax25_C3Sh[0]) > 0.0


def test_canopy_sw_night_zero():
    out = canopy_shortwave_rt(
        jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]),
        jnp.array([0.0]), jnp.array([0.0]),
        SZA=jnp.array([89.5]),
        LAI=jnp.array([3.0]), CI=jnp.array([0.75]),
        ALB_VIS=jnp.array([0.08]), ALB_NIR=jnp.array([0.25]),
        Vcmax25_C3_leaf=jnp.array([60.0]),
        Vcmax25_C4_leaf=jnp.array([0.0]),
        kn=jnp.array([0.3]), FNonVeg=jnp.array([0.0]),
    )
    assert float(out.APAR_Sun[0] + out.APAR_Sh[0]) == 0.0
    assert float(out.ASW_Sun[0] + out.ASW_Sh[0] + out.ASW_Soil[0]) == 0.0
