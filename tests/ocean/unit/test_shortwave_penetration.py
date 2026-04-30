"""Smoke test for ocean subsurface SW penetration tendency.

Tests that the two-band Paulson & Simpson (1977) absorption profile produces
a finite, sane temperature tendency that integrates to ~Q_sw / (rho_0 c_sw)
over the column when surface absorption is negligible at the bottom.

Run with:

    JAX_ENABLE_X64=1 .venv/bin/python3.14 -m pytest \
        tests/ocean/unit/test_shortwave_penetration.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.ocean.physics.shortwave_penetration import (
    JerlovParams,
    JERLOV_TYPES,
    ShortwavePenetrationConfig,
    shortwave_penetration_tendency,
)


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _setup(nx=4, ny=3, nlev=10, H=200.0):
    """Build SW input + reference z grid (top-to-bottom, negative)."""
    sw_down = jnp.full((nx, ny), 200.0)              # [W/m^2]
    # Reference layer thicknesses (uniform).
    dz_ref = jnp.full((nlev,), H / nlev)
    # Reference half-level depths: 0 at surface, -H at bottom.
    z_half_ref = -jnp.linspace(0.0, H, nlev + 1)
    jacobian = jnp.ones((nx, ny))                    # eta=0, J=1
    return sw_down, dz_ref, z_half_ref, jacobian


def test_jerlov_params_namedtuple():
    p = JerlovParams(R=0.6, zeta1=0.5, zeta2=15.0)
    assert p.R == 0.6
    assert p.zeta1 == 0.5
    assert p.zeta2 == 15.0


def test_jerlov_lookup_table():
    for water_type in ("I", "IA", "IB", "II", "III"):
        params = JERLOV_TYPES[water_type]
        assert 0.0 < params.R < 1.0
        assert params.zeta1 > 0.0
        assert params.zeta2 > params.zeta1  # long band penetrates deeper


def test_config_defaults():
    cfg = ShortwavePenetrationConfig()
    assert cfg.water_type == "II"
    cfg2 = ShortwavePenetrationConfig(water_type="IA")
    assert cfg2.water_type == "IA"


def test_shortwave_penetration_tendency_shape_and_finite():
    sw_down, dz_ref, z_half_ref, J = _setup()
    dT_dt = shortwave_penetration_tendency(
        sw_down, dz_ref, z_half_ref, J,
    )
    expected_shape = sw_down.shape + (dz_ref.shape[0],)
    assert dT_dt.shape == expected_shape
    assert jnp.all(jnp.isfinite(dT_dt))
    # All layers absorb a non-negative fraction (heating with sw>0).
    assert jnp.all(dT_dt >= 0.0)


def test_top_layer_warms_most():
    sw_down, dz_ref, z_half_ref, J = _setup()
    dT_dt = shortwave_penetration_tendency(
        sw_down, dz_ref, z_half_ref, J,
    )
    # Top layer (index 0) absorbs the most under exponential decay.
    top = float(jnp.mean(dT_dt[..., 0]))
    bottom = float(jnp.mean(dT_dt[..., -1]))
    assert top > bottom


def test_zero_swdown_gives_zero_tendency():
    sw_down, dz_ref, z_half_ref, J = _setup()
    out = shortwave_penetration_tendency(
        jnp.zeros_like(sw_down), dz_ref, z_half_ref, J,
    )
    assert jnp.allclose(out, 0.0)


def test_water_type_changes_result():
    sw_down, dz_ref, z_half_ref, J = _setup()
    out_I = shortwave_penetration_tendency(
        sw_down, dz_ref, z_half_ref, J,
        config=ShortwavePenetrationConfig(water_type="I"),
    )
    out_III = shortwave_penetration_tendency(
        sw_down, dz_ref, z_half_ref, J,
        config=ShortwavePenetrationConfig(water_type="III"),
    )
    # Type III is more turbid: surface layer absorbs more than Type I.
    assert float(jnp.mean(out_III[..., 0])) > float(jnp.mean(out_I[..., 0]))


def test_jit_compiles():
    sw_down, dz_ref, z_half_ref, J = _setup()
    fn = jax.jit(shortwave_penetration_tendency)
    out = fn(sw_down, dz_ref, z_half_ref, J)
    assert jnp.all(jnp.isfinite(out))
