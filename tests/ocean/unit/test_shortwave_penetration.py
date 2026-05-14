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


def test_dry_column_gives_zero_finite_tendency():
    """Dry / land cells (jacobian = 0 → dz_actual = 0) must yield zero
    tendency, not NaN/Inf.

    Before the iter-56 fix, the division ``sw / (rho · c · dz)`` produced
    Inf on land cells, contaminating the summed ocean tendency.
    ``NaN * 0 = NaN`` in IEEE so a downstream output mask cannot scrub
    the contamination; the leaf must produce zero on dry columns
    directly.
    """
    sw_down, dz_ref, z_half_ref, J = _setup()
    # Mark half the grid as dry (J = 0); other half wet (J = 1).
    J_mixed = J.at[: J.shape[0] // 2, :].set(0.0)
    out = shortwave_penetration_tendency(
        sw_down, dz_ref, z_half_ref, J_mixed,
    )
    assert jnp.all(jnp.isfinite(out))
    # Dry rows: zero everywhere.
    assert jnp.allclose(out[: J.shape[0] // 2, :, :], 0.0)
    # Wet rows: non-zero (positive heating).
    assert float(jnp.max(out[J.shape[0] // 2 :, :, :])) > 0.0


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


def test_column_sw_conservation_deep_water():
    """SW penetration must conserve the column heat budget:
    sum over layers of dT/dt · rho_0 · c_sw · dz_layer = sw_down.

    For deep water (H >> zeta2), the original formulation already
    closed to ~6 sig figs.  The iter-1 fix added the bottom-layer
    leakage absorption; the column closure must hold to bit-precision.
    """
    from legoesm.ocean.eos import rho_0, c_sw

    sw_down = jnp.full((4, 3), 200.0)
    nlev = 30
    H = 500.0  # deep ocean
    dz_ref = jnp.full((nlev,), H / nlev)
    z_half_ref = -jnp.linspace(0.0, H, nlev + 1)
    jacobian = jnp.ones((4, 3))

    dT_dt = shortwave_penetration_tendency(
        sw_down, dz_ref, z_half_ref, jacobian,
    )
    # Column heating in W/m²: sum(dT_dt * rho_0 * c_sw * dz_actual)
    dz_actual = dz_ref * jacobian[..., None]
    col_heating = jnp.sum(dT_dt * rho_0 * c_sw * dz_actual, axis=-1)
    # Should equal sw_down to high precision.
    assert jnp.allclose(col_heating, sw_down, rtol=1e-12)


def test_column_sw_conservation_shallow_water():
    """The iter-1 bottom-layer leakage fix matters most in shallow
    water (H comparable to zeta2 = 23 m).  Without the fix, ~6 % of
    SW would escape from the column; with the fix the column heating
    must still equal sw_down exactly.
    """
    from legoesm.ocean.eos import rho_0, c_sw

    sw_down = jnp.full((2, 2), 200.0)
    nlev = 10
    H = 50.0  # shallow shelf — zeta2 = 23 m → I_half[-1] ≈ 0.06
    dz_ref = jnp.full((nlev,), H / nlev)
    z_half_ref = -jnp.linspace(0.0, H, nlev + 1)
    jacobian = jnp.ones((2, 2))

    dT_dt = shortwave_penetration_tendency(
        sw_down, dz_ref, z_half_ref, jacobian,
    )
    dz_actual = dz_ref * jacobian[..., None]
    col_heating = jnp.sum(dT_dt * rho_0 * c_sw * dz_actual, axis=-1)
    # Bottom-layer absorption fix must keep column closure exact.
    assert jnp.allclose(col_heating, sw_down, rtol=1e-12)
