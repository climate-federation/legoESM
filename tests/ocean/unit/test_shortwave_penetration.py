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


# ---------------------------------------------------------------------
# #1226 traqsr.F90:665-712 qsr_2BD live gdepw ladder (z_half_stretch=)
# ---------------------------------------------------------------------

def test_z_half_stretch_none_is_bit_identical_to_no_kwarg():
    """z_half_stretch=None (the default) must be byte-identical to the
    pre-existing call signature -- non-bridged callers are unaffected."""
    sw_down, dz_ref, z_half_ref, J = _setup()
    out_no_kwarg = shortwave_penetration_tendency(sw_down, dz_ref, z_half_ref, J)
    out_none = shortwave_penetration_tendency(
        sw_down, dz_ref, z_half_ref, J, z_half_stretch=None)
    assert jnp.max(jnp.abs(out_no_kwarg - out_none)) == 0.0


def test_z_half_stretch_one_matches_static():
    """A stretch of exactly 1.0 everywhere (r3t=0, the DINOConfig() default
    'static' case) must reproduce the static-ladder tendency bit-for-bit --
    proves the ONLY difference the new kwarg introduces is the stretch
    itself, matching the #1226 tra_qsr_tem_piece_decompose.py Part 2b
    self-check pattern."""
    sw_down, dz_ref, z_half_ref, J = _setup(nx=4, ny=3, nlev=10, H=200.0)
    stretch_one = jnp.ones((4, 3))
    out_static = shortwave_penetration_tendency(sw_down, dz_ref, z_half_ref, J)
    out_stretch_one = shortwave_penetration_tendency(
        sw_down, dz_ref, z_half_ref, J, z_half_stretch=stretch_one)
    assert jnp.max(jnp.abs(out_static - out_stretch_one)) == 0.0


def test_z_half_stretch_matches_independent_transcription():
    """A per-column stretch != 1 must reproduce an INDEPENDENT transcription
    of NEMO's qsr_2BD (traqsr.F90:665-712): the two-band formula evaluated
    at the stretched interface depths AND the stretched layer thickness
    (domzgr_substitute.h90:139's single e3t_0*(1+r3t) factor), built here
    from raw numpy/jnp, never calling the function under test for its own
    formula."""
    import numpy as np

    nx, ny, nlev, H = 3, 2, 6, 120.0
    sw_down = jnp.full((nx, ny), 180.0)
    dz_ref = jnp.full((nlev,), H / nlev)
    z_half_ref = -jnp.linspace(0.0, H, nlev + 1)
    J = jnp.ones((nx, ny))
    stretch = jnp.array([[1.02, 0.97], [1.10, 0.90], [1.00, 1.05]])  # (nx, ny)

    params = JERLOV_TYPES["II"]
    R, zeta1, zeta2 = params.R, params.zeta1, params.zeta2
    from legoesm.ocean.eos import rho_0, c_sw

    z_half_live = np.asarray(z_half_ref)[None, None, :] * np.asarray(stretch)[:, :, None]
    I_half = R * np.exp(z_half_live / zeta1) + (1.0 - R) * np.exp(z_half_live / zeta2)
    frac = I_half[:, :, :-1] - I_half[:, :, 1:]
    frac[:, :, -1] += I_half[:, :, -1]
    dz_live = np.asarray(dz_ref)[None, None, :] * np.asarray(stretch)[:, :, None]
    expect = np.asarray(sw_down)[:, :, None] * frac / (rho_0 * c_sw * dz_live)

    got = shortwave_penetration_tendency(
        sw_down, dz_ref, z_half_ref, J, z_half_stretch=stretch)
    assert jnp.allclose(got, jnp.asarray(expect), rtol=1e-12, atol=1e-18)


def test_z_half_stretch_preserves_column_conservation():
    """Column heat closure must still hold under a live (non-unity) stretch
    -- the interface-flux telescoping argument does not depend on the
    ladder being static."""
    from legoesm.ocean.eos import rho_0, c_sw

    nx, ny, nlev, H = 2, 2, 8, 300.0
    sw_down = jnp.full((nx, ny), 220.0)
    dz_ref = jnp.full((nlev,), H / nlev)
    z_half_ref = -jnp.linspace(0.0, H, nlev + 1)
    J = jnp.ones((nx, ny))
    stretch = jnp.array([[1.05, 0.95], [1.2, 0.8]])

    dT_dt = shortwave_penetration_tendency(
        sw_down, dz_ref, z_half_ref, J, z_half_stretch=stretch)
    dz_live = dz_ref[None, None, :] * stretch[:, :, None]
    col_heating = jnp.sum(dT_dt * rho_0 * c_sw * dz_live, axis=-1)
    assert jnp.allclose(col_heating, sw_down, rtol=1e-12)


def _partial_cell_columns():
    """Two columns on a 5 x 10 m reference ladder: a deep one (all wet) and a
    partial-cell one whose seabed sits at 25 m (third cell 5 m, then dry)."""
    import numpy as np
    dz_ref = jnp.full(5, 10.0)
    z_half_ref = -jnp.concatenate([jnp.zeros(1), jnp.cumsum(dz_ref)])
    dz_live = jnp.asarray(np.array([[10.0, 10.0, 10.0, 10.0, 10.0],
                                    [10.0, 10.0, 5.0, 0.0, 0.0]]))
    return dz_ref, z_half_ref, dz_live


def test_live_geometry_column_integral_equals_absorbed_sw():
    """With dz_live the column integral of rho*c*dT*dz is exactly the absorbed
    shortwave, and the seabed remainder lands in the deepest WET cell."""
    import numpy as np
    from legoesm.ocean.eos import rho_0, c_sw
    dz_ref, z_half_ref, dz_live = _partial_cell_columns()
    sw = jnp.array([200.0, 150.0])
    dT = shortwave_penetration_tendency(
        sw, dz_ref, z_half_ref, jnp.ones(2), ShortwavePenetrationConfig(),
        rho_0=rho_0, c_sw=c_sw, dz_live=dz_live)
    col = np.asarray((rho_0 * c_sw * dT * dz_live).sum(axis=-1))
    np.testing.assert_allclose(col, [200.0, 150.0], rtol=1e-12)
    # no deposition below the seabed of the partial-cell column
    assert np.all(np.asarray(dT)[1, 3:] == 0.0)
    # the deepest wet cell also carries the light that reached the seabed
    p = JERLOV_TYPES[ShortwavePenetrationConfig().water_type]
    I = lambda z: p.R * np.exp(-z / p.zeta1) + (1 - p.R) * np.exp(-z / p.zeta2)
    np.testing.assert_allclose(
        float(dT[1, 2]) * rho_0 * c_sw * 5.0, 150.0 * I(20.0), rtol=1e-12)


def test_live_geometry_rejects_both_stretch_and_dz_live():
    dz_ref, z_half_ref, dz_live = _partial_cell_columns()
    with pytest.raises(ValueError, match="either dz_live or z_half_stretch"):
        shortwave_penetration_tendency(
            jnp.ones(2), dz_ref, z_half_ref, jnp.ones(2),
            dz_live=dz_live, z_half_stretch=jnp.ones(2))
