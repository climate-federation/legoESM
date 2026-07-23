"""Direct unit tests for ``legoesm.diagnostics.column_integrals``.

The slopbuster audit (2026-05-13, Pass 2) flagged this module as
indirectly tested only — 6 src references, 0 direct tests.
Plotters and CLAUDE.md explicitly require this module as the canonical
column-water-vapor helper (no inlined ``sum(q*p_s*dsigma)/g`` allowed),
so it must have its own test.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.diagnostics.column_integrals import (
    column_d_ext_field,
    column_mass_weighted_mean,
    column_water_vapor,
)


class TestColumnWaterVapor:
    def test_constant_humidity_column(self):
        """Uniform q_v=1e-2 over a single-column standard atmosphere.

        CWV = q_v_const * p_s / g for a column with Σ dsigma = 1.
        With p_s = 1e5 Pa, q_v = 1e-2: CWV = 1e3 / 9.80616 ≈ 101.97 kg/m^2.
        """
        nlev = 20
        dsigma = jnp.full((nlev,), 1.0 / nlev)
        q_v = jnp.full((nlev,), 1.0e-2)
        p_s = jnp.array(1.0e5)
        cwv = column_water_vapor(q_v, p_s, dsigma)
        expected = 1.0e-2 * 1.0e5 / constants.g
        assert jnp.allclose(cwv, expected, rtol=1e-6)

    def test_zero_humidity_yields_zero(self):
        nlev = 10
        dsigma = jnp.full((nlev,), 1.0 / nlev)
        q_v = jnp.zeros((nlev,))
        p_s = jnp.array(1.0e5)
        cwv = column_water_vapor(q_v, p_s, dsigma)
        assert jnp.allclose(cwv, 0.0)

    def test_horizontal_broadcast(self):
        """Leading dims (nlat, nlon) must broadcast against the level axis."""
        nlat, nlon, nlev = 2, 3, 4
        dsigma = jnp.full((nlev,), 1.0 / nlev)
        q_v = jnp.full((nlat, nlon, nlev), 5.0e-3)
        p_s = jnp.full((nlat, nlon), 1.0e5)
        cwv = column_water_vapor(q_v, p_s, dsigma)
        assert cwv.shape == (nlat, nlon)
        expected = 5.0e-3 * 1.0e5 / constants.g
        assert jnp.allclose(cwv, expected, rtol=1e-6)

    def test_nonuniform_dsigma_partitions(self):
        """If dsigma sums to 1 the CWV is independent of the partition."""
        q_v_const = 2.0e-2
        p_s = jnp.array(1.0e5)
        cwv_a = column_water_vapor(
            jnp.full((4,), q_v_const), p_s, jnp.array([0.1, 0.2, 0.3, 0.4])
        )
        cwv_b = column_water_vapor(
            jnp.full((4,), q_v_const), p_s, jnp.array([0.25, 0.25, 0.25, 0.25])
        )
        assert jnp.allclose(cwv_a, cwv_b, rtol=1e-12)


class TestColumnMassWeightedMean:
    def test_uniform_field_returns_field_value(self):
        """The mass-weighted mean of a uniform field equals that field value."""
        field = jnp.full((5,), 273.0)
        mass = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = column_mass_weighted_mean(field, mass)
        assert jnp.allclose(out, 273.0)

    def test_zero_mass_returns_zero(self):
        field = jnp.array([1.0, 2.0, 3.0])
        mass = jnp.array([0.0, 0.0, 0.0])
        out = column_mass_weighted_mean(field, mass)
        assert jnp.allclose(out, 0.0)

    def test_weighted_average(self):
        field = jnp.array([10.0, 20.0])
        mass = jnp.array([1.0, 3.0])
        # (10*1 + 20*3) / (1 + 3) = 70 / 4 = 17.5
        out = column_mass_weighted_mean(field, mass)
        assert jnp.allclose(out, 17.5)

    def test_horizontal_broadcast(self):
        field = jnp.ones((3, 4, 5))
        mass = jnp.ones((3, 4, 5))
        out = column_mass_weighted_mean(field, mass)
        assert out.shape == (3, 4)
        assert jnp.allclose(out, 1.0)


class TestColumnDExtField:
    def test_returns_zero_when_disabled(self):
        delp = jnp.ones((4, 4, 5))
        vt = jnp.ones((4, 4, 5))
        out = column_d_ext_field(vt, delp, d_ext=0.0, da_min_c=1.0)
        assert out.shape == (4, 4)
        assert jnp.allclose(out, 0.0)

    def test_returns_scaled_when_enabled(self):
        delp = jnp.ones((2, 2, 3))
        vt = jnp.ones((2, 2, 3))
        out = column_d_ext_field(vt, delp, d_ext=0.02, da_min_c=1e9)
        # column_mean_vt = 1.0 → out = 0.02 * 1e9 * 1.0
        assert out.shape == (2, 2)
        assert jnp.allclose(out, 0.02 * 1e9)

    def test_negative_d_ext_disabled(self):
        delp = jnp.ones((3, 4))
        vt = jnp.ones((3, 4))
        out = column_d_ext_field(vt, delp, d_ext=-0.5, da_min_c=1.0)
        assert jnp.allclose(out, 0.0)


# ---------------------------------------------------------------------------
# Regression check: column_water_vapor matches the canonical formula.
# ---------------------------------------------------------------------------


def test_module_uses_constants_not_literal_g():
    """``column_water_vapor`` must divide by ``constants.g``.

    CLAUDE.md forbids re-deriving column integrals with a hardcoded
    9.80616 literal; the canonical helper must reach for the
    centralised value.  This test pins that contract so a future
    accidental refactor that swaps in a literal is caught.
    """
    import inspect
    from legoesm.diagnostics.column_integrals import column_mass_integral
    # column_water_vapor now delegates to the shared column_mass_integral
    # (2026-07-22 process-ledger extraction) — the contract moves with it.
    src = inspect.getsource(column_mass_integral)
    assert "constants.g" in src, "column_mass_integral must use constants.g"
    src_cwv = inspect.getsource(column_water_vapor)
    assert "column_mass_integral" in src_cwv, (
        "column_water_vapor must delegate to the shared helper")
    # No hardcoded gravity values in either
    for s in (src, src_cwv):
        assert "9.80616" not in s  # const-ok: ratchet ASSERTS the literal is absent
        assert "9.81" not in s  # const-ok: ratchet ASSERTS the literal is absent
